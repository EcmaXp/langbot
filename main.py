from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from functools import cache
from pathlib import Path
from typing import List, Optional, cast

import hikari
import tiktoken
from async_lru import alru_cache
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI

__author__ = "EcmaXp <ecmaxp@ecmaxp.kr>"
__version__ = "0.1"


if os.name != "nt":
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


def get_chat_model(chat_model_name: str) -> BaseChatModel:
    provider, chat_model_name = chat_model_name.split(":")
    if provider == "openai":
        return ChatOpenAI(model=chat_model_name)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=chat_model_name)
    elif provider == "google-genai":
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
            HarmBlockThreshold,
            HarmCategory,
        )

        return ChatGoogleGenerativeAI(
            model=chat_model_name,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )
    else:
        raise ValueError("Unknown provider")


def get_tokens(chat_model: BaseChatModel, text: str) -> int:
    if isinstance(chat_model, ChatOpenAI):
        return _get_num_tokens_openai(chat_model.model_name, text)
    elif type(chat_model).__name__ == "ChatGoogleGenerativeAI":
        return _get_num_tokens_openai("gpt-4-turbo-preview", text)  # fallback
    else:
        return chat_model.get_num_tokens(text)


@cache
def _get_num_tokens_openai(model_name: str, text: str) -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))


class Chat:
    def __init__(
        self,
        history: list[BaseMessage],
        chat_model: BaseChatModel,
    ):
        self.history = history
        self.chat_model = chat_model

    def build_chat_chain(self) -> Runnable:
        return (
            RunnablePassthrough()
            | ChatPromptTemplate.from_messages(
                [MessagesPlaceholder(variable_name="chat_history")]
            )
            | self.chat_model
            | StrOutputParser()
        )

    async def ask(
        self,
        text: Optional[str] = None,
        *,
        max_tokens: int,
    ):
        if text is not None:
            self.history.append(HumanMessage(text))

        available_tokens = max_tokens - self.get_tokens()
        if available_tokens <= 0:
            raise ValueError("No available tokens, please start a new chat.")

        chat_chain = self.build_chat_chain()
        output = await chat_chain.ainvoke({"chat_history": self.history})

        self.history.append(AIMessage(output))
        return output

    def get_tokens(self) -> int:
        return 3 + sum(
            get_tokens(self.chat_model, item.content) + 5 for item in self.history
        )

    def copy(self):
        return Chat(deepcopy(self.history), self.chat_model)

    async def compress_large_messages(
        self,
        chat_compress_threshold: int,
        message_compress_threshold: int,
    ):
        if self.get_tokens() < chat_compress_threshold:
            return

        for pos, item in enumerate(self.history[3:-3], 3):
            if isinstance(item, SystemMessage):
                continue
            elif get_tokens(self.chat_model, item.content) > message_compress_threshold:
                self.history[pos] = type(item)(await get_summary(item.content))


@alru_cache(maxsize=1024, typed=True)
async def get_summary(text: str) -> str:
    chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125")
    chain = (
        RunnablePassthrough()
        | ChatPromptTemplate.from_messages(
            [
                SystemMessage("[system] Summarize the following:"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )
        | chat_model
        | StrOutputParser()
    )

    summary = await chain.ainvoke({"input": text})

    old_num_tokens = get_tokens(chat_model, text)
    new_num_tokens = get_tokens(chat_model, summary)

    print(f"Summarized: {old_num_tokens} -> {new_num_tokens} tokens")
    return summary


class ChatGPT:
    def __init__(self, bot: hikari.GatewayBot, config: dict):
        self.bot = bot
        self.config = config
        self.chat_model = get_chat_model(config["chat_model"])
        self.cached_channels = {}
        self.cached_messages = defaultdict(dict)
        self.cached_reply_ids: dict[int, set] = defaultdict(set)
        config.setdefault("total_tokens", 0)
        config.setdefault("total_cost", 0)

    @property
    def bot_id(self):
        return self.bot.get_me().id

    async def fetch_channel(self, channel_id: hikari.Snowflakeish):
        if channel_id not in self.cached_channels:
            self.cached_channels[channel_id] = await self.bot.rest.fetch_channel(
                channel_id
            )

        return self.cached_channels[channel_id]

    async def on_ready(self, event: hikari.ShardReadyEvent):
        await self.update_presence()

    async def on_message(
        self, event: hikari.MessageCreateEvent | hikari.MessageUpdateEvent
    ):
        if not event.is_human:
            return

        if self.bot_id not in event.message.user_mentions_ids and (
            not event.message.referenced_message
            or not event.message.referenced_message.author
            or event.message.referenced_message.author.id != self.bot_id
        ):
            return

        channel = await self.fetch_channel(event.channel_id)
        if not isinstance(channel, hikari.TextableChannel):
            return

        await self.chatgpt(event.message)

    async def on_message_edit(self, event: hikari.MessageUpdateEvent):
        channel = await self.fetch_channel(event.channel_id)
        if not isinstance(channel, hikari.TextableChannel):
            return

        if event.message.id in self.cached_reply_ids:
            reply_ids = self.cached_reply_ids.pop(event.message.id, None)
            if reply_ids:
                await channel.delete_messages(reply_ids)

        await self.on_message(event)

    async def on_message_delete(self, event: hikari.MessageDeleteEvent):
        channel = await self.fetch_channel(event.channel_id)
        if not isinstance(channel, hikari.TextableChannel):
            return

        if event.message_id in self.cached_reply_ids:
            reply_ids = self.cached_reply_ids.pop(event.message_id, None)
            if reply_ids:
                await channel.delete_messages(reply_ids)

    async def chatgpt(self, message: hikari.Message):
        channel = await self.fetch_channel(message.channel_id)

        try:
            chat = await self.build_chat(message)
            if isinstance(chat.history[-1], SystemMessage) == "system":
                await self.reply(message, "[SYSTEM] System message is set.")
                return

            with get_openai_callback() as cb:
                async with channel.trigger_typing():
                    await self.preprocessing_chat(message, chat)
                    answer = await chat.ask(max_tokens=8192)
                    await self.reply(message, answer)

                self.config["total_tokens"] += cb.total_tokens or chat.get_tokens()
                self.config["total_cost"] += cb.total_cost or (
                    chat.get_tokens()
                    * Decimal(os.environ.get("LANGBOT_COST_PER_TOKEN", "0.0001"))
                )
        except Exception as e:
            logging.exception("Error in chatgpt")
            await self.reply(message, f":warning: **{type(e).__name__}**: {e}")

        await self.update_presence()

    async def preprocessing_chat(self, message: hikari.Message, chat: Chat):
        title = f"{message.author} @ {datetime.now().replace(microsecond=0)}"

        before_tokens = chat.get_tokens()
        print(f"{title}: Requesting {before_tokens} tokens")

        await chat.compress_large_messages(
            chat_compress_threshold=4096,
            message_compress_threshold=2048,
        )
        after_tokens = chat.get_tokens()
        discarded_tokens = before_tokens - after_tokens
        if discarded_tokens:
            print(
                f"{title}: Requesting {after_tokens} tokens; discarded {discarded_tokens} tokens"
            )

    async def update_presence(self):
        chat_model_name = self.config["chat_model"]
        total_tokens = self.config["total_tokens"]
        total_cost = self.config["total_cost"]
        await self.bot.update_presence(
            activity=hikari.Activity(
                type=hikari.ActivityType.PLAYING,
                name=f"{total_tokens:,} tokens = {total_cost:,.2f} $",
                state=chat_model_name,
            )
        )

    async def reply(self, message: hikari.Message, answer: str) -> hikari.Message:
        if len(answer) >= 2000:
            path = Path(tempfile.mktemp(suffix=".log"))
            try:
                path.write_text(answer, encoding="utf-8")
                answer_file = hikari.File(path, filename="message.md")
                reply = await message.respond(
                    attachment=answer_file,
                    reply=True,
                    mentions_reply=True,
                )
            finally:
                path.unlink(missing_ok=True)
        else:
            reply = await message.respond(answer, reply=True, mentions_reply=True)

        self.cached_reply_ids[message.id].add(reply.id)
        self.cached_messages[message.channel_id][reply.id] = reply
        return reply

    async def build_chat(self, message: hikari.Message) -> Chat:
        bot_mention = self.bot.get_me().mention

        messages: list[BaseMessage] = []
        for message in await self.fetch_all_messages(message, 64):
            role = "assistant" if message.author.id == self.bot_id else "user"
            text = cast(str, message.content or "")
            text = text.removeprefix(bot_mention).strip()

            if text.lower().startswith("[system]"):
                if role == "user":
                    role = "system"
                    text = text[len("[system]") :].strip()
                elif role == "assistant":
                    continue
                else:
                    raise ValueError("Unknown role")

            if message.attachments:
                text += "\n\n" + (await self.fetch_attachment(message))
                if get_tokens(self.chat_model, text) > 8192:
                    raise ValueError("Attachment too large")

            if not text:
                text = "-" if messages else "hello"

            if role == "system":
                messages.insert(0, SystemMessage(text))
            elif role == "user":
                messages.append(HumanMessage(text))
            elif role == "assistant":
                messages.append(AIMessage(text))

        return Chat(messages, self.chat_model)

    async def fetch_all_messages(
        self,
        message: hikari.Message,
        limit: int,
    ) -> List[hikari.Message]:
        channel = await self.fetch_channel(message.channel_id)
        if not isinstance(channel, hikari.TextableChannel):
            raise ValueError("Not a text channel")

        messages = []
        for i in range(limit):
            self.cached_messages[channel.id][message.id] = message
            messages.append(message)

            if message.referenced_message:
                if message.referenced_message.id in self.cached_messages[channel.id]:
                    message = self.cached_messages[channel.id][
                        message.referenced_message.id
                    ]
                else:
                    message = await channel.fetch_message(message.referenced_message.id)
            else:
                break

        return messages[::-1]

    @alru_cache(maxsize=64, typed=True, ttl=3600)
    async def fetch_attachment(self, message: hikari.Message) -> str:
        if len(message.attachments) > 1:
            raise ValueError("Too many attachments")

        for attachment in message.attachments:
            if attachment.size > 1024 * 64:
                raise ValueError("Attachment too large")
            elif not attachment.filename.endswith((".txt", ".py", ".log", ".md")):
                raise ValueError("Attachment is not text")

            content = await attachment.read()
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError:
                raise ValueError("Attachment is not text (utf-8)")


bot = hikari.GatewayBot(
    os.environ.get("DISCORD_TOKEN"),
    intents=hikari.Intents.ALL_MESSAGES | hikari.Intents.MESSAGE_CONTENT,
)


chatgpt = ChatGPT(
    bot,
    {"chat_model": os.environ["LANGBOT_CHAT_MODEL"]},
)


@bot.listen()
async def on_ready(event: hikari.ShardReadyEvent):
    await chatgpt.on_ready(event)


@bot.listen()
async def on_message(event: hikari.MessageCreateEvent):
    await chatgpt.on_message(event)


@bot.listen()
async def on_message_edit(event: hikari.MessageUpdateEvent):
    await chatgpt.on_message_edit(event)


@bot.listen()
async def on_message_delete(event: hikari.MessageDeleteEvent):
    await chatgpt.on_message_delete(event)


def main():
    bot.run()


if __name__ == "__main__":
    main()
