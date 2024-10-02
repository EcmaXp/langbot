from __future__ import annotations

import logging
import tempfile
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from functools import cache, cached_property
from pathlib import Path
from typing import cast

import hikari
import tiktoken
import tokencost
from async_lru import alru_cache
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI

from .attachment import AttachmentGroup, GPTImageAttachment, TextAttachment
from .options import ImageQuality, Settings, openai_settings, policy, settings

MAX_TOKENS = policy.token_limit
MESSAGE_FETCH_LIMIT = policy.message_fetch_limit
TOKEN_MARKER_ATTR = "__pre_calc_tokens"


def get_chat_model(chat_model_name: str) -> BaseChatModel:
    get_chat_model_cost(chat_model_name)  # check model costs
    provider, model_name = chat_model_name.split(":")
    if provider == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=1 if model_name.startswith("o") else 0.7,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_name)
    elif provider == "google-genai":
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
            HarmBlockThreshold,
            HarmCategory,
        )

        return ChatGoogleGenerativeAI(
            model=model_name,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )
    else:
        raise ValueError("Unknown provider")


def get_chat_model_cost(chat_model_name: str) -> dict:
    _, model = chat_model_name.split(":")
    return tokencost.TOKEN_COSTS[model]


def get_text_tokens(
    chat_model: BaseChatModel,
    text: str,
    *,
    tiktoken_model: str | None = None,
) -> int:
    if not tiktoken_model:
        if isinstance(chat_model, ChatOpenAI):
            tiktoken_model = chat_model.model_name
        elif type(chat_model).__name__ == "ChatGoogleGenerativeAI":
            tiktoken_model = "gpt-4-turbo-preview"  # fallback

    if tiktoken_model:
        return _get_num_text_tokens_openai(tiktoken_model, text)
    else:
        return chat_model.get_num_tokens(text)


@cache
def _get_num_text_tokens_openai(model_name: str, text: str) -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))


class Chat:
    def __init__(
        self,
        history: list[BaseMessage],
        chat_model: BaseChatModel,
        settings: Settings,
    ):
        self.history = history
        self.chat_model = chat_model
        self.settings = settings

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
        text: str | None = None,
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
        tokens = 3
        for message in self.history:
            tokens += 5
            if hasattr(message, TOKEN_MARKER_ATTR):
                tokens += getattr(message, TOKEN_MARKER_ATTR)
            elif isinstance(message.content, str):
                tokens += get_text_tokens(
                    self.chat_model,
                    message.content,
                    tiktoken_model=self.settings.tiktoken_model,
                )
            elif isinstance(message.content, dict):
                # TODO: implement this
                raise NotImplementedError(
                    "Calculating tokens in multimedia responds is not implemented."
                )

        return tokens

    def copy(self):
        return Chat(deepcopy(self.history), self.chat_model, self.settings)


@alru_cache(maxsize=1024, typed=True)
async def get_summary(message: BaseMessage):
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

    img_count = 0
    if isinstance(message.content, str):
        text = message.content
    else:
        pos, texts = 0, []
        for _ in range(len(message.content)):
            item = message.content[pos]
            if item["type"] == "text":
                texts.append(item["text"])
                del message.content[pos]
                pos -= 1
            elif item["type"] == "image_url":
                img_count += 1
                item["image_url"]["detail"] = "low"
            pos += 1
        text = "\n".join(texts)

    old_num_tokens = getattr(message, TOKEN_MARKER_ATTR)
    new_num_tokens = 0

    if text:
        summary = await chain.ainvoke({"input": text})
        new_num_tokens += get_text_tokens(
            chat_model,
            summary,
            tiktoken_model=settings.tiktoken_model,
        )

        if isinstance(message.content, str):
            message.content = summary
        elif summary:
            message.content.append({"type": "text", "text": summary})

    new_num_tokens += img_count * openai_settings.OPENAI_DEFAULT_IMAGE_COST

    print(f"Summarized: {old_num_tokens} -> {new_num_tokens} tokens")
    setattr(message, TOKEN_MARKER_ATTR, new_num_tokens)


class ChatGPT:
    # noinspection PyShadowingNames
    def __init__(self, bot: hikari.GatewayBot, settings: Settings):
        self.bot = bot
        self.state = {}
        self.settings = settings
        self.cached_channels = {}
        self.cached_messages = defaultdict(dict)
        self.cached_reply_ids: dict[int, set] = defaultdict(set)
        self.state.setdefault("total_tokens", 0)
        self.state.setdefault("total_cost", 0)

    @cached_property
    def chat_model(self):
        return get_chat_model(self.settings.chat_model)

    @cached_property
    def chat_model_cost(self):
        return get_chat_model_cost(self.settings.chat_model)

    @property
    def prompt_cost_per_token(self):
        return self.chat_model_cost["input_cost_per_token"]

    @property
    def completion_cost_per_token(self):
        return self.chat_model_cost["output_cost_per_token"]

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

                self.state["total_tokens"] += cb.prompt_tokens + cb.completion_tokens
                self.state["total_cost"] += (
                    cb.prompt_tokens * self.prompt_cost_per_token
                    + cb.completion_tokens * self.completion_cost_per_token
                )
        except Exception as e:
            logging.exception("Error in chatgpt")
            await self.reply(message, f":warning: **{type(e).__name__}**: {e}")

        await self.update_presence()

    async def preprocessing_chat(self, message: hikari.Message, chat: Chat):
        title = f"{message.author} @ {datetime.now().replace(microsecond=0)}"

        before_tokens = chat.get_tokens()
        print(f"{title}: Requesting {before_tokens} tokens")

    async def update_presence(self):
        chat_model_name = self.settings.chat_model
        total_tokens = self.state["total_tokens"]
        total_cost = self.state["total_cost"]
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
        fetched = await self.fetch_all_messages(message, MESSAGE_FETCH_LIMIT)
        for msg_index, message in enumerate(fetched):
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

            group: AttachmentGroup | None = None
            if message.attachments:
                group = await self.fetch_attachments(message)

                if (
                    len(group.images) > policy.image_count_tolerance
                    or msg_index < len(fetched) - policy.image_message_tolerance
                ):
                    for i in range(len(group.images)):
                        group.images[i] = GPTImageAttachment(
                            group.images[i].attachment,
                            quality=ImageQuality.Low,
                            strict=True,
                        )

            msg_type = {
                "system": SystemMessage,
                "user": HumanMessage,
                "assistant": AIMessage,
            }
            if role not in msg_type.keys():
                continue
            content = await group.export(text) if group else None
            if content:
                total_tokens = 0
                # Text tokens
                for item in content:
                    if item["type"] == "text":
                        total_tokens += get_text_tokens(
                            self.chat_model,
                            item["text"],
                            tiktoken_model=self.settings.tiktoken_model,
                        )

                # Image tokens
                total_tokens += sum(img.tokens for img in group.images)

                if total_tokens > MAX_TOKENS:
                    raise ValueError("Attachments are too large to upload.")
                else:
                    msg = msg_type[role](content=content)
                    setattr(msg, TOKEN_MARKER_ATTR, total_tokens)
            else:
                if not text:
                    text = "-" if messages else "hello"
                total_tokens = get_text_tokens(
                    self.chat_model,
                    text,
                    tiktoken_model=self.settings.tiktoken_model,
                )
                if total_tokens > MAX_TOKENS:
                    raise ValueError("Text is too long to read.")
                msg = msg_type[role](text)
                setattr(msg, TOKEN_MARKER_ATTR, total_tokens)

            if role == "system":
                messages.insert(0, msg)
            else:
                messages.append(msg)

        return Chat(messages, self.chat_model, self.settings)

    async def fetch_all_messages(
        self,
        message: hikari.Message,
        limit: int,
    ) -> list[hikari.Message]:
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
    async def fetch_attachments(self, message: hikari.Message) -> AttachmentGroup:
        if len(message.attachments) > policy.max_attachment_count:
            raise ValueError(
                f"The total number of attachments cannot exceed {policy.max_attachment_count}."
            )

        text_attachments = [
            attachment
            for attachment in message.attachments
            if attachment.filename.endswith(policy.allowed_text_extensions)
        ]

        _, model_name = self.settings.chat_model.split(":")
        if model_name in policy.allowed_image_models:
            image_attachments = [
                attachment
                for attachment in message.attachments
                if attachment.filename.endswith(policy.allowed_image_extensions)
            ]
        else:
            image_attachments = []

        if len(image_attachments) > policy.max_image_attachment_count:
            raise ValueError(
                f"The number of image attachments cannot exceed {policy.max_image_attachment_count}."
            )

        texts = list(map(TextAttachment, text_attachments))
        images = list(map(GPTImageAttachment, image_attachments))

        return AttachmentGroup(texts=texts, images=images)
