from __future__ import annotations

import logging
import tempfile
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, cast

import hikari
import litellm
import tokencost
from async_lru import alru_cache

from .attachment import AttachmentGroup, GPTImageAttachment, TextAttachment
from .options import ImageQuality, Settings, fallback, policy

MAX_TOKENS = policy.token_limit
MESSAGE_FETCH_LIMIT = policy.message_fetch_limit
TOKEN_MARKER_ATTR = "__pre_calc_tokens"

# Configure litellm
litellm.drop_params = True  # Drop unsupported params automatically
litellm.set_verbose = False  # Set to True for debugging


def get_chat_model_cost(chat_model_name: str) -> dict:
    if fallback.override_costs:
        return {
            "input_cost_per_token": fallback.input_cost_per_token,
            "output_cost_per_token": fallback.output_cost_per_token,
        }

    return tokencost.TOKEN_COSTS[chat_model_name]


def get_text_len(text: str) -> int:
    return len(text.encode("utf-8"))


class Chat:
    def __init__(
        self,
        history: list[dict[str, Any]],
        model: str,
        settings: Settings,
    ):
        self.history = history
        self.model = model
        self.settings = settings

    async def ask(
        self,
        text: str | None = None,
        *,
        max_tokens: int,
    ):
        if text is not None:
            self.history.append({"role": "user", "content": text})

        available_tokens = max_tokens - self.get_tokens()
        if available_tokens <= 0:
            raise ValueError("No available tokens, please start a new chat.")

        # Get appropriate temperature based on model
        temperature = 1.0 if "o1" in self.model else 0.7

        # Create completion using litellm
        response = await litellm.acompletion(
            model=self.model,
            messages=self.history,
            temperature=temperature,
            max_tokens=available_tokens,
        )

        # Extract the response content
        output = response.choices[0].message.content

        # Store response metadata for cost tracking
        self.last_response = response

        self.history.append({"role": "assistant", "content": output})
        return output

    def get_tokens(self) -> int:
        tokens = 3
        for message in self.history:
            tokens += 5
            if TOKEN_MARKER_ATTR in message:
                tokens += message[TOKEN_MARKER_ATTR]
            elif isinstance(message.get("content"), str):
                tokens += get_text_len(message["content"])
            elif isinstance(message.get("content"), list):
                # Handle multimodal content
                total_tokens = 0
                for item in message["content"]:
                    if item.get("type") == "text":
                        total_tokens += get_text_len(item.get("text", ""))
                tokens += total_tokens

        return tokens

    def copy(self):
        return Chat(deepcopy(self.history), self.model, self.settings)


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
    def model_name(self):
        return self.settings.chat_model

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

            # Check if last message is system message
            if chat.history and chat.history[-1].get("role") == "system":
                await self.reply(message, "[SYSTEM] System message is set.")
                return

            async with channel.trigger_typing():
                await self.preprocessing_chat(message, chat)
                answer = await chat.ask(max_tokens=8192)
                await self.reply(message, answer)

            # Update cost tracking using litellm response metadata
            if hasattr(chat, "last_response"):
                response = chat.last_response
                # Get token usage from response
                usage = response.usage
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens

                self.state["total_tokens"] += prompt_tokens + completion_tokens
                self.state["total_cost"] += (
                    prompt_tokens * self.prompt_cost_per_token
                    + completion_tokens * self.completion_cost_per_token
                )

        except Exception as e:
            logging.exception("Error in chatgpt")
            try:
                await self.reply(message, f":warning: **{type(e).__name__}**: {e}")
            except hikari.ForbiddenError:
                logging.error(
                    f"Cannot send error message - missing permissions in channel {message.channel_id}"
                )
            except Exception as reply_error:
                logging.error(f"Failed to send error message: {reply_error}")

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

        messages: list[dict] = []
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

            if role not in ["system", "user", "assistant"]:
                continue

            content = await group.export(text) if group else None
            if content:
                total_tokens = 0
                # Text tokens
                for item in content:
                    if item["type"] == "text":
                        total_tokens += get_text_len(item["text"])

                # Image tokens
                total_tokens += sum(img.tokens for img in group.images)

                if total_tokens > MAX_TOKENS:
                    raise ValueError("Attachments are too large to upload.")
                else:
                    msg = {
                        "role": role,
                        "content": content,
                        TOKEN_MARKER_ATTR: total_tokens,
                    }
            else:
                if not text:
                    text = "-" if messages else "hello"
                total_tokens = get_text_len(text)
                if total_tokens > MAX_TOKENS:
                    raise ValueError("Text is too long to read.")
                msg = {"role": role, "content": text, TOKEN_MARKER_ATTR: total_tokens}

            if role == "system":
                messages.insert(0, msg)
            else:
                messages.append(msg)

        return Chat(messages, self.model_name, self.settings)

    async def fetch_all_messages(
        self,
        message: hikari.Message,
        limit: int,
    ) -> list[hikari.Message]:
        channel = await self.fetch_channel(message.channel_id)
        if not isinstance(channel, hikari.TextableChannel):
            raise ValueError("Not a text channel")

        messages = []
        for _ in range(limit):
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

        if self.settings.chat_model in policy.allowed_image_models:
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
