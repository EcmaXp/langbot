from __future__ import annotations

import asyncio
import functools
import logging
import tempfile
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, cast

import hikari
from async_lru import alru_cache
from pydantic_ai import Agent, BinaryContent, ImageUrl
from pydantic_ai.builtin_tools import WebSearchTool
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.anthropic import AnthropicModelSettings
from pydantic_ai.settings import ModelSettings

from .attachment import AttachmentGroup, GPTImageAttachment, TextAttachment
from .errors import (
    EmptyResponseError,
    LangBotError,
    NotTextChannelError,
    ResponseTruncatedError,
    TokenLimitExceededError,
    TooManyAttachmentsError,
    UnknownRoleError,
)
from .options import ImageQuality, Settings, policy

INPUT_TOKEN_LIMIT = policy.input_token_limit
OUTPUT_TOKEN_LIMIT = policy.output_token_limit
MESSAGE_FETCH_LIMIT = policy.message_fetch_limit
TOKEN_MARKER_ATTR = "__pre_calc_tokens"

DISCORD_SYSTEM_PROMPT = """\
You are an AI assistant deployed as a Discord bot. Users summon you by mentioning you with @<bot>; the mention is stripped before you see the message.

Discord context:
- Conversation history is reconstructed from Discord's native reply chain. Different turns may come from different authors; the latest turn is the current speaker. You have no memory beyond this chain.
- Messages longer than ~2000 characters are auto-converted to a `.md` file attachment, which is a degraded reading experience. Default to concise replies; go long only when the user asks for depth, full code, or a document.
- Use Discord-flavored Markdown only: **bold**, *italic*, __underline__, ~~strike~~, `inline code`, fenced code blocks with a language hint (e.g. ```python), > quote, # / ## / ### headings, - lists, [text](url). No HTML, no LaTeX, no math via $...$ — those render as raw text in Discord.
- You may receive image attachments on vision-capable models, and short text files inlined into the user's turn.

Style: chat tone, direct, no boilerplate preamble. Match the user's language (Korean, English, etc.). For code, prefer fenced blocks with language hints over inline.\
"""


@functools.cache
def _get_agent(model: str) -> Agent[None, str]:
    return Agent(model, builtin_tools=[WebSearchTool()])


def _build_model_settings(model: str, max_tokens: int) -> ModelSettings:
    if model.startswith("anthropic:"):
        return AnthropicModelSettings(
            max_tokens=max_tokens,
            thinking="high",
            anthropic_cache_instructions=True,
            anthropic_cache_messages=True,
        )
    return ModelSettings(max_tokens=max_tokens, thinking="high")


UserPrompt = str | list[str | ImageUrl | BinaryContent]


def _to_user_prompt(content: Any) -> UserPrompt:
    if isinstance(content, str):
        return content
    parts: list[str | ImageUrl | BinaryContent] = []
    for item in content:
        if not isinstance(item, (str, ImageUrl, BinaryContent)):
            raise TypeError(f"Unsupported content item type: {type(item).__name__}")
        parts.append(item)
    return parts


def _split_history(
    history: list[dict[str, Any]],
) -> tuple[UserPrompt, list[ModelMessage], str]:
    """Convert internal OpenAI-style dicts into Pydantic AI inputs.

    Returns (user_prompt, message_history, instructions). The trailing user
    turn becomes user_prompt; instructions always start with the Discord
    baseline, with any user-injected `[SYSTEM]` entries appended after
    (Pydantic AI does not carry a system role in message_history).
    """
    if not history:
        raise ValueError("history must not be empty")

    last = history[-1]
    if last.get("role") != "user":
        raise ValueError(
            f"history must end with a user turn, got role={last.get('role')!r}"
        )

    instructions_parts: list[str] = [DISCORD_SYSTEM_PROMPT]
    message_history: list[ModelMessage] = []
    for msg in history[:-1]:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            if isinstance(content, str):
                instructions_parts.append(content)
            else:
                for item in content or []:
                    if isinstance(item, str):
                        instructions_parts.append(item)
        elif role == "user":
            message_history.append(
                ModelRequest(parts=[UserPromptPart(content=_to_user_prompt(content))])
            )
        elif role == "assistant":
            text = (
                content
                if isinstance(content, str)
                else "".join(item for item in (content or []) if isinstance(item, str))
            )
            message_history.append(ModelResponse(parts=[TextPart(content=text)]))
        else:
            raise ValueError(f"Unknown role in history: {role!r}")

    instructions = "\n\n".join(instructions_parts)
    return _to_user_prompt(last.get("content")), message_history, instructions


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

        current_tokens = self.get_tokens()
        if current_tokens > INPUT_TOKEN_LIMIT:
            raise TokenLimitExceededError(
                value=current_tokens,
                limit=INPUT_TOKEN_LIMIT,
            )

        prompt, message_history, instructions = _split_history(self.history)
        result = await _get_agent(self.model).run(
            user_prompt=prompt,
            message_history=message_history,
            instructions=instructions,
            model_settings=_build_model_settings(self.model, max_tokens),
        )

        output = result.output
        if not output:
            last = result.all_messages()[-1]
            finish_reason = (
                last.finish_reason
                if isinstance(last, ModelResponse) and last.finish_reason
                else "unknown"
            )
            if finish_reason == "length":
                raise ResponseTruncatedError(finish_reason=finish_reason)
            raise EmptyResponseError(finish_reason=finish_reason)

        self.last_result = result

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
                total_tokens = 0
                for item in message["content"]:
                    if isinstance(item, str):
                        total_tokens += get_text_len(item)
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

    @cached_property
    def model_name(self):
        return self.settings.chat_model

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
        await asyncio.sleep(5)
        await self.update_presence()

    async def on_message(
        self, event: hikari.MessageCreateEvent | hikari.MessageUpdateEvent
    ):
        if not event.is_human:
            return

        if self.bot_id not in event.message.user_mentions_ids:
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
                answer = await chat.ask(max_tokens=OUTPUT_TOKEN_LIMIT)
                await self.reply(message, answer)

            if hasattr(chat, "last_result"):
                usage = chat.last_result.usage()
                self.state["total_tokens"] += usage.input_tokens + usage.output_tokens

        except LangBotError as e:
            logging.exception("Error in chatgpt")
            await self._send_error_message(
                message,
                f"{e:discord}",
            )
        except Exception as e:
            logging.exception("Error in chatgpt")
            await self._send_error_message(
                message,
                f":warning: **{type(e).__name__}** - {e}",
            )

        await self.update_presence()

    async def _send_error_message(self, message: hikari.Message, error_msg: str):
        try:
            await self.reply(message, error_msg)
        except hikari.ForbiddenError:
            logging.error(
                f"Cannot send error message - missing permissions in channel {message.channel_id}"
            )
        except Exception as reply_error:
            logging.error(f"Failed to send error message: {reply_error}")

    async def preprocessing_chat(self, message: hikari.Message, chat: Chat):
        title = f"{message.author} @ {datetime.now().replace(microsecond=0)}"

        before_tokens = chat.get_tokens()
        print(f"{title}: Requesting {before_tokens} tokens")

    async def update_presence(self):
        chat_model_name = self.settings.chat_model
        total_tokens = self.state["total_tokens"]
        await self.bot.update_presence(
            activity=hikari.Activity(
                type=hikari.ActivityType.PLAYING,
                name=chat_model_name,
                state=f"{total_tokens:,} tokens",
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
                    raise UnknownRoleError(role=role)

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
                for item in content:
                    if isinstance(item, str):
                        total_tokens += get_text_len(item)

                total_tokens += sum(img.tokens for img in group.images)

                if total_tokens > INPUT_TOKEN_LIMIT:
                    raise TokenLimitExceededError(
                        value=total_tokens,
                        limit=INPUT_TOKEN_LIMIT,
                    )
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
                if total_tokens > INPUT_TOKEN_LIMIT:
                    raise TokenLimitExceededError(
                        value=total_tokens,
                        limit=INPUT_TOKEN_LIMIT,
                    )
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
            raise NotTextChannelError(channel_id=message.channel_id)

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
            raise TooManyAttachmentsError(
                value=len(message.attachments),
                limit=policy.max_attachment_count,
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
            raise TooManyAttachmentsError(
                value=len(image_attachments),
                limit=policy.max_image_attachment_count,
            )

        texts = list(map(TextAttachment, text_attachments))
        images = list(map(GPTImageAttachment, image_attachments))

        return AttachmentGroup(texts=texts, images=images)
