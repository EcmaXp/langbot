from __future__ import annotations

import base64
import math
from abc import ABCMeta, abstractmethod
from io import BytesIO

from PIL import Image
from hikari import Attachment

from .options import openai_settings, attachment_policy
from .utils import humanize_tuple


class AttachmentGroup:
    def __init__(
        self,
        *,
        texts: list[TextAttachment],
        images: list[GPTImageAttachment],
    ):
        self.texts = texts
        self.images = images

    async def export(self, text_content: str) -> list[dict]:
        content = []

        text = text_content
        if len(self.texts) == 0:
            pass
        elif len(self.texts) == 1:
            text += "\n\n" + await self.texts[0].digest()
        else:
            for txt in self.texts:
                text += f"\n\n{txt.attachment.filename}:\n" + await txt.digest()

        if text:
            content.append({"type": "text", "text": text})

        for img in self.images:
            content.append(await img.digest())

        return content


class DiscordAttachment(metaclass=ABCMeta):
    def __init__(self, attachment: Attachment):
        self._attachment = attachment

    @property
    def attachment(self):
        return self._attachment

    @property
    def size(self):
        return self.attachment.size

    @abstractmethod
    def check_error(self) -> Exception | None:
        pass

    @abstractmethod
    async def digest(self):
        pass


class TextAttachment(DiscordAttachment):
    def check_error(self):
        att = self.attachment
        if not att.filename.endswith(attachment_policy.allowed_text_extensions):
            return ValueError(
                f"Only supported extensions in text files are {humanize_tuple(attachment_policy.allowed_text_extensions)}."
            )
        elif att.size > attachment_policy.max_text_file_size:
            return ValueError(
                f"File is too large to upload. It must be at most {attachment_policy.max_text_file_size // 1024} KB."
            )

        return None

    async def digest(self) -> str:
        err = self.check_error()
        if err is not None:
            raise err

        raw_data = await self.attachment.read()
        try:
            result = raw_data.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("Text attachment should be encoded to UTF-8.")

        return result


class GPTImageAttachment(DiscordAttachment):
    def __init__(
        self,
        attachment: Attachment,
        *,
        quality: str = attachment_policy.default_image_quality,
        strict: bool = False,
    ):
        super().__init__(attachment)

        self._strict = strict
        self._quality = self.determine_image_quality(attachment, quality, strict)
        self._tokens = self.calc_openai_tokens(
            attachment.width, attachment.height, quality=self._quality
        )

    @classmethod
    def determine_image_quality(
        cls, attachment: Attachment, quality: str, strict: bool
    ):
        if strict:
            return quality

        if attachment_policy.strict_image_quality:
            return attachment_policy.strict_image_quality

        width, height = attachment.width, attachment.height
        cost = cls.calc_openai_tokens(width, height, quality=quality)

        if max(width, height) > attachment_policy.low_quality_threshold:
            return "low"
        if cost > attachment_policy.low_quality_token_threshold:
            return "low"

        return quality

    @classmethod
    def calc_openai_tokens(cls, width, height, *, quality: str):
        if width == 0 and height == 0:
            return 0
        elif quality == "low":
            return openai_settings.OPENAI_DEFAULT_IMAGE_COST
        elif quality == "high":
            cost = openai_settings.OPENAI_DEFAULT_IMAGE_COST
            if max(width, height) <= 512:
                return cost + openai_settings.OPENAI_IMAGE_BLOCK_COST

            ratio, ratio_inv = width / height, height / width

            if max(width, height) > 2048:
                if ratio > 1:
                    width = 2048
                    height = math.floor(2048 * ratio_inv)
                elif ratio < 1:
                    width = math.floor(2048 * ratio)
                    height = 2048
                else:
                    width = height = 2048

            if min(width, height) <= 768:
                cost += (
                    openai_settings.OPENAI_IMAGE_BLOCK_COST
                    * math.ceil(width / 512)
                    * math.ceil(width / 512)
                )
            elif ratio > 1:
                cost += (
                    openai_settings.OPENAI_IMAGE_BLOCK_COST
                    * 2
                    * math.ceil(math.floor(768 * ratio) / 512)
                )
            elif ratio < 1:
                cost += (
                    openai_settings.OPENAI_IMAGE_BLOCK_COST
                    * 2
                    * math.ceil(math.floor(768 * ratio_inv) / 512)
                )
            else:
                cost += openai_settings.OPENAI_IMAGE_BLOCK_COST * 4

            return cost
        elif quality == "auto":
            if width <= 512 and height <= 512:
                return openai_settings.OPENAI_DEFAULT_IMAGE_COST
            else:
                return cls.calc_openai_tokens(width, height, quality="high")
        else:
            return math.inf

    @property
    def quality(self):
        return self._quality

    @property
    def strict(self):
        return self._strict

    @property
    def tokens(self):
        return self._tokens

    def check_error(self):
        att = self.attachment
        if not att.filename.endswith(attachment_policy.allowed_image_extensions):
            return ValueError(
                f"Only supported extensions in image files are {humanize_tuple(attachment_policy.allowed_image_extensions)}."
            )
        elif att.size > attachment_policy.max_image_file_size:
            return ValueError(
                f"Each image file's size cannot exceed {attachment_policy.max_image_file_size / (1024 ** 2):.1f } MB."
            )
        elif att.width is None or att.height is None:
            return ValueError(f"Image not found: {att.filename}")
        elif att.width <= 0 or att.height <= 0:
            return ValueError(
                f"Both width and height of an image '{att.filename}' must be positive."
            )
        elif (
            att.width > attachment_policy.max_image_width
            or att.height > attachment_policy.max_image_height
        ):
            msg = f"The image size cannot be over "
            msg += f"{attachment_policy.max_image_file_size} x {attachment_policy.max_image_height} px."
            msg += f" ({att.filename}: {att.width}x{att.height})"
            return ValueError(msg)

        return None

    async def digest(self) -> dict:
        err = self.check_error()
        if err is not None:
            raise err

        # Discord CDN URL is blocked by bot-scrapping so User-Agent opener hack is required.
        # It doesn't seem like OpenAI image scrapper supports this hack.
        # So we supply base64 encoded image (* token cost is the same as URL).
        if attachment_policy.discord_url_allowed:
            content = {
                "type": "image_url",
                "image_url": {"url": self.attachment.url, "detail": self.quality},
            }
            return content

        raw_data = await self.attachment.read()
        image = Image.open(BytesIO(raw_data))

        if hasattr(image, "is_animated"):
            if getattr(image, "is_animated"):
                image.seek(0)  # Fixed to the first frame

        buffer = BytesIO()
        image.save(buffer, format=image.format)
        encoded_image = base64.b64encode(buffer.getvalue())

        img_format = image.format.lower()
        if img_format == "jpg":
            img_format = "jpeg"

        content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{img_format};base64,{encoded_image}",
                "detail": self.quality,
            },
        }

        return content