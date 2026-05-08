from __future__ import annotations

import math
from abc import ABCMeta, abstractmethod
from io import BytesIO

from hikari import Attachment
from PIL import Image
from pydantic_ai import BinaryContent, ImageUrl

from .options import ImageQuality, openai_settings, policy
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

    async def export(self, text_content: str) -> list[str | ImageUrl | BinaryContent]:
        content: list[str | ImageUrl | BinaryContent] = []

        text = text_content
        if not self.texts:
            pass
        elif len(self.texts) == 1:
            text += "\n\n" + await self.texts[0].digest()
        else:
            for txt in self.texts:
                text += f"\n\n{txt.attachment.filename}:\n" + await txt.digest()

        if text:
            content.append(text)

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
    def __init__(self, attachment: Attachment):
        super().__init__(attachment)
        self.check_error()

    def check_error(self):
        att = self.attachment
        if not att.filename.endswith(policy.allowed_text_extensions):
            raise ValueError(
                f"Only supported extensions in text files are {humanize_tuple(policy.allowed_text_extensions)}."
            )
        elif att.size > policy.max_text_file_size:
            raise ValueError(
                f"File is too large to upload. It must be at most {policy.max_text_file_size // 1024} KB."
            )

    async def digest(self) -> str:
        raw_data = await self.attachment.read()
        try:
            result = raw_data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError("Text attachment should be encoded to UTF-8.") from e

        return result


class GPTImageAttachment(DiscordAttachment):
    def __init__(
        self,
        attachment: Attachment,
        *,
        quality: ImageQuality = policy.default_image_quality,
        strict: bool = False,
    ):
        super().__init__(attachment)

        self.strict = strict
        self.quality = self.determine_image_quality(attachment, quality, strict)
        self.tokens = self.calc_openai_tokens(
            attachment.width, attachment.height, quality=self.quality
        )

        self.check_error()

    @classmethod
    def determine_image_quality(
        cls,
        attachment: Attachment,
        quality: ImageQuality,
        strict: bool,
    ):
        if strict:
            return quality

        if policy.strict_image_quality:
            return policy.strict_image_quality

        width, height = attachment.width, attachment.height
        cost = cls.calc_openai_tokens(width, height, quality)

        if max(width, height) > policy.low_quality_threshold:
            return ImageQuality.Low
        if cost > policy.low_quality_token_threshold:
            return ImageQuality.Low

        return quality

    @classmethod
    def calc_openai_tokens(cls, width, height, quality: ImageQuality):
        if width == 0 and height == 0:
            return 0
        elif quality == ImageQuality.Low:
            return openai_settings.OPENAI_DEFAULT_IMAGE_COST
        elif quality == ImageQuality.High:
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
                    * math.ceil(height / 512)
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
        elif quality == ImageQuality.Auto:
            if width <= 512 and height <= 512:
                return openai_settings.OPENAI_DEFAULT_IMAGE_COST
            else:
                return cls.calc_openai_tokens(width, height, quality=ImageQuality.High)
        else:
            return math.inf

    def check_error(self):
        att = self.attachment
        if not att.filename.endswith(policy.allowed_image_extensions):
            raise ValueError(
                f"Only supported extensions in image files are {humanize_tuple(policy.allowed_image_extensions)}."
            )
        elif att.size > policy.max_image_file_size:
            raise ValueError(
                f"Each image file's size cannot exceed {policy.max_image_file_size / (1024**2):.1f } MB."
            )
        elif att.width is None or att.height is None:
            raise ValueError(f"Image not found: {att.filename}")
        elif att.width <= 0 or att.height <= 0:
            raise ValueError(
                f"Both width and height of an image '{att.filename}' must be positive."
            )
        elif att.width > policy.max_image_width or att.height > policy.max_image_height:
            msg = "The image size cannot be over "
            msg += f"{policy.max_image_file_size} x {policy.max_image_height} px."
            msg += f" ({att.filename}: {att.width}x{att.height})"
            raise ValueError(msg)

    async def digest(self) -> ImageUrl | BinaryContent:
        self.check_error()

        # Discord CDN URL is blocked by bot-scrapping so User-Agent opener hack is required.
        # It doesn't seem like OpenAI image scrapper supports this hack.
        # So we supply binary image content directly when CDN URLs are disallowed.
        if policy.discord_url_allowed:
            return ImageUrl(
                url=self.attachment.url,
                vendor_metadata={"detail": self.quality.value},
            )

        raw_data = await self.attachment.read()
        image = Image.open(BytesIO(raw_data))

        if hasattr(image, "is_animated") and image.is_animated:
            image.seek(0)  # Fixed to the first frame

        buffer = BytesIO()
        image.save(buffer, format=image.format)

        return BinaryContent(
            data=buffer.getvalue(),
            media_type=f"image/{image.format.lower()}",
            vendor_metadata={"detail": self.quality.value},
        )
