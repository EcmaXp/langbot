from __future__ import annotations

import math
import base64

from hikari import Attachment
from abc import ABCMeta, abstractmethod
from typing import List
from PIL import Image
from io import BytesIO

from utils import humanize_tuple
from options import ATTACHMENT_POLICIES as GLOBAL_OPTIONS

class AttachmentGroup:
    def __init__(
        self,
        *,
        texts: list[TextAttachment],
        images: list[GPTImageAttachment]
    ):
        self.texts = texts
        self.images = images

    async def export(self, text_content: str) -> List[dict]:
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
            content.append({ "type": "text", "text": text })

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
    MAX_SIZE, ALLOWED_EXTENSIONS = map(GLOBAL_OPTIONS.get, (
        "max_text_file_size",
        "allowed_text_extensions"
    ))

    def check_error(self):
        att = self.attachment
        if not att.filename.endswith(self.ALLOWED_EXTENSIONS):
            return ValueError(f"Only supported extensions in text files are {humanize_tuple(self.ALLOWED_EXTENSIONS)}.")
        elif att.size > self.MAX_SIZE:
            return ValueError(f"File is too large to upload. It must be at most {self.MAX_SIZE // 1024} KB.")

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
    (
        MAX_WIDTH,
        MAX_HEIGHT,
        MAX_SIZE,

        LOW_QUALITY_THRESHOLD,
        LOW_QUALITY_TOKEN_THRESHOLD,

        ALLOWED_MODEL,
        ALLOWED_EXTENSIONS,

        DISCORD_URL_ALLOWED
     ) = map(GLOBAL_OPTIONS.get, (
        "max_image_width",
        "max_image_height",
        "max_image_file_size",

        "low_quality_threshold",
        "low_quality_token_threshold",

        "allowed_image_models",
        "allowed_image_extensions",

        "discord_url_allowed"
    ))

    def __init__(self,
            attachment: Attachment,
            *,
            quality: str = GLOBAL_OPTIONS["default_image_quality"],
            strict: bool = False
        ):
        super().__init__(attachment)

        self._strict = strict
        if not strict and GLOBAL_OPTIONS["strict_image_quality"] is not None:
            self._quality = GLOBAL_OPTIONS["strict_image_quality"]
        else:
            self._quality = quality
            if not strict:
                width, height = attachment.width, attachment.height
                cost = self.calc_openai_tokens(width, height, quality=quality)

                if max(width, height) > GLOBAL_OPTIONS["low_quality_threshold"]:
                    self._quality = "low"
                if cost > GLOBAL_OPTIONS["low_quality_token_threshold"]:
                    self._quality = "low"

        self._tokens = self.calc_openai_tokens(attachment.width, attachment.height, quality=self._quality)

    @staticmethod
    def calc_openai_tokens(width, height, *, quality: str):
        from options import OPENAI_DEFAULT_IMAGE_COST, OPENAI_IMAGE_BLOCK_COST
        default_cost, block_cost = OPENAI_DEFAULT_IMAGE_COST, OPENAI_IMAGE_BLOCK_COST

        if width == 0 and height == 0:
            return 0
        elif quality == "low":
            return default_cost
        elif quality == "high":
            cost = default_cost
            if max(width, height) <= 512:
                return cost + block_cost

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
                cost += block_cost * math.ceil(width / 512) * math.ceil(width / 512)
            elif ratio > 1:
                cost += block_cost * 2 * math.ceil(math.floor(768 * ratio) / 512)
            elif ratio < 1:
                cost += block_cost * 2 * math.ceil(math.floor(768 * ratio_inv) / 512)
            else:
                cost += block_cost * 4

            return cost
        elif quality == "auto":
            if width <= 512 and height <= 512:
                return default_cost
            else:
                return GPTImageAttachment.calc_openai_tokens(width, height, quality="high")
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
        if not att.filename.endswith(self.ALLOWED_EXTENSIONS):
            return ValueError(f"Only supported extensions in image files are {humanize_tuple(self.ALLOWED_EXTENSIONS)}.")
        elif att.size > self.MAX_SIZE:
            return ValueError(f"Each image file's size cannot exceed { self.MAX_SIZE / (1024 ** 2):.1f } MB.")
        elif att.width is None or att.height is None:
            return ValueError(f"Image not found: {att.filename}")
        elif att.width <= 0 or att.height <= 0:
            return ValueError(f"Both width and height of an image '{att.filename}' must be positive.")
        elif att.width > self.MAX_WIDTH or att.height > self.MAX_HEIGHT:
            msg = f"The image size cannot be over {self.MAX_SIZE} x {self.MAX_HEIGHT} px."
            msg += f" ({att.filename}: {att.width}x{att.height})"
            return ValueError(msg)

        return None

    async def digest(self) -> dict:
        err = self.check_error()
        if err is not None:
            raise err

        """
        # Discord CDN URL is blocked by bot-scrapping so User-Agent opener hack is required.
        # It doesn't seem like OpenAI image scrapper supports this hack.
        # So we supply base64 encoded image (* token cost is the same as URL).
        """
        if self.DISCORD_URL_ALLOWED:
            content = {
                "type": "image_url",
                "image_url": {
                    "url": self.attachment.url,
                    "detail": self.quality
                }
            }
            return content

        raw_data = await self.attachment.read()
        image = Image.open(BytesIO(raw_data))

        if hasattr(image, "is_animated"):
            if getattr(image, "is_animated"):
                image.seek(0) # Fixed to the first frame

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
                "detail": self.quality
            }
        }

        return content
