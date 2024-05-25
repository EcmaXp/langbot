from enum import Enum

from pydantic import BaseModel

# Constants
K, M, G = 1024, 1024**2, 1024**3
KB, MB, GB = K, M, G


class OpenAISettings(BaseModel):
    # Officially OpenAI API restricts 20MB per image
    API_MAX_IMAGE_SIZE: int = 20 * MB

    # Current price of OpenAI image vision API (Price may be changed).
    OPENAI_DEFAULT_IMAGE_COST: int = 85
    OPENAI_IMAGE_BLOCK_COST: int = 170


openai_settings = OpenAISettings()


class ImageQuality(Enum, str):
    Low = "low"
    High = "high"
    Auto = "auto"


# Policies
class Policy(BaseModel):
    token_limit: int = 8 * K
    message_fetch_limit: int = 64

    compress_threshold_per_chat: int = 4 * K
    compress_threshold_per_message: int = 2 * K

    max_attachment_count: int = 3
    max_text_attachment_count: int = 2
    max_image_attachment_count: int = 2

    # Images in last N messages hold their original quality, others are forcibly degraded to low quality
    image_message_tolerance: int = 3
    # If there are more than N images per message, all images are forcibly degraded to low quality
    image_count_tolerance: int = 1

    # Policies for text attachments
    allowed_text_extensions: tuple[str, ...] = (
        ".py",
        ".txt",
        ".log",
        ".md",
    )
    max_text_file_size: int = 3 * K

    # Policies for text attachments
    allowed_image_extensions: tuple[str, ...] = (
        ".png",
        ".gif",
        ".jpg",
        ".jpeg",
        ".webp",
    )
    allowed_image_models: tuple[str, ...] = (
        "gpt-4o",
        "gpt-4o-2024-05-13",
    )
    max_image_file_size: int = 10 * MB
    max_image_width: int = 8 * K
    max_image_height: int = 8 * K
    default_image_quality: ImageQuality = ImageQuality.Auto

    # If this option is set, every request is fixed to this value.
    # However, the 'strict' argument in method is True, 'quality' argument in method has more priority than this value.
    # (default: None)
    strict_image_quality: ImageQuality | None = None
    low_quality_threshold: int = 2 * K
    # 2x2 blocks in high quality
    low_quality_token_threshold: int = (
        2 * 2 * openai_settings.OPENAI_IMAGE_BLOCK_COST
        + openai_settings.OPENAI_DEFAULT_IMAGE_COST
    )

    # Since Discord CDN may block bot-scrapping, this option should be able to be optional. (default: False)
    discord_url_allowed: int = False


policy = Policy()
