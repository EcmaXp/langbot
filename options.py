
## Constants
K, M, G = 1024, 1024 ** 2, 1024 ** 3
KB, MB, GB = K, M, G

# Officially OpenAI API restricts 20MB per image
API_MAX_IMAGE_SIZE = 20 * MB

# Current price of OpenAI image vision API (Price may be changed).
OPENAI_DEFAULT_IMAGE_COST = 85
OPENAI_IMAGE_BLOCK_COST = 170

## Policies
CHAT_POLICIES = {
    "token_limit": 8 * K,
    "message_fetch_limit": 64,

    "compress_threshold_per_chat": 4 * K,
    "compress_threshold_per_message": 2 * K,

    "max_attachment_count": 3,
    "max_text_attachment_count": 2,
    "max_image_attachment_count": 2,

    # Images in last N messages hold their original quality, others are forcibly degraded to low quality
    "image_message_tolerance": 3,
    # If there are more than N images per message, all images are forcibly degraded to low quality
    "image_count_tolerance": 1
}

ATTACHMENT_POLICIES = {
    # Policies for text attachments
    "allowed_text_extensions": (".py", ".txt", ".log", ".md"),
    "max_text_file_size": 3 * K,

    # Policies for text attachments
    "allowed_image_extensions": (".png", ".gif", ".jpg", ".jpeg", ".webp"),
    "allowed_image_models": ("gpt-4o", "gpt-4o-2024-05-13"),

    "max_image_file_size": 10 * MB,
    "max_image_width": 8 * K,
    "max_image_height": 8 * K,

    "default_image_quality": "auto",
    # If this option is set, every request is fixed to this value.
    # However, the 'strict' argument in method is True, 'quality' argument in method has more priority than this value.
    # (default: None)
    "strict_image_quality": None,
    "low_quality_threshold": 2 * K,
    # 2x2 blocks in high quality
    "low_quality_token_threshold": 2 * 2 * OPENAI_IMAGE_BLOCK_COST + OPENAI_DEFAULT_IMAGE_COST,

    # Since Discord CDN may block bot-scrapping, this option should be able to be optional. (default: False)
    "discord_url_allowed": False
}