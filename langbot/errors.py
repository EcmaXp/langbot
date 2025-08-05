class LangBotError(Exception):
    reason: str = "An error occurred"
    solution: str | None = None

    def __format__(self, format_spec: str) -> str:
        if format_spec == "discord":
            message = f":x: **{self.reason}**"
            if self.solution:
                message += f" - {self.solution}"
            return message
        return str(self)


class LimitExceededError(LangBotError):
    def __init__(self, value: int, limit: int):
        self.value = value
        self.limit = limit
        self.reason = f"{self.reason} ({value} > {limit})"
        super().__init__(self.reason)


class TokenLimitExceededError(LimitExceededError):
    reason = "Token limit exceeded"
    solution = "Start a new conversation"


class TooManyAttachmentsError(LimitExceededError):
    reason = "Too many attachments"
    solution = "Try fewer or smaller files"


class InvalidResponseError(LangBotError):
    reason = "Invalid response"
    solution = "Try a shorter prompt or start a new conversation"

    def __init__(self, finish_reason: str | None = None):
        self.finish_reason = finish_reason
        if finish_reason:
            self.reason = f"{self.reason} (reason: {finish_reason})"
        super().__init__(self.reason)


class ResponseTruncatedError(InvalidResponseError):
    reason = "Response was truncated"
    solution = "Try a shorter prompt"


class EmptyResponseError(InvalidResponseError):
    reason = "No response received"
    solution = "Try again or rephrase your request"


class UnknownRoleError(LangBotError):
    reason = "Unknown message role"

    def __init__(self, role: str):
        self.role = role
        self.reason = f"{self.reason}: {role}"
        super().__init__(self.reason)


class NotTextChannelError(LangBotError):
    reason = "Channel is not a text channel"

    def __init__(self, channel_id: int):
        self.channel_id = channel_id
        super().__init__(self.reason)
