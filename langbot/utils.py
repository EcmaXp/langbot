import re


def humanize_tuple(tup: tuple):
    if len(tup) == 1:
        return str(tup[0])
    elif len(tup) == 2:
        return f"{tup[0]} and {tup[1]}"
    else:
        return ", ".join(tup[:-1]) + f", and {tup[-1]}"


DISCORD_MESSAGE_LIMIT = 2000

_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_LANG_RE = re.compile(r"[\w+#.-]{1,32}")


def _fence_states(lines: list[str]) -> list[tuple[str, str] | None]:
    """Return the open fence, as (opening_line, closing_marker), at each line boundary."""
    states: list[tuple[str, str] | None] = [None]
    current: tuple[str, str] | None = None
    for line in lines:
        match = _FENCE_RE.match(line)
        if current is None:
            if match and not (match[1][0] == "`" and "`" in match[2]):
                lang = _LANG_RE.match(match[2].strip())
                current = (match[1] + (lang[0] if lang else ""), match[1])
        elif match and match[2].strip() == "":
            marker = match[1]
            if marker[0] == current[1][0] and len(marker) >= len(current[1]):
                current = None
        states.append(current)
    return states


def _split_long_line(line: str, width: int) -> list[str]:
    parts = []
    while len(line) > width:
        cut = line.rfind(" ", width // 2, width)
        if cut == -1:
            cut = width
        parts.append(line[:cut])
        line = line[cut:].lstrip(" ")
    parts.append(line)
    return parts


def split_message(text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    """Split text into Discord-sized chunks without breaking Markdown code blocks."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []

    width = max(limit - 200, limit // 2)
    lines = [
        part for line in text.split("\n") for part in _split_long_line(line, width)
    ]
    fences = _fence_states(lines)

    chunks: list[str] = []
    start = 0
    while start < len(lines):
        reopen = fences[start]
        size = len(reopen[0]) + 1 if reopen else 0
        end = start
        best = None
        while end < len(lines):
            close = fences[end + 1]
            added = len(lines[end]) + 1
            if (
                end > start
                and size + added + (len(close[1]) + 1 if close else 0) > limit
            ):
                break
            size += added
            end += 1
            at_paragraph = (
                end == len(lines)
                or lines[end].strip() == ""
                or lines[end].startswith("#")
            )
            if fences[end] is None and at_paragraph and size >= limit // 2:
                best = end
        if end < len(lines) and best is not None:
            end = best

        body = lines[start:end]
        if reopen:
            body = [reopen[0], *body]
        if fences[end]:
            body.append(fences[end][1])
        chunk = "\n".join(body).strip()
        if chunk:
            chunks.append(chunk)
        start = end

    return chunks
