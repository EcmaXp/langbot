FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /bin/uv
WORKDIR /app

COPY ./pyproject.toml ./uv.lock /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --compile-bytecode

COPY ./langbot /app/langbot
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --compile-bytecode

ENV PYTHONOPTIMIZE=2
CMD ["uv", "run", "langbot"]
