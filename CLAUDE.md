# CLAUDE.md

This file provides guidance to Claude Code when working with the Langbot Discord bot.

## Overview

Langbot is a Discord bot that provides AI-powered chat functionality using multiple language models (OpenAI, Anthropic Claude, Google Gemini).

## Quick Start

1. Install dependencies: `uv sync --frozen`
2. Set required environment variables
3. Run locally: `uv run langbot`
4. For development: `uv run reloader langbot` (auto-reload)

## Architecture

### Core Components

- **Entry Point** (`__main__.py`): Starts the bot with uvloop for performance
- **Bot Setup** (`bot.py`): Hikari GatewayBot configuration and event listeners
- **Chat Handler** (`chatgpt.py`): Message processing and LangChain integration
- **Attachments** (`attachment.py`): Image and text file handling with size limits
- **Configuration** (`options.py`): Pydantic settings and policy management

### Message Processing Flow

1. **Receive**: Check if message is from bot (prevent loops)
2. **Validate**: Apply rate limits and permission checks
3. **Context**: Fetch reply chains and attachments
4. **Process**: Send to appropriate AI model
5. **Respond**: Stream or send complete response

## Key Component Reference

### Chat Handler (`chatgpt.py`)

- `on_message_create()`: Message entry point
- `fetch_referenced_messages()`: Reply context retrieval
- `process_attachments()`: File/image handling

### Attachment Processing (`attachment.py`)

- Images: Converted to base64 for vision models
- Text files: Extracted with encoding detection
- Enforces size/type limits from policy configuration

### LangChain Integration

- `get_chat_model()`: Model instantiation
- Chain building with prompt templates
- Response streaming with `astream()`

## Configuration

### Environment Variables

All configuration uses `LANGBOT_` prefix with `__` for nested settings.

**Required:**

- `LANGBOT_DISCORD_TOKEN`: Discord bot token
- `LANGBOT_CHAT_MODEL`: Format `provider:model-name`
- API keys based on chosen provider (ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY)

**Policy Defaults:**

- `LANGBOT_POLICY__TOKEN_LIMIT`: 8192
- `LANGBOT_POLICY__MESSAGE_FETCH_LIMIT`: 64
- `LANGBOT_POLICY__MAX_ATTACHMENT_COUNT`: 3
- `LANGBOT_POLICY__MAX_TEXT_FILE_SIZE`: 3072 (3KB)
- `LANGBOT_POLICY__MAX_IMAGE_FILE_SIZE`: 10485760 (10MB)

## Development Guidelines

### Critical Discord Patterns

1. **Always check bot messages first** - Prevents infinite message loops
2. **All Discord operations must be async** - Synchronous calls will fail
3. **Wrap event handlers in try/except** - Uncaught exceptions crash the bot
4. **Use hikari event types** - Provides proper type hints and attributes

### Code Standards

- Python 3.12+ with comprehensive type hints
- Async/await for all I/O operations
- Pydantic for data validation
- Import order: future annotations → standard library → third-party → local
- Error handling: specific exceptions before general Exception
- Format with `ruff format`, lint with `ruff check`

### Testing Requirements

Before committing changes:

- Run linting and formatting checks
- Verify bot responds to messages
- Test attachment handling (images and text)
- Validate different model providers work
- Monitor memory usage for leaks

## Commit Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New features
- `fix:` Bug fixes
- `chore:` Maintenance tasks
- `refactor:` Code restructuring
- `docs:` Documentation changes
- Breaking changes: `feat!:` or `fix!:`

**Important:** If the commit involves complex changes, write a detailed message explaining what was changed and why. For simple, self-explanatory commits, the description can be left empty.

**Examples:**

- `feat: add Claude 3.5 support`
- `fix: handle empty messages`
- `chore: update dependencies`
- `refactor: simplify attachment handling`
- `docs: update setup instructions`

## Common Issues & Solutions

- **Discord operations fail silently**: Missing `await` → Always use async/await
- **Bot freezes**: Blocking I/O → Use aiohttp, not requests
- **Bot crashes**: Unhandled exceptions → Wrap handlers in try/except
- **Memory leaks**: Large attachments not cleaned → Clean up after processing
- **Rate limit errors**: Too many requests → Use built-in rate limiters
