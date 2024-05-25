import os

import hikari

from langbot.chatgpt import ChatGPT

bot = hikari.GatewayBot(
    os.environ.get("DISCORD_TOKEN"),
    intents=hikari.Intents.ALL_MESSAGES | hikari.Intents.MESSAGE_CONTENT,
)


chatgpt = ChatGPT(
    bot,
    {"chat_model": os.environ["LANGBOT_CHAT_MODEL"]},
)


@bot.listen()
async def on_ready(event: hikari.ShardReadyEvent):
    await chatgpt.on_ready(event)


@bot.listen()
async def on_message(event: hikari.MessageCreateEvent):
    await chatgpt.on_message(event)


@bot.listen()
async def on_message_edit(event: hikari.MessageUpdateEvent):
    await chatgpt.on_message_edit(event)


@bot.listen()
async def on_message_delete(event: hikari.MessageDeleteEvent):
    await chatgpt.on_message_delete(event)
