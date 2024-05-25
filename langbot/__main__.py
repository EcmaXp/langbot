import asyncio

from langbot.bot import bot

try:
    import uvloop
except ImportError:
    uvloop = None


def main():
    if uvloop:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    bot.run()


if __name__ == "__main__":
    main()
