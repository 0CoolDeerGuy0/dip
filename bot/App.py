import asyncio
import json
import logging
import sys
from os import getenv

from aiogram import Bot, Dispatcher, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message

import requests

with open('config.json', 'r') as file:
    config = json.load(file)

# Bot token can be obtained via https://t.me/BotFather
TOKEN = "7085756019:AAGNIyTreXapTCJMpTM_1VCjaYGcQsZ_QGU"

# All handlers should be attached to the Router (or Dispatcher)

dp = Dispatcher()


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:

    await message.answer(f"Привет, {html.bold(message.from_user.full_name)}!")


@dp.message()
async def stats(message: Message) -> None:

    st = requests.get(f"http://${config['server']['ip']}:${config['server']['port']}/get-stat?name=" + message.text)
    r = st.json()

    try:
        
        
        await message.answer(f"Открытие: {r["open"]}, Закрытие: {r["close"]}, Процент разницы: {r["percent"]}, Тренд: {'Положительный' if r["trend"] else 'Отрицательный'}")
    except TypeError:
        # But not all the types is supported to be copied so need to handle it
        await message.answer("Nice try!")


async def main() -> None:
    # Initialize Bot instance with default bot properties which will be passed to all API calls
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    # And the run events dispatching
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())