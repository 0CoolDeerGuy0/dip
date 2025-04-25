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

with open('../secrets.json', 'r') as file:
  secrets = json.load(file)

TOKEN = secrets['bot']

dp = Dispatcher()

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:

    await message.answer(f"Привет, {html.bold(message.from_user.full_name)}!")

@dp.message()
async def stats(message: Message) -> None:

    st = requests.get(f"http://{config['server']['ip']}:{config['server']['port']}/get-stat?name=" + message.text)
    r = st.json()

    if len(r['data']) == 0:
        await message.answer(f"{r['header']}")
        return

    try:
        await message.answer(f"Предсказанные значения: {r['data']}")
    except TypeError:
        await message.answer("Произошла непредвиденная ошибка в модуле бота!")

async def main() -> None:
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    await dp.start_polling(bot)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
