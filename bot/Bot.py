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
from aiogram.types import KeyboardButton, ReplyKeyboardMarkup
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

import requests

with open('config.json', 'r') as file:
    config = json.load(file)

with open('../secrets.json', 'r') as file:
  secrets = json.load(file)

TOKEN = secrets['bot']

dp = Dispatcher()

class Form(StatesGroup):
    prediction = State()
    graph = State()

def main_kb():
    kb_list = [
        [KeyboardButton(text="❓ Прогноз"), KeyboardButton(text="📈 Котировки")],
    ]
    keyboard = ReplyKeyboardMarkup(keyboard=kb_list, resize_keyboard=True, one_time_keyboard=False, input_field_placeholder="Воспользуйтесь меню:")
    return keyboard

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:

    await message.answer(f"Привет, {html.bold(message.from_user.full_name)}!", reply_markup=main_kb())

@dp.message(lambda message: message.text == "❓ Прогноз")
async def stats(message: Message, state: FSMContext) -> None:

    await message.answer(f"Введите короткое имя актива")
    await state.set_state(Form.prediction)

@dp.message(lambda message: message.text == "📈 Котировки")
async def stats(message: Message, state: FSMContext) -> None:

    await message.answer(f"Введите короткое имя актива")
    await state.set_state(Form.graph)
    
@dp.message(Form.prediction)
async def stats(message: Message, state: FSMContext) -> None:

    st = requests.get(f"http://{config['server']['ip']}:{config['server']['port']}/get-stat?name=" + message.text)
    r = st.json()

    if len(r['data']) == 0:
        await message.answer(f"{r['header']}")
        return

    try:
        await message.answer(f"Предсказанные значения: {r['data']['predictionData']} \nТренд: {'Положительный' if r['trend'] == '+' else 'Отрицательный' if r['trend'] == '-' else 'Стабильный'}")
    except TypeError:
        await message.answer("Произошла непредвиденная ошибка в модуле бота!")

@dp.message(Form.graph)
async def stats(message: Message, state: FSMContext) -> None:

    st = requests.get(f"http://{config['server']['ip']}:{config['server']['port']}/get-stat-small?name=" + message.text)
    r = st.json()

    if len(r['data']) == 0:
        await message.answer(f"{r['header']}")
        return

    try:
        await message.answer(f"Текущие котировки: {r['data']}")
    except TypeError:
        await message.answer("Произошла непредвиденная ошибка в модуле бота!")

async def main() -> None:
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    await dp.start_polling(bot)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
