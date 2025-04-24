from flask import Flask
from flask import request
import json

import requests
import datetime

with open('../secrets.json', 'r') as file:
  secrets = json.load(file)

INVESTTOKEN = secrets["invest"]

app = Flask(__name__)

def filter_by_class_code(instruments, class_code="TQBR"):
  return [i for i in instruments if i["classCode"]== class_code][0]['figi']

def get_figi(query):
  url = "https://invest-public-api.tinkoff.ru/rest/tinkoff.public.invest.api.contract.v1.InstrumentsService/FindInstrument"
  payload = {"query": query}  # Или "AAPL" для Apple

  headers = {
    "Authorization": f"Bearer {INVESTTOKEN}",
    "Content-Type": "application/json"
  }

  response = requests.post(url, headers=headers, json=payload).json()
  return filter_by_class_code(response['instruments'])

def get_money (figi):
  url = "https://invest-public-api.tinkoff.ru/rest/tinkoff.public.invest.api.contract.v1.MarketDataService/GetCandles"
  headers = {
    "Authorization": f"Bearer {INVESTTOKEN}",
    "Content-Type": "application/json"
  }

  now = datetime.datetime.now(datetime.UTC)
  start_time = now - datetime.timedelta(hours=32)

  payload = {
    "figi": figi,
    "from": start_time.isoformat(),
    "to": now.isoformat(),
    "interval": "CANDLE_INTERVAL_HOUR"
  }

  response = requests.post(url, headers=headers, json=payload)

  if response.status_code == 200:
    data = response.json()
    candles = data["candles"]
  else:
    print("Ошибка:", response.status_code, response.text)
    candles = []

  result = []
  print(len(candles))
  for i in candles:
    result.append(float(f'{i['close']['units']}.{i['close']['nano']//10000000}'))

  return result

print(get_money(get_figi('SBER')))

err = {"error": 0}

@app.route("/")
def hello_world():
    return json.dumps(True)

@app.route("/get-stat")
def stats():
    
    tradeName = request.args.get('name')

    res = json.dumps(err)

    return res