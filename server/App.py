from flask import Flask, jsonify
from flask import request
import json

import requests
import datetime

with open('../secrets.json', 'r') as file:
  secrets = json.load(file)

with open('config.json', 'r') as file:
  config = json.load(file)

INVESTTOKEN = secrets["invest"]

app = Flask(__name__)

def filter_by_class_code(instruments, class_code="TQBR"):
  return [i for i in instruments if i["classCode"]== class_code][0]['figi']

def get_figi(query):
  url = "https://invest-public-api.tinkoff.ru/rest/tinkoff.public.invest.api.contract.v1.InstrumentsService/FindInstrument"
  payload = {"query": query}  

  headers = {
    "Authorization": f"Bearer {INVESTTOKEN}",
    "Content-Type": "application/json"
  }

  response = requests.post(url, headers=headers, json=payload).json()

  if hasattr(response, 'instruments'):
    return filter_by_class_code(response['instruments'])
  else:
    return 'none'

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
  for i in candles:
    result.append(float(f'{i['close']['units']}.{i['close']['nano']//10000000}'))

  print(result)

  return result

err = {"error": 0}

@app.route("/")
def hello_world():
    return json.dumps(True)

@app.route("/get-stat")
def stats():
    
    answ = {"header": '', 'data': []}
    
    tradeName = request.args.get('name')

    figi = get_figi(tradeName)

    if figi == 'none':
      answ['header'] = 'Неправильное имя актива'
      return jsonify(answ)

    res = {'tradeData': get_money(figi)}
    
    if len(res['tradeData']) < 24:
      answ['header'] = 'Недостаточно данных от брокера'
      return jsonify(answ)

    url = f"http://{config['server']['ip']}:{config['server']['port']}/getPrediction"
    headers = {'Content-type' : 'application/json'}

    r = requests.post(url, headers=headers, data=json.dumps(res))

    if r.status_code == 500:
      answ['header'] = 'Произошла ошибка в работе модели'
      return jsonify(answ)

    answ['data'] = r.json()
    answ['header'] = 'Успешно'

    return jsonify(answ)

@app.route("/get-stat-small")
def stats():
    
    answ = {"header": '', 'data': []}
    
    tradeName = request.args.get('name')

    figi = get_figi(tradeName)

    if figi == 'none':
      answ['header'] = 'Неправильное имя актива'
      return jsonify(answ)

    res = {'tradeData': get_money(figi)}
    
    if len(res['tradeData']) < 24:
      answ['header'] = 'Недостаточно данных от брокера'
      return jsonify(answ)

    answ['data'] = res.json()
    answ['header'] = 'Успешно'

    return jsonify(answ)