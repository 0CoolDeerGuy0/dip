from flask import Flask
from flask import request
import json

app = Flask(__name__)

def get_money ():
  

    return 0

x = {
  "open": 30,
  "close": 31,
  "percent": 0,
  "trend": True
}

y = {
  "open": 43,
  "close": 98,
  "percent": 0,
  "trend": True
}

z = {
  "open": 65,
  "close": 53,
  "percent": 0,
  "trend": False
}

err = {"error": 0}

@app.route("/")
def hello_world():
    return json.dumps(True)

@app.route("/get-stat")
def stats():
    
    tradeName = request.args.get('name')

    res = json.dumps(err)

    if (tradeName == 'x'):
        res = json.dumps(x)

    if (tradeName == 'y'):
        res = json.dumps(y)

    if (tradeName == 'z'):
        res = json.dumps(z)

    return res