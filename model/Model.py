import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from flask import Flask, jsonify
from flask import request

app = Flask(__name__)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=1, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTMModel()
model.load_state_dict(torch.load('lstm_model_weights49.pth'))
model.eval()

scaler = joblib.load('scaler49.save')

def predict(model, scaler, last_sequence, future_steps=12):
    """
    Прогнозирует future_steps значений вперёд
    
    :param model: обученная модель
    :param scaler: масштабировщик
    :param last_sequence: последняя известная последовательность (24 часа)
    :param future_steps: сколько шагов прогнозировать
    :return: список спрогнозированных значений
    """
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(future_steps):
        input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0)
        
        with torch.no_grad():
            predicted = model(input_tensor).numpy()[0]
        
        predictions.append(predicted[0])
        
        current_sequence = np.vstack([current_sequence[1:], predicted])
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    return predictions.flatten()

@app.route("/")
def hello_world():
    return 'hello'

@app.route("/getPrediction", methods=["GET", "POST"])
def prediction():

    data = request.get_json()

    y = np.array(data['tradeData'])
    window = 3
    smoothed = np.convolve(y, np.ones(window)/window, mode='valid')
    diff_smoothed = np.diff(smoothed)
    trend_direction = "+" if np.mean(diff_smoothed) > 0 else "-" if np.mean(diff_smoothed) < 0 else "0"

    input_data = np.array(data['tradeData'])

    input_data_2d = input_data.reshape(-1, 1)
    input_scaled = scaler.transform(input_data_2d) 

    last_sequence = input_scaled[-24:].reshape(24, 1)

    future_steps = 12
    predictions = predict(model, scaler, last_sequence, future_steps)

    r = {'trend': trend_direction, 'predictionData': predictions.tolist()}

    return jsonify(r)