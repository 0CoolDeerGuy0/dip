import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=1, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Инициализация весов
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

def predict_future(model, scaler, last_sequence, future_steps=12):
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
        # Преобразуем в тензор и добавляем размерность batch
        input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0)
        
        # Делаем прогноз
        with torch.no_grad():
            predicted = model(input_tensor).numpy()[0]
        
        predictions.append(predicted[0])
        
        # Обновляем последовательность: удаляем первый элемент и добавляем прогноз
        current_sequence = np.vstack([current_sequence[1:], predicted])
    
    # Обратное преобразование масштаба
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    return predictions.flatten()


input_data = np.array([306.25, 310.11, 308.14, 309.85, 309.9, 309.74, 309.51, 310.59, 
                      311.79, 312.3, 311.62, 310.56, 310.6, 312.0, 310.77, 311.27, 
                      311.29, 311.38, 312.6, 314.4, 312.77, 312.4, 311.7, 316.9, 
                      310.45, 310.32, 310.63])

# 1. Преобразуем данные в 2D формат и нормализуем
input_data_2d = input_data.reshape(-1, 1)  # преобразуем в (n_samples, n_features)
input_scaled = scaler.transform(input_data_2d)  # используем загруженный scaler

# 2. Берем последние 24 значения (как ожидает модель)
last_sequence = input_scaled[-24:].reshape(24, 1)  # последовательность из 24 часов

# 3. Делаем прогноз на 12 часов вперед
future_steps = 12
predictions = predict_future(model, scaler, last_sequence, future_steps)

print("Прогнозируемые значения:")
for i, price in enumerate(predictions, 1):
    print(f"Час {i}: {price:.2f}")

# 4. Визуализация
plt.figure(figsize=(12, 6))
plt.plot(input_data, 'b-', label='Исторические данные')
plt.plot(range(len(input_data), len(input_data)+future_steps), predictions, 'r-', label='Прогноз')
plt.title('Прогноз цены на следующие 12 часов')
plt.xlabel('Время (часы)')
plt.ylabel('Цена')
plt.legend()
plt.grid()
plt.show()