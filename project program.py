import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
ticker = 'AAPL'
df = yf.download(ticker, start='2015-01-01', end='2024-12-31')
data = df[['Close']].values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
train_len = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_len]
test_data = scaled_data[train_len - 60:]
def create_sequences(data, seq_length):
 x, y = [], []
 for i in range(seq_length, len(data)):
 x.append(data[i-seq_length:i])
 y.append(data[i])
 return np.array(x), np.array(y)
seq_length = 60
x_train, y_train = create_sequences(train_data, seq_length)
x_test, y_test = create_sequences(test_data, seq_length)
model = Sequential([
 LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
 LSTM(50),
 Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=10, batch_size=32)
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
real_prices = scaler.inverse_transform(y_test)
plt.figure(figsize=(12, 6))
plt.plot(real_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
