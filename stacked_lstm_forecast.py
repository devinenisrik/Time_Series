# stacked_lstm_forecast.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import yfinance as yf
import math
import warnings

yf.pdr_override()
warnings.filterwarnings("ignore")


# ---------------------- Data Loader ----------------------
def load_stock_data(stock_symbol='ICICIBANK.NS'):
    df = pdr.get_data_yahoo(stock_symbol)
    df1 = df.reset_index()[['Close']]
    return df1


# ---------------------- Preprocessing ----------------------
def preprocess_data(df1):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1_scaled = scaler.fit_transform(np.array(df1).reshape(-1, 1))
    return df1_scaled, scaler


def split_data(data, time_step=100):
    training_size = int(len(data) * 0.8)
    train_data = data[0:training_size]
    test_data = data[training_size:]
    return train_data, test_data


def create_dataset(dataset, time_step=100):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        dataX.append(dataset[i:(i + time_step), 0])
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# ---------------------- Model ----------------------
def build_model(input_shape=(100, 1)):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(1))
    optimizer = Adam(clipvalue=1.0)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


# ---------------------- Forecasting ----------------------
def forecast_next_days(model, last_100, n_steps=100, n_days=30):
    temp_input = list(last_100)
    lst_output = []

    i = 0
    while i < n_days:
        x_input = np.array(temp_input[-n_steps:]).reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.append(yhat[0][0])
        lst_output.append(yhat[0][0])
        i += 1

    return np.array(lst_output).reshape(-1, 1)


# ---------------------- Plotting ----------------------
def plot_results(actual, predicted, title='Prediction vs Actual'):
    plt.figure(figsize=(12, 5))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.title(title)
    plt.legend()
    plt.show()


# ---------------------- Main Pipeline ----------------------
def run_pipeline():
    df1 = load_stock_data()
    df1_scaled, scaler = preprocess_data(df1)
    train_data, test_data = split_data(df1_scaled)

    X_train, y_train = create_dataset(train_data)
    X_test, y_test = create_dataset(test_data)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    y_train = SimpleImputer(strategy='mean').fit_transform(y_train.reshape(-1, 1))
    y_test = SimpleImputer(strategy='mean').fit_transform(y_test.reshape(-1, 1))

    model = build_model()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=1)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    test_actual = scaler.inverse_transform(test_data[101:])[:len(test_predict)]

    plot_results(test_actual, test_predict, title="Test Prediction vs Actual")

    rmse = math.sqrt(mean_squared_error(test_actual, test_predict))
    print("Test RMSE:", rmse)

    x_input = test_data[-100:].reshape(1, -1)
    x_input = x_input[0].tolist()
    next_30 = forecast_next_days(model, x_input, n_steps=100, n_days=30)
    next_30 = scaler.inverse_transform(next_30)

    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(1, 101), scaler.inverse_transform(test_data[-100:]), label='Last 100 Days')
    plt.plot(np.arange(101, 131), next_30, label='Next 30 Days Forecast')
    plt.legend()
    plt.title("Next 30 Days Forecast")
    plt.show()


if __name__ == "__main__":
    run_pipeline()
