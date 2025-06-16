# Time_Series
## 1. Time Series using FB Prophet

### 📈 Stock Price Forecasting using Facebook Prophet

This project uses **Facebook Prophet** to forecast stock prices (e.g., ICICI Bank). It pulls historical stock data using Yahoo Finance, applies rolling statistics, exponential moving averages, and trains a forecasting model to predict future prices with evaluation.

---

### 🔍 Features

- Fetch historical stock price data using `yfinance`
- Calculate and visualize:
  - Rolling averages
  - Exponential moving averages (EMA)
- Time series forecasting using `Prophet`
- Visualize future trends and yearly/weekly components
- Evaluate predictions with RMSE
- Perform cross-validation for time series forecasts

---

### 🛠 Installation

Make sure you have Python 3.7+ installed.

```bash
pip install yfinance pandas matplotlib prophet statsmodels
------------------------------------
----------------------------------------------------
--------------------------


# 2. Using Stacked LSTM


### 📈 Stock Price Prediction Using Stacked LSTM

This project demonstrates how to build a **Stacked LSTM** model using TensorFlow and Keras to predict and forecast the stock prices of **ICICI Bank (ICICIBANK.NS)** using historical data from Yahoo Finance.

---

## 🚀 Features

* Data fetched using `yfinance` & `pandas_datareader`
* Data normalization with `MinMaxScaler`
* Stacked LSTM model with 3 LSTM layers
* Visualization of actual vs. predicted values
* 30-day price forecasting
* RMSE evaluation

---

## 🧠 Technologies Used

* Python
* TensorFlow / Keras
* Scikit-learn
* Numpy / Pandas
* Matplotlib
* Yahoo Finance API (via `yfinance`)

---

## 🛠️ How It Works

### 1. Load & Preprocess Stock Data

* Fetch closing prices of the stock.
* Normalize using MinMaxScaler.

### 2. Create Sequences

* Define 100-day sequences for LSTM input.

### 3. Build and Train Stacked LSTM Model

* Architecture: `LSTM(100) → LSTM(100) → LSTM(100) → Dense(1)`

### 4. Evaluate

* Predict on test set and inverse transform.
* RMSE is used to evaluate performance.

### 5. Forecast Future

* Forecast next 30 days based on the last 100 values.

---

## 📊 Visualizations

* Actual vs Predicted Price Plot
* Next 30-Day Price Forecast

---

## 📁 File Structure

```bash
.
├── stacked_lstm_forecast.py    # Main modular pipeline
└── README.md                   # This file
```

---

## 🧪 Run the Code

```bash
python stacked_lstm_forecast.py
```

---

## 📌 Requirements

```bash
pip install yfinance pandas numpy matplotlib scikit-learn tensorflow pandas_datareader
```

---

## 🔮 Sample Output

* RMSE: Root Mean Squared Error on test data
* Plots: Real-time comparison and forward forecast

---

## 📬 Contact

For queries or suggestions, feel free to raise an issue or connect.

