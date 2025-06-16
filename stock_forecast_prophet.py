import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from statsmodels.tools.eval_measures import rmse
import warnings

warnings.filterwarnings("ignore")


def load_stock_data(ticker, start=None, end=None):
    yf.pdr_override()
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    return df[['Date', 'Close']]


def plot_rolling_averages(df):
    df['Close:10'] = df['Close'].rolling(10).mean()
    df['Close:30'] = df['Close'].rolling(30).mean()
    df['Close:50'] = df['Close'].rolling(50).mean()
    df[['Close', 'Close:10', 'Close:30', 'Close:50']].plot(figsize=(12, 5), title='Rolling Averages')
    plt.show()


def plot_ema(df):
    df['EMA_0.1'] = df['Close'].ewm(alpha=0.1, adjust=False).mean()
    df['EMA_0.3'] = df['Close'].ewm(alpha=0.3, adjust=False).mean()
    df['EMA_Span5'] = df['Close'].ewm(span=5).mean()
    df[['Close', 'EMA_0.1', 'EMA_0.3', 'EMA_Span5']].plot(figsize=(12, 5), title='Exponential Moving Averages')
    plt.show()


def prepare_prophet_data(df, cutoff_date):
    df_m = df[df['Date'] <= cutoff_date].copy()
    df_m.columns = ['ds', 'y']
    return df_m


def train_prophet_model(df_m):
    model = Prophet()
    model.fit(df_m)
    return model


def forecast_future(model, periods=365):
    future_dates = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future_dates)
    return forecast


def plot_forecast(model, forecast):
    model.plot(forecast)
    plt.show()
    model.plot_components(forecast)
    plt.show()
    plot_plotly(model, forecast)


def evaluate_model(model, df, forecast, cutoff_date):
    test = df[df['Date'] > cutoff_date].copy()
    predicted = forecast[['ds', 'yhat']]
    predicted.columns = ['Date', 'yhat']
    merged = pd.merge(test, predicted, on='Date')
    merged.set_index('Date', inplace=True)
    merged[['Close', 'yhat']].plot(figsize=(12, 5), title='Actual vs Forecasted')
    plt.show()
    return merged


def prophet_cross_validation(model):
    df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='365 days')
    df_p = performance_metrics(df_cv)
    plot_cross_validation_metric(df_cv, metric='rmse')
    return df_cv, df_p


def calculate_rmse(actual, predicted):
    return rmse(actual, predicted)


def main():
    ticker = "ICICIBANK.NS"
    cutoff_date = '2023-06-30'

    df = load_stock_data(ticker)
    plot_rolling_averages(df)
    plot_ema(df)

    df_m = prepare_prophet_data(df, cutoff_date)
    model = train_prophet_model(df_m)
    forecast = forecast_future(model)

    plot_forecast(model, forecast)

    results = evaluate_model(model, df, forecast, cutoff_date)
    print(f"\nModel RMSE: {calculate_rmse(results['Close'], results['yhat'])}")
    print(f"Average Close Price: {results['Close'].mean()}")

    prophet_cross_validation(model)


if __name__ == "__main__":
    main()
