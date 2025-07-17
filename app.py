from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import plotly.io as pio

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    stock = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2024-01-01'

    if request.method == 'POST':
        stock = request.form['stock']
        start_date = request.form['start_date']
        end_date = request.form['end_date']

    df = yf.download(stock, start=start_date, end=end_date)

    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * 0.8))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:int(training_data_len), :]
    x_train, y_train = [], []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    test_data = scaled_data[training_data_len - 60:, :]
    x_test, y_test = [], dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)

    # Future prediction
    last_60_days = data[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)
    X_future = []
    X_future.append(last_60_days_scaled)
    X_future = np.array(X_future)
    X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], 1))
    predicted_price = model.predict(X_future)
    predicted_price = scaler.inverse_transform(predicted_price)[0][0]

    current_price = dataset[-1][0]
    expected_return = predicted_price - current_price

    trend = "Up" if predicted_price > current_price else "Down"
    recommendation = "Buy" if trend == "Up" else "Sell"

    # Plot using Plotly
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], name='Train'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], name='Actual'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], name='Predicted'))
    fig.update_layout(title='Stock Price Prediction', xaxis_title='Date', yaxis_title='Close Price')

    graph_html = pio.to_html(fig, full_html=False)

    return render_template(
        'index.html',
        current_price=round(current_price, 2),
        predicted_price=round(predicted_price, 2),
        trend=trend,
        recommendation=recommendation,
        rmse=round(rmse, 2),
        mae=round(mae, 2),
        expected_return=round(expected_return, 2),
        chart=graph_html
    )

if __name__ == '__main__':
    app.run(debug=True)

