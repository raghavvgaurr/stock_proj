from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objs as go
import plotly.io as pio
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    ticker = None
    if request.method == 'POST':
        popular = request.form.get('popular_ticker')
        custom = request.form.get('ticker')
        ticker = custom.strip().upper() if custom else popular

        if not ticker:
            return render_template("index.html", error="Please select or enter a ticker.")

        df = yf.download(ticker, start='2020-01-01', end='2024-01-01')
        if df.empty:
            return render_template("index.html", error=f"No data found for '{ticker}'.")

        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['Close_lag1'] = df['Close'].shift(1)
        df['Close_lag2'] = df['Close'].shift(2)
        df = df.dropna()

        X = df[['Close_lag1', 'Close_lag2', 'MA10']]
        y = df['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

        model = LinearRegression()
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, predicted))
        mae = mean_absolute_error(y_test, predicted)

        signal_list = ["Hold"]
        for i in range(1, len(predicted)):
            signal_list.append("Buy" if predicted[i] > predicted[i - 1] else "Sell")

        latest_price = float(y_test.values[-1])
        latest_predicted = float(predicted[-1])
        trend = "📈 Upward Trend" if latest_predicted > float(predicted[-2]) else "📉 Downward Trend"
        recommendation = "✅ Consider Buying" if "Upward" in trend else "❌ Consider Selling"

        investment = 10000
        expected_return = (latest_predicted - latest_price) / latest_price * investment

        returns = pd.DataFrame({
            'Date': y_test.index,
            'Actual Price': y_test.values.flatten(),
            'Predicted Price': predicted.flatten(),
            'Return (%)': ((predicted.flatten() - y_test.values.flatten()) / y_test.values.flatten()) * 100,
            'Signal': signal_list
        })

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=returns['Date'],
            y=returns['Actual Price'],
            mode='lines+markers',
            name='Actual Price',
            marker=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=returns['Date'],
            y=returns['Predicted Price'],
            mode='lines+markers',
            name='Predicted Price',
            marker=dict(color='orange'),
            customdata=np.stack((returns['Return (%)'], returns['Signal']), axis=-1),
            hovertemplate='Date: %{x}<br>Predicted: ₹%{y:.2f}<br>Return: %{customdata[0]:.2f}%<br>Signal: %{customdata[1]}<extra></extra>'
        ))

        fig.update_layout(
            title=f'{ticker} Stock Price Prediction',
            xaxis_title='Date',
            yaxis_title='Price (₹)',
            template='plotly_white',
            width=1000,
            height=600
        )

        chart_html = pio.to_html(fig, full_html=False)

        return render_template("index.html",
                               chart=chart_html,
                               ticker=ticker,
                               current_price=f"{latest_price:.2f}",
                               predicted_price=f"{latest_predicted:.2f}",
                               trend=trend,
                               recommendation=recommendation,
                               expected_return=f"{expected_return:.2f}",
                               rmse=f"{rmse:.2f}",
                               mae=f"{mae:.2f}")

    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)



