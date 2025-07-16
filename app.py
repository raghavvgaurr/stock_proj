from flask import Flask, render_template
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

@app.route('/')
def home():
    # Step 1: Download stock data
    df = yf.download('AAPL', start='2020-01-01', end='2024-01-01')
    df['MA10'] = df['Close'].rolling(window=10).mean()

    # Step 2: Feature Engineering
    df['Close_lag1'] = df['Close'].shift(1)
    df['Close_lag2'] = df['Close'].shift(2)
    df = df.dropna()

    # Step 3: Define Features and Target
    X = df[['Close_lag1', 'Close_lag2', 'MA10']]
    y = df['Close']

    # Step 4: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    # Step 5: Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    # Step 6: Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, predicted))
    mae = mean_absolute_error(y_test, predicted)

    # Step 7: Buy/Sell Signals
    signal_list = []
    for i in range(len(predicted)):
        if i == 0:
            signal_list.append("Hold")
        elif predicted[i] > predicted[i - 1]:
            signal_list.append("Buy")
        else:
            signal_list.append("Sell")

    # Step 8: Investment Info
    latest_price = float(y_test.values[-1])
    latest_predicted = float(predicted[-1])
    trend = "↑ Upward" if latest_predicted > float(predicted[-2]) else "↓ Downward"
    recommendation = "Buy ✅" if trend == "↑ Upward" else "Exit ❌"
    investment = 10000
    expected_return = (latest_predicted - latest_price) / latest_price * investment

    # Step 9: Prepare DataFrame
    returns = pd.DataFrame({
        'Date': y_test.index,
        'Actual Price': y_test.values.flatten(),
        'Predicted Price': predicted.flatten(),
        'Return (%)': ((predicted.flatten() - y_test.values.flatten()) / y_test.values.flatten()) * 100,
        'Signal': signal_list
    })

    # Step 10: Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=returns['Date'],
        y=returns['Actual Price'],
        mode='lines+markers',
        name='Actual Price',
        marker=dict(color='blue'),
        hovertemplate='Date: %{x}<br>Actual Price: ₹%{y:.2f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=returns['Date'],
        y=returns['Predicted Price'],
        mode='lines+markers',
        name='Predicted Price',
        marker=dict(color='orange'),
        customdata=np.stack((returns['Return (%)'], returns['Signal']), axis=-1),
        hovertemplate=(
            'Date: %{x}<br>'
            'Predicted Price: ₹%{y:.2f}<br>'
            'Return: %{customdata[0]:.2f}%<br>'
            'Signal: %{customdata[1]}<extra></extra>'
        )
    ))

    fig.update_layout(
        title='📊 Interactive Stock Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        hovermode='x unified',
        template='plotly_white',
        width=1000,
        height=600
    )

    chart_html = pio.to_html(fig, full_html=False)

    return render_template("index.html",
                           chart=chart_html,
                           current_price=f"{latest_price:.2f}",
                           predicted_price=f"{latest_predicted:.2f}",
                           trend=trend,
                           recommendation=recommendation,
                           expected_return=f"{expected_return:.2f}",
                           rmse=f"{rmse:.2f}",
                           mae=f"{mae:.2f}")
if __name__ == "__main__":
    app.run(debug=True)
