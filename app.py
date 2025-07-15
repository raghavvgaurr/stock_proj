import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go

# ------------------- UI -------------------
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📊 Stock Price Predictor")
st.write("This app predicts stock prices using Linear Regression.")

# ------------------- Data -------------------
df = yf.download('AAPL', start='2020-01-01', end='2024-01-01')
df['MA10'] = df['Close'].rolling(window=10).mean()
df['Close_lag1'] = df['Close'].shift(1)
df['Close_lag2'] = df['Close'].shift(2)
df = df.dropna()

# ------------------- Features -------------------
X = df[['Close_lag1', 'Close_lag2', 'MA10']]
y = df['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)
predicted = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predicted))
mae = mean_absolute_error(y_test, predicted)

st.markdown(f"### 📉 Model Performance")
st.write(f"- RMSE: `{rmse:.2f}`")
st.write(f"- MAE: `{mae:.2f}`")

# ------------------- Buy/Sell Logic -------------------
buy_signals = []
sell_signals = []
signal_list = []

for i in range(len(predicted)):
    if i == 0:
        signal_list.append("Hold")
        continue
    if predicted[i] > predicted[i - 1]:
        buy_signals.append(i)
        signal_list.append("Buy")
    else:
        sell_signals.append(i)
        signal_list.append("Sell")

latest_price = float(y_test.values[-1])
latest_predicted = float(predicted[-1])
trend = "↑ Upward" if latest_predicted > float(predicted[-2]) else "↓ Downward"
recommendation = "Buy ✅" if trend == "↑ Upward" else "Exit ❌"
investment = 10000
expected_return = (latest_predicted - latest_price) / latest_price * investment

# ------------------- Summary -------------------
st.markdown("### 🧾 Investment Summary")
st.write(f"📊 Current Price: ₹{latest_price:.2f}")
st.write(f"📈 Predicted Price: ₹{latest_predicted:.2f}")
st.write(f"📉 Trend: {trend}")
st.write(f"🧠 Recommendation: {recommendation}")
if recommendation == "Buy ✅":
    st.write(f"💰 Suggested Investment: ₹{investment}")
    st.write(f"📈 Expected Profit: ₹{expected_return:.2f}")

# ------------------- Table -------------------
returns = pd.DataFrame({
    'Date': y_test.index,
    'Actual Price': y_test.values.flatten(),
    'Predicted Price': predicted.flatten(),
    'Return (%)': ((predicted.flatten() - y_test.values.flatten()) / y_test.values.flatten()) * 100,
    'Signal': signal_list
})

st.markdown("### 📋 Prediction Table (first 10 rows)")
st.dataframe(returns.head(10))

# ------------------- Plot -------------------
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
    hovertemplate='Date: %{x}<br>Predicted Price: ₹%{y:.2f}<br>Return: %{customdata[0]:.2f}%<br>Signal: %{customdata[1]}<extra></extra>',
    customdata=np.stack((returns['Return (%)'], returns['Signal']), axis=-1)
))

fig.update_layout(
    title='📈 Interactive Stock Price Prediction Chart',
    xaxis_title='Date',
    yaxis_title='Price (₹)',
    hovermode='x unified',
    template='plotly_white'
)

st.plotly_chart(fig, use_container_width=True)
