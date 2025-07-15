# app.py
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go

st.set_page_config(page_title="📈 Stock Predictor", layout="wide")
st.title("📊 Stock Price Prediction App (Linear Regression)")

# Sidebar inputs
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g. AAPL, TSLA, GOOG)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

if start_date >= end_date:
    st.error("End date must be after start date.")
    st.stop()

# Download data
df = yf.download(symbol, start=start_date, end=end_date)
if df.empty:
    st.warning("No data found. Try another symbol.")
    st.stop()

df['MA10'] = df['Close'].rolling(window=10).mean()
df['Close_lag1'] = df['Close'].shift(1)
df['Close_lag2'] = df['Close'].shift(2)
df = df.dropna()

# Features and Target
X = df[['Close_lag1', 'Close_lag2', 'MA10']]
y = df['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)
predicted = model.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, predicted))
mae = mean_absolute_error(y_test, predicted)

# Investment Suggestion
latest_price = float(y_test.values[-1])
latest_predicted = float(predicted[-1])
trend = "↑ Upward" if latest_predicted > float(predicted[-2]) else "↓ Downward"
recommendation = "Buy ✅" if trend == "↑ Upward" else "Exit ❌"
investment = 10000
expected_return = (latest_predicted - latest_price) / latest_price * investment

# Show metrics
st.subheader("📈 Model Evaluation")
st.markdown(f"- RMSE: **{rmse:.2f}**")
st.markdown(f"- MAE: **{mae:.2f}**")

# Return DataFrame
returns = pd.DataFrame({
    'Date': y_test.index,
    'Actual Price': y_test.values.flatten(),
    'Predicted Price': predicted.flatten(),
    'Return (%)': ((predicted - y_test.values) / y_test.values) * 100
}).reset_index(drop=True)

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=returns['Date'], y=returns['Actual Price'],
                         mode='lines+markers', name='Actual Price', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=returns['Date'], y=returns['Predicted Price'],
                         mode='lines+markers', name='Predicted Price', line=dict(color='orange')))

fig.update_layout(title="📊 Actual vs Predicted Stock Price",
                  xaxis_title="Date", yaxis_title="Price (₹)", template='plotly_white')

st.plotly_chart(fig, use_container_width=True)

# Investment Advice
st.subheader("💡 Investment Suggestion")
st.markdown(f"- Current Price: ₹**{latest_price:.2f}**")
st.markdown(f"- Predicted Price: ₹**{latest_predicted:.2f}**")
st.markdown(f"- Trend: **{trend}**")
st.markdown(f"- Recommendation: **{recommendation}**")
if recommendation == "Buy ✅":
    st.markdown(f"- Suggested Investment: ₹**{investment}**")
    st.markdown(f"- Expected Profit: ₹**{expected_return:.2f}**")

# Return Table
st.subheader("📊 Return Table (Preview)")
st.dataframe(returns.head(10), use_container_width=True)
