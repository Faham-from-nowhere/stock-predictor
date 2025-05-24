import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import math
import seaborn as sns
import tempfile
from datetime import datetime
from sklearn.model_selection import ParameterGrid

# === Streamlit UI ===
st.title("📈 Stock Price Prediction, Signals & Backtesting")
st.write("Visualize LSTM predictions, trend signals, and evaluate strategy with backtesting.")

# File upload
uploaded_file = st.file_uploader("Upload your CSV data", type=["csv"])
model_file = st.file_uploader("Upload your trained LSTM Model (.h5)", type=["h5"])

# Sidebar configurations
st.sidebar.subheader("Technical Indicators")
ma_window = st.sidebar.slider("Moving Average Window", 5, 100, 20)
rsi_window = st.sidebar.slider("RSI Window", 5, 100, 14)

st.sidebar.subheader("Monte Carlo Simulation")
enable_monte_carlo = st.sidebar.checkbox("Enable Monte Carlo Simulation")
n_simulations = st.sidebar.slider("Number of Simulations", 100, 5000, 1000)

st.sidebar.subheader("Sensitivity Analysis")
enable_sensitivity = st.sidebar.checkbox("Enable Sensitivity Analysis")
ma_range = st.sidebar.slider("MA Range (Start, End)", 5, 100, (10, 50))
rsi_range = st.sidebar.slider("RSI Range (Start, End)", 5, 50, (10, 30))
step = st.sidebar.slider("Step Size", 1, 10, 5)

# === Streamlit Sidebar for Rebalancing Settings ===
st.sidebar.subheader("Portfolio Rebalancing & Optimization")
rebalance_frequency = st.sidebar.selectbox("Rebalancing Frequency", ["Daily", "Weekly", "Monthly"])
optimize_allocation = st.sidebar.checkbox("Optimize Allocation")

# === Data Preparation ===
if uploaded_file and model_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values(by='Date').dropna(subset=['Date'])
    
    # Technical Indicators
    df['MA'] = df['Close'].rolling(window=ma_window).mean()
    delta = df['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Scale and prepare data for LSTM
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume', 'MA', 'RSI']])
    X_pred = np.array([scaled_data[i-60:i] for i in range(60, len(scaled_data))])
    
    # Load Model and Predict
    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as temp_model_file:
        temp_model_file.write(model_file.read())
        model = load_model(temp_model_file.name)

    predictions = model.predict(X_pred)
    predicted_full = np.zeros((len(predictions), 7))
    predicted_full[:, 3] = predictions[:, 0]
    predictions_rescaled = scaler.inverse_transform(predicted_full)[:, 3]
    
    df['Predictions'] = np.nan
    df.loc[60:, 'Predictions'] = predictions_rescaled

    # Plot Predictions
    plt.figure(figsize=(15, 7))
    plt.plot(df['Date'], df['Close'], label='Close Price', color='lightgray')
    plt.plot(df['Date'], df['MA'], label='Moving Average', color='orange')
    plt.legend()
    st.pyplot(plt)

    # === Monte Carlo Parameters ===
n_simulations = st.sidebar.slider("Number of Monte Carlo Simulations", 100, 5000, 1000)
n_days = st.sidebar.slider("Forecast Days", 30, 365, 252)
mu = st.sidebar.slider("Expected Return (%)", -20.0, 20.0, 5.0) / 100
sigma = st.sidebar.slider("Volatility (%)", 0.0, 100.0, 20.0) / 100
crash_probability = st.sidebar.slider("Crash Probability (%)", 0.0, 10.0, 2.0) / 100
crash_impact = st.sidebar.slider("Crash Impact (%)", 0.0, 50.0, 20.0) / 100

# === Monte Carlo Simulation ===
last_price = df['Close'].iloc[-1]
simulation_data = np.zeros((n_simulations, n_days))

for sim in range(n_simulations):
    prices = [last_price]
    for day in range(n_days):
        shock = np.random.normal(mu / n_days, sigma / np.sqrt(n_days))
        crash_event = np.random.rand() < crash_probability
        if crash_event:
            shock -= crash_impact
        price = prices[-1] * (1 + shock)
        prices.append(price)
    simulation_data[sim] = prices[1:]

# === Plotting the Monte Carlo Simulations ===
plt.figure(figsize=(15, 6))
plt.plot(simulation_data.T, color='lightgray', alpha=0.3)
plt.title(f"{n_simulations} Monte Carlo Simulations Over {n_days} Days")
plt.xlabel("Days")
plt.ylabel("Simulated Price")
plt.grid(True)

# === Risk Quantiles ===
quantiles = np.percentile(simulation_data[:, -1], [5, 50, 95])
plt.axhline(quantiles[1], color='blue', linestyle='--', label='Median Prediction')
plt.axhline(quantiles[0], color='red', linestyle='--', label='5th Percentile (Lower Risk)')
plt.axhline(quantiles[2], color='green', linestyle='--', label='95th Percentile (Upper Risk)')
plt.legend()

# === Streamlit Plot Output ===
st.pyplot(plt)

# === Simulation Metrics ===
st.write(f"**5th Percentile (Conservative):** ${quantiles[0]:,.2f}")
st.write(f"**Median Price (Expected):** ${quantiles[1]:,.2f}")
st.write(f"**95th Percentile (Optimistic):** ${quantiles[2]:,.2f}")

# === Distribution of Final Prices ===
plt.figure(figsize=(10, 5))
plt.hist(simulation_data[:, -1], bins=50, color='skyblue', edgecolor='black')
plt.axvline(quantiles[1], color='blue', linestyle='--')
plt.axvline(quantiles[0], color='red', linestyle='--')
plt.axvline(quantiles[2], color='green', linestyle='--')
plt.title("Distribution of Final Simulated Prices")
plt.xlabel("Final Price")
plt.ylabel("Frequency")
st.pyplot(plt)

# === Streamlit Inputs for Sensitivity Analysis ===
if enable_sensitivity:
    st.subheader("🔎 Sensitivity Analysis for MA and RSI")

    ma_values = list(range(ma_range[0], ma_range[1] + 1, step))
    rsi_values = list(range(rsi_range[0], rsi_range[1] + 1, step))
    param_grid = list(ParameterGrid({"MA": ma_values, "RSI": rsi_values}))

    # === DataFrame to Store Results ===
    results = pd.DataFrame(columns=['MA', 'RSI', 'Net Profit (%)', 'Max Drawdown (%)'])

    # === Sensitivity Analysis Loop ===
    for params in param_grid:
        ma_window = params['MA']
        rsi_window = params['RSI']

        # === Technical Indicators Calculation ===
        df['MA'] = df['Close'].rolling(window=ma_window).mean()
        delta = df['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # === Generate Buy/Sell Signals ===
        df['Signal'] = None
        df.loc[(df['Close'] > df['MA']) & (df['RSI'] > 30), 'Signal'] = 'Buy'
        df.loc[(df['Close'] < df['MA']) & (df['RSI'] < 70), 'Signal'] = 'Sell'

        # === Backtesting Logic ===
        initial_balance = 10000
        balance = initial_balance
        holdings = 0

        for i, row in df.iterrows():
            if row['Signal'] == 'Buy' and balance > row['Close']:
                holdings = balance / row['Close']
                balance = 0
            elif row['Signal'] == 'Sell' and holdings > 0:
                balance = holdings * row['Close']
                holdings = 0

        final_value = balance + (holdings * df['Close'].iloc[-1])
        net_profit = ((final_value - initial_balance) / initial_balance) * 100
        
        # === Drawdown Calculation ===
        portfolio_values = [initial_balance]
        for i, row in df.iterrows():
            if row['Signal'] == 'Buy':
                portfolio_values.append(portfolio_values[-1] * (1 + (row['Close'] / df['Close'].iloc[i-1] - 1)))
            elif row['Signal'] == 'Sell':
                portfolio_values.append(portfolio_values[-1])
        
        max_drawdown = (1 - (min(portfolio_values) / max(portfolio_values)))

        # === Store Results ===
        new_row = pd.DataFrame([{
    'MA': ma_window,
    'RSI': rsi_window,
    'Net Profit (%)': net_profit,
    'Max Drawdown (%)': max_drawdown * 100
}])

    results = pd.concat([results, new_row], ignore_index=True)


    # === Pivot and Heatmap Visualization ===
    st.write("### 📊 Net Profit Sensitivity Analysis")
    profit_pivot = results.pivot(index='MA', columns='RSI', values='Net Profit (%)')
    plt.figure(figsize=(10, 6))
    sns.heatmap(profit_pivot, annot=True, fmt=".2f", cmap='YlGnBu')
    plt.title("Net Profit (%) by MA and RSI")
    st.pyplot(plt)

    st.write("### 📉 Max Drawdown Sensitivity Analysis")
    drawdown_pivot = results.pivot(index='MA', columns='RSI', values='Net Profit (%)')
    plt.figure(figsize=(10, 6))
    sns.heatmap(drawdown_pivot, annot=True, fmt=".2f", cmap='OrRd')
    plt.title("Max Drawdown (%) by MA and RSI")
    st.pyplot(plt)

    # === Optimal Parameters ===
    optimal_row = results.loc[results['Net Profit (%)'].idxmax()]
    st.write(f"**Optimal Parameters:** MA = {int(optimal_row['MA'])}, RSI = {int(optimal_row['RSI'])}")
    st.write(f"**Max Profit Achieved:** {optimal_row['Net Profit (%)']:.2f}%")
    st.write(f"**Drawdown at Optimal:** {optimal_row['Max Drawdown (%)']:.2f}%")
# === Performance Metrics ===
final_value = portfolio_values[-1]
profit_percent = ((final_value - initial_balance) / initial_balance) * 100
st.write(f"**Final Portfolio Value:** ${final_value:,.2f}")
st.write(f"**Net Profit:** {profit_percent:.2f}%")

# === Drawdown Calculation ===
drawdowns = [1 - (v / max(portfolio_values[:i+1])) for i, v in enumerate(portfolio_values)]
max_drawdown = max(drawdowns)
st.write(f"**Max Drawdown:** {max_drawdown * 100:.2f}%")

# === Risk Metrics ===
returns = pd.Series(portfolio_values).pct_change().dropna()
VaR = returns.quantile(0.05) * initial_balance
CVaR = returns[returns <= returns.quantile(0.05)].mean() * initial_balance
sharpe_ratio = (returns.mean() / returns.std()) * math.sqrt(252)
st.write(f"**Value-at-Risk (95%):** ${abs(VaR):,.2f}")
st.write(f"**Expected Shortfall (CVaR):** ${abs(CVaR):,.2f}")
st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")

# === Plot Portfolio Value ===
plt.figure(figsize=(15, 7))
plt.plot(portfolio_values, label='Portfolio Value', color='blue')
plt.title('Portfolio Value Over Time')
plt.xlabel('Days')
plt.ylabel('Value ($)')
plt.legend()
st.pyplot(plt)

# === Portfolio Rebalancing and Optimization ===
def portfolio_rebalance(df, initial_balance=10000, stop_loss=0.05, take_profit=0.15, frequency='Weekly'):
    balance = initial_balance
    holdings = 0
    portfolio_values = []
    rebalance_days = {
        "Daily": 1,
        "Weekly": 5,
        "Monthly": 20
    }
    rebalance_step = rebalance_days[frequency]
    
    for i, row in df.iterrows():
        if i % rebalance_step == 0 and row['Signal'] == 'Buy' and balance > row['Close']:
            # === Buy Logic ===
            holdings += balance / row['Close']
            balance = 0

        elif i % rebalance_step == 0 and row['Signal'] == 'Sell' and holdings > 0:
            # === Sell Logic ===
            balance += holdings * row['Close']
            holdings = 0

        # === Apply Stop-Loss and Take-Profit ===
        current_value = holdings * row['Close']
        if current_value > 0:
            if current_value >= (1 + take_profit) * initial_balance:
                balance += current_value
                holdings = 0
            elif current_value <= (1 - stop_loss) * initial_balance:
                balance += current_value
                holdings = 0

        portfolio_values.append(balance + (holdings * row['Close']))

    # === Calculate Performance Metrics ===
    final_value = portfolio_values[-1]
    net_profit = ((final_value - initial_balance) / initial_balance) * 100
    drawdowns = [1 - (v / max(portfolio_values[:i + 1])) for i, v in enumerate(portfolio_values)]
    max_drawdown = max(drawdowns)

    # === Display Results ===
    st.write(f"**Final Portfolio Value:** ${final_value:,.2f}")
    st.write(f"**Net Profit:** {net_profit:.2f}%")
    st.write(f"**Max Drawdown:** {max_drawdown * 100:.2f}%")
    
    # === Plot Portfolio Performance ===
    plt.figure(figsize=(15, 7))
    plt.plot(portfolio_values, label='Portfolio Value', color='blue')
    plt.title(f"Portfolio Value Over Time ({frequency} Rebalancing)")
    plt.legend()
    st.pyplot(plt)

    return portfolio_values, final_value
if optimize_allocation:
    st.write("### 🔄 Portfolio Rebalancing & Optimization Analysis")
    portfolio_values, final_value = portfolio_rebalance(df, 
                                                        initial_balance=10000, 
                                                        stop_loss=stop_loss, 
                                                        take_profit=take_profit, 
                                                        frequency=rebalance_frequency)
