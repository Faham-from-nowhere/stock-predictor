# 📈 Stock Price Prediction, Signals & Backtesting

An interactive Streamlit application that predicts stock prices using an LSTM model, generates buy/sell signals with technical indicators (MA & RSI), performs backtesting, and evaluates strategy risk and performance through optimization, Value-at-Risk (VaR), CVaR, Sharpe Ratio, and Monte Carlo simulations.

## 🚀 Features

- 📊 **LSTM-based Stock Price Prediction**
- 📈 **Technical Indicators:**
  - Moving Average (MA)
  - Relative Strength Index (RSI)
- 🛠️ **Customizable Strategy Parameters**
- 🔍 **Backtesting with Portfolio Simulation**
- 📉 **Risk Metrics:**
  - Max Drawdown
  - Sharpe Ratio
  - Value-at-Risk (VaR)
  - Conditional Value-at-Risk (CVaR)
- 🧪 **Strategy Optimization** (Grid search for best MA/RSI)
- 🎲 **Monte Carlo Simulation** for future portfolio projection
- 📌 **Sensitivity Analysis** for parameter impact


## 🛠️ Requirements

- Python 3.8+
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- Streamlit

### Install Dependencies with pip install requirements.txt



📁 Input Files
📄 CSV Stock Data: Must contain at least these columns:
Date, Open, High, Low, Close, Volume
🤖 Trained LSTM Model: .h5 file trained on normalized data using features: [Open, High, Low, Close, Volume, MA, RSI]
To run streamlit run app.py

Once launched:

->Upload your stock CSV data.

->Upload your trained LSTM .h5 model.

->Configure parameters on the sidebar (MA window, RSI window, Risk settings, etc.).

->View predictions, signals, backtest results, and risk metrics.

📊 Sample Output
Buy/Sell signals over time

Final portfolio value

Net profit (%)

Max drawdown (%)

Sharpe Ratio

Monte Carlo simulation projection

Optimal MA/RSI combination from grid search

📈 Monte Carlo Simulation Example
Visualizes multiple portfolio value paths over time to assess uncertainty and potential outcomes.

🧠 Model Training (Optional)
Training is done separately using an LSTM model on historical stock data after computing MA & RSI and scaling features. The trained .h5 model is then used in this app.

You can use MinMaxScaler and Sequential model with layers:
LSTM ➝ Dropout ➝ Dense(1) for Close price prediction.


