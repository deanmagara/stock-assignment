# Volatility Guard: S&P 500 Market Regime Classifier

Volatility Guard is a machine learning project designed to classify S&P 500 market conditions as either **stable** or **volatile**. The goal of this project is not simply to maximize return, but to reduce downside risk by helping investors avoid highly volatile market periods.

This project uses historical S&P 500 price data and transforms it into technical indicators such as moving average crossover, ATR, RSI, and Bollinger Band position. These features are then used to train a Support Vector Machine classifier with an RBF kernel. The model predicts whether the market is entering a volatile regime, and the prediction is later tested through a simple backtesting strategy.

The main workflow includes three parts: data processing, model training, and strategy backtesting. The data pipeline loads the raw S&P 500 CSV file, calculates daily returns, creates a 20-day rolling volatility target, and engineers technical indicators for model training. The training script uses a chronological train/test split to avoid looking into future data, applies StandardScaler, and tunes the SVM model using GridSearchCV. The backtest compares the Volatility Guard strategy against a normal buy-and-hold strategy.

In the final backtest, the buy-and-hold strategy achieved a higher total return, but Volatility Guard significantly reduced the maximum drawdown. Buy and hold returned **252.76%** with a maximum drawdown of **-56.78%**, while Volatility Guard returned **57.07%** with a much smaller maximum drawdown of **-14.14%**. This shows that the model is more useful as a risk-management tool than as a pure return-maximizing strategy.

Overall, Volatility Guard demonstrates how machine learning can be applied to financial market regime detection. By identifying periods of high volatility, the model provides a defensive trading signal that may help reduce exposure during risky market conditions.
