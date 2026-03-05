import pandas as pd
import numpy as np
import joblib

def calculate_max_drawdown(cumulative_returns):
    """Calculates the maximum drawdown of a return series."""
    rolling_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / rolling_max - 1.0
    return drawdown.min()

def run_backtest(data_filepath="sp500_engineered_data.csv"):
    """Simulates the Volatility Guard strategy against Buy & Hold."""
    print("Loading data and trained model...")
    df = pd.read_csv(data_filepath, index_col='Date', parse_dates=True)
    
    try:
        model = joblib.load('volatility_guard_svm.pkl')
        scaler = joblib.load('volatility_scaler.pkl')
    except FileNotFoundError:
        print("Error: Model or scaler not found. Please run train.py first.")
        return

    # Isolate the test set exactly as we did in train.py (last 20%)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    
    features = ['MA_Crossover', 'ATR_14', 'RSI_14', 'BB_Position']
    X_test = test_df[features]
    
    # Generate Predictions
    print("Generating market regime predictions...")
    X_test_scaled = scaler.transform(X_test)
    test_df['Predicted_Regime'] = model.predict(X_test_scaled)
    
    # --- Strategy Simulation ---
    # Shift predictions by 1 day to simulate executing the trade at the NEXT day's open
    # (You can't trade on today's close using today's close to predict)
    test_df['Signal'] = test_df['Predicted_Regime'].shift(1)
    
    # Strategy: If Signal == 1 (Volatile), hold cash (0 return). If Signal == 0 (Stable), hold SP500.
    test_df['Strategy_Returns'] = np.where(test_df['Signal'] == 0, test_df['Returns'], 0)
    
    # Drop the first row (NaN due to shifting)
    test_df.dropna(subset=['Signal'], inplace=True)
    
    # Calculate Cumulative Returns
    test_df['Cumulative_Market'] = (1 + test_df['Returns']).cumprod()
    test_df['Cumulative_Strategy'] = (1 + test_df['Strategy_Returns']).cumprod()
    
    # Calculate Metrics
    market_return = (test_df['Cumulative_Market'].iloc[-1] - 1) * 100
    strategy_return = (test_df['Cumulative_Strategy'].iloc[-1] - 1) * 100
    
    market_mdd = calculate_max_drawdown(test_df['Cumulative_Market']) * 100
    strategy_mdd = calculate_max_drawdown(test_df['Cumulative_Strategy']) * 100
    
    # --- Output Results ---
    print("\n" + "="*40)
    print(" BACKTEST RESULTS (TEST SET)".center(40))
    print("="*40)
    print(f"Total Trading Days: {len(test_df)}")
    print(f"Days Invested (Guard): {len(test_df[test_df['Signal'] == 0])}")
    print(f"Days in Cash (Guard):  {len(test_df[test_df['Signal'] == 1])}")
    print("-" * 40)
    print(f"Buy & Hold Return:   {market_return:.2f}%")
    print(f"Vol. Guard Return:   {strategy_return:.2f}%")
    print("-" * 40)
    print(f"Buy & Hold Max Drawdown: {market_mdd:.2f}%")
    print(f"Vol. Guard Max Drawdown: {strategy_mdd:.2f}%")
    print("="*40)
    print("\nNote: A successful strategy should show a significantly lower Max Drawdown,")
    print("even if total returns are slightly lower or comparable.")

if __name__ == "__main__":
    run_backtest()