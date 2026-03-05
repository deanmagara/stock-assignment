import pandas as pd
import numpy as np

def load_kaggle_data(filepath="sp500_kaggle.csv"):
    """Loads S&P 500 data from a local Kaggle CSV file."""
    print(f"Loading data from {filepath}...")
    
    # Load CSV, assuming there is a 'Date' column
    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    
    # Standardize column names to match expectations (Capitalized)
    df.columns = df.columns.str.capitalize()
    
    # Keep only the required base features
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[required_cols].copy()
    
    # Sort index to ensure chronological order (crucial for time-series)
    df.sort_index(inplace=True)
    
    print(f"Data loaded successfully. Total trading days: {len(df)}")
    return df

def engineer_features(df):
    """Calculates technical indicators and the rolling variance target."""
    print("Engineering features...")
    data = df.copy()

    # 1. Daily Returns
    data['Returns'] = data['Close'].pct_change()

    # 2. Target Variable: 20-day rolling standard deviation of returns
    data['Rolling_Std_20'] = data['Returns'].rolling(window=20).std()
    
    # Define "Volatile" (1) as days where the rolling standard dev is in the top 25% (High Vol)
    volatility_threshold = data['Rolling_Std_20'].quantile(0.75)
    data['Target'] = np.where(data['Rolling_Std_20'] > volatility_threshold, 1, 0)

    # 3. Moving Average Crossovers (50-day & 200-day SMA)
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['MA_Crossover'] = np.where(data['SMA_50'] > data['SMA_200'], 1, -1) # 1 = Bullish, -1 = Bearish

    # 4. Average True Range (ATR) - 14-day
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    data['ATR_14'] = true_range.rolling(14).mean()

    # 5. Relative Strength Index (RSI) - 14-day
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI_14'] = 100 - (100 / (1 + rs))

    # 6. Bollinger Bands (20-day SMA +/- 2 Std Dev)
    data['BB_Mid'] = data['Close'].rolling(window=20).mean()
    data['BB_Std'] = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Mid'] + (data['BB_Std'] * 2)
    data['BB_Lower'] = data['BB_Mid'] - (data['BB_Std'] * 2)
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])

    # Drop rows with NaN values created by rolling windows (e.g., first 200 days for SMA_200)
    data.dropna(inplace=True)
    
    print("Feature engineering complete.")
    return data

if __name__ == "__main__":
    # Point this to your actual Kaggle file name!
    kaggle_file = "your_kaggle_dataset.csv" 
    
    try:
        raw_data = load_kaggle_data(filepath=kaggle_file)
        processed_data = engineer_features(raw_data)
        
        processed_data.to_csv("sp500_engineered_data.csv")
        print("\nData saved to 'sp500_engineered_data.csv'. Ready for model training!")
        
        print("\nClass Balance (0 = Low Vol, 1 = High Vol):")
        print(processed_data['Target'].value_counts(normalize=True))
    except FileNotFoundError:
        print(f"Error: Could not find '{kaggle_file}'. Please ensure the file is in the same directory as this script.")