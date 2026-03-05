import argparse
import sys
import pandas as pd

# Import the functions we built in the other scripts
try:
    from data_pipeline import load_kaggle_data, engineer_features
    from train import load_and_prepare_data, train_and_tune_svm, evaluate_model
    from backtest import run_backtest
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure data_pipeline.py, train.py, and backtest.py are in the same folder as app.py.")
    sys.exit(1)

def print_banner():
    """Prints a terminal banner for the application."""
    print("="*50)
    print(" THE VOLATILITY GUARD ".center(50))
    print(" S&P 500 Market Regime Classifier ".center(50))
    print("="*50)

def main():
    parser = argparse.ArgumentParser(
        description="The Volatility Guard: Detecting Market Regimes in the S&P 500."
    )
    
    # Create subparsers for our different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Command 1: Pipeline (Fetch & Engineer Data)
    parser_pipeline = subparsers.add_parser("pipeline", help="Run the data pipeline and feature engineering.")
    parser_pipeline.add_argument("--file", type=str, required=True, help="Path to your downloaded Kaggle CSV file.")
    
    # Command 2: Train (Train and Tune the SVM)
    parser_train = subparsers.add_parser("train", help="Train the Support Vector Machine (SVM) model.")
    parser_train.add_argument("--data", type=str, default="sp500_engineered_data.csv", help="Path to the engineered CSV data.")
    
    # Command 3: Backtest (Run the strategy simulation)
    parser_backtest = subparsers.add_parser("backtest", help="Backtest the 'Guard' strategy against Buy & Hold.")
    parser_backtest.add_argument("--data", type=str, default="sp500_engineered_data.csv", help="Path to the engineered CSV data.")

    args = parser.parse_args()
    
    # If no command is passed, print help
    if args.command is None:
        print_banner()
        parser.print_help()
        sys.exit(0)

    print_banner()

    # Route the command to the correct functions
    if args.command == "pipeline":
        print(f"Starting Data Pipeline using source: {args.file}\n")
        try:
            raw_data = load_kaggle_data(filepath=args.file)
            processed_data = engineer_features(raw_data)
            processed_data.to_csv("sp500_engineered_data.csv")
            print("\nSUCCESS: Data saved to 'sp500_engineered_data.csv'. Ready for training!")
        except Exception as e:
            print(f"\nPipeline Error: {e}")
            
    elif args.command == "train":
        print("Starting Model Training Phase...\n")
        try:
            X_train, X_test, y_train, y_test = load_and_prepare_data(filepath=args.data)
            best_svm, trained_scaler = train_and_tune_svm(X_train, y_train)
            evaluate_model(best_svm, trained_scaler, X_test, y_test)
            print("\nSUCCESS: Model and Scaler saved. Ready for backtesting!")
        except Exception as e:
            print(f"\nTraining Error: {e}")
            
    elif args.command == "backtest":
        print("Starting Backtest Simulation...\n")
        try:
            run_backtest(data_filepath=args.data)
        except Exception as e:
            print(f"\nBacktesting Error: {e}")

if __name__ == "__main__":
    main()