import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def load_and_prepare_data(filepath="sp500_engineered_data.csv"):
    """Loads engineered data and performs a chronological train/test split."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
    
    # Features generated from the Kaggle pipeline
    features = ['MA_Crossover', 'ATR_14', 'RSI_14', 'BB_Position']
    
    X = df[features]
    y = df['Target']
    
    # Chronological train/test split (80% train, 20% test) to prevent looking into the future
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training set: {len(X_train)} days")
    print(f"Testing set: {len(X_test)} days")
    
    return X_train, X_test, y_train, y_test

def train_and_tune_svm(X_train, y_train):
    """Scales data and tunes an RBF SVM using GridSearch."""
    print("\nApplying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    joblib.dump(scaler, 'volatility_scaler.pkl')
    
    print("Initializing SVM with RBF Kernel...")
    svm = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.1, 1]
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    print("Running Grid Search for hyperparameter tuning...")
    grid_search = GridSearchCV(svm, param_grid, cv=tscv, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    
    joblib.dump(best_model, 'volatility_guard_svm.pkl')
    print("Model saved to 'volatility_guard_svm.pkl'")
    
    return best_model, scaler

def evaluate_model(model, scaler, X_test, y_test):
    """Evaluates the model on the unseen test set."""
    print("\n--- Model Evaluation ---")
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    print(classification_report(y_test, y_pred, target_names=['Stable (0)', 'Volatile (1)']))
    
    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Stable (TN): {cm[0][0]}  | False Volatile (FP): {cm[0][1]}")
    print(f"False Stable (FN): {cm[1][0]} | True Volatile (TP): {cm[1][1]}")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    best_svm, trained_scaler = train_and_tune_svm(X_train, y_train)
    evaluate_model(best_svm, trained_scaler, X_test, y_test)