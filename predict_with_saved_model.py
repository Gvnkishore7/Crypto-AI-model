# --- START OF predict_explain_from_raw.py (Simplified Output v2) ---

import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import shap
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import xgboost # Explicit import

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

print("--- Bitcoin Prediction & SHAP Explanation ---")

# --- Configuration ---
MODEL_FILE = "final_ai_model.pkl"
SCALER_FILE = "final_scaler.pkl"
FEATURES_FILE = "final_selected_features.joblib"
RAW_DATA_FILE = "btc_raw_data_yf.csv" # From download_data.py
PREDICT_FOR_DAYS = 1
TOP_N_SHAP_FEATURES = 5

# --- 1. Feature Engineering Function (Must match training!) ---
# (Function definition remains unchanged - vital for consistency)
def engineer_features(btc_df):
    # Contains all feature calculations from training script v5
    # print("Engineering features...") # Silenced
    btc_df['prev_close'] = btc_df['close'].shift(1); btc_df['prev_price_change'] = btc_df['close'].diff()
    btc_df['price_volatility_5d'] = btc_df['close'].pct_change().rolling(window=5, min_periods=1).std()
    btc_df['moving_avg_5d'] = btc_df['close'].rolling(window=5, min_periods=1).mean()
    btc_df['moving_avg_10d'] = btc_df['close'].rolling(window=10, min_periods=1).mean()
    btc_df['volume_change'] = btc_df['volume'].diff(); btc_df['return_1d'] = btc_df['close'].pct_change(1); btc_df['return_3d'] = btc_df['close'].pct_change(3)
    period_rsi = 14; delta = btc_df['close'].diff(1); gain = delta.where(delta > 0, 0).fillna(0); loss = -delta.where(delta < 0, 0).fillna(0)
    avg_gain = gain.rolling(window=period_rsi, min_periods=period_rsi).mean(); avg_loss = loss.rolling(window=period_rsi, min_periods=period_rsi).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9); btc_df['rsi'] = 100 - (100 / (1 + rs))
    exp1 = btc_df['close'].ewm(span=12, adjust=False).mean(); exp2 = btc_df['close'].ewm(span=26, adjust=False).mean()
    btc_df['macd'] = exp1 - exp2; btc_df['macd_signal'] = btc_df['macd'].ewm(span=9, adjust=False).mean(); btc_df['macd_hist'] = btc_df['macd'] - btc_df['macd_signal']
    btc_df['h-l'] = btc_df['high'] - btc_df['low']; btc_df['h-pc'] = np.abs(btc_df['high'] - btc_df['close'].shift(1)); btc_df['l-pc'] = np.abs(btc_df['low'] - btc_df['close'].shift(1))
    tr_df = pd.DataFrame({'hl': btc_df['h-l'], 'hc': btc_df['h-pc'], 'lc': btc_df['l-pc']}); btc_df['tr'] = tr_df.max(axis=1, skipna=True)
    period_atr = 14; btc_df['atr'] = btc_df['tr'].rolling(window=period_atr, min_periods=period_atr).mean()
    period_roc = 9; btc_df['roc'] = btc_df['close'].pct_change(periods=period_roc) * 100
    period_bollinger = 20; std_bollinger = 2
    btc_df['bollinger_ma'] = btc_df['close'].rolling(window=period_bollinger, min_periods=period_bollinger).mean(); btc_df['bollinger_std'] = btc_df['close'].rolling(window=period_bollinger, min_periods=period_bollinger).std()
    btc_df['bollinger_upper'] = btc_df['bollinger_ma'] + (btc_df['bollinger_std'] * std_bollinger); btc_df['bollinger_lower'] = btc_df['bollinger_ma'] - (btc_df['bollinger_std'] * std_bollinger)
    btc_df['bollinger_bandwidth'] = ((btc_df['bollinger_upper'] - btc_df['bollinger_lower']) / btc_df['bollinger_ma'].replace(0, np.nan)) * 100
    btc_df['bollinger_bandwidth'].fillna(0, inplace=True)
    if not isinstance(btc_df.index, pd.DatetimeIndex): btc_df.index = pd.to_datetime(btc_df.index)
    btc_df['day_of_week'] = btc_df.index.dayofweek; btc_df['month'] = btc_df.index.month
    btc_df.drop(['h-l', 'h-pc', 'l-pc', 'tr', 'bollinger_ma', 'bollinger_std'], axis=1, inplace=True, errors='ignore')
    # print("Features engineered.") # Silenced
    return btc_df

# --- 2. Load Raw Data & Prepare (Minimal Output) ---
try:
    raw_btc_df = pd.read_csv(RAW_DATA_FILE, parse_dates=['Date'], index_col='Date')
    if raw_btc_df.empty: exit(f"ERROR: Raw data file '{RAW_DATA_FILE}' empty.")
    raw_btc_df.rename(columns={'Close': 'close', 'High': 'high', 'Low': 'low', 'Open': 'open', 'Volume': 'volume'}, inplace=True, errors='ignore')
    required_cols = ['open', 'high', 'low', 'close', 'volume']; missing_cols = [c for c in required_cols if c not in raw_btc_df.columns]
    if missing_cols: exit(f"ERROR: Raw data missing: {missing_cols}")
    for col in required_cols:
        if raw_btc_df[col].dtype == 'object': raw_btc_df[col] = raw_btc_df[col].astype(str).str.replace(',', '', regex=False)
        raw_btc_df[col] = pd.to_numeric(raw_btc_df[col], errors='coerce')
    raw_btc_df.dropna(subset=required_cols, inplace=True)
    if raw_btc_df.empty: exit("ERROR: Raw data empty after cleaning.")
    raw_btc_df = raw_btc_df[required_cols]
    df_with_all_features = engineer_features(raw_btc_df.copy()) # Apply features
except FileNotFoundError: exit(f"ERROR: Raw data file '{RAW_DATA_FILE}' not found.")
except Exception as e: exit(f"ERROR loading/processing raw data: {e}")

# --- 3. Load Model Artifacts (Minimal Output) ---
try:
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    selected_features = joblib.load(FEATURES_FILE) # Needed for prediction step
except FileNotFoundError as e: exit(f"ERROR: Artifact file missing: {e}.")
except Exception as e: exit(f"ERROR loading artifacts: {e}")

# --- 4. Prepare Latest Data Point(s) (Minimal Output) ---
potential_features_pred = [ # Full feature list for scaler (must match training)
    'close', 'high', 'low', 'open', 'volume', 'prev_close', 'prev_price_change',
    'price_volatility_5d', 'moving_avg_5d', 'moving_avg_10d', 'volume_change',
    'return_1d', 'return_3d', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'atr', 'roc',
    'bollinger_upper', 'bollinger_lower', 'bollinger_bandwidth', 'day_of_week', 'month'
]
data_prepared = df_with_all_features.dropna(subset=potential_features_pred).copy()
if data_prepared.empty or data_prepared.shape[0] < PREDICT_FOR_DAYS: exit(f"ERROR: Not enough valid data ({data_prepared.shape[0]}) after feature NaN removal.")

latest_feature_data = data_prepared.tail(PREDICT_FOR_DAYS)
input_date_for_pred = latest_feature_data.index.max()
# print(f"\nUsing data from: {input_date_for_pred.date()} to predict.") # Silenced

missing_potential = [f for f in potential_features_pred if f not in latest_feature_data.columns];
if missing_potential: exit(f"ERROR: Features for scaler missing: {missing_potential}")
features_for_scaling = latest_feature_data[potential_features_pred]
scaled_features = scaler.transform(features_for_scaling)
scaled_features_df = pd.DataFrame(scaled_features, index=features_for_scaling.index, columns=potential_features_pred)
missing_model = [f for f in selected_features if f not in scaled_features_df.columns];
if missing_model: exit(f"ERROR: Model features missing after scaling: {missing_model}")
data_final_predict = scaled_features_df[selected_features]

# --- 5. Make Prediction (Simplified Output) ---
print("\n--- Prediction ---")
try:
    predictions = model.predict(data_final_predict)
    probabilities = model.predict_proba(data_final_predict)
except Exception as e: exit(f"ERROR during prediction: {e}")

# Process and print the last/only prediction
last_pred_idx = -1
input_date = data_final_predict.index[last_pred_idx]
prediction_date = input_date + timedelta(days=1)
pred_value = predictions[last_pred_idx]
pred_prob = probabilities[last_pred_idx]
signal = "BUY" if pred_value == 1 else "SELL/HOLD"
prob_buy = pred_prob[1]; prob_sell = pred_prob[0]

print(f"Input Data Date   : {input_date.date()}")
print(f"Prediction For Date : {prediction_date.date()}")
print(f"Predicted Signal    : {signal}")
print(f"Signal Confidence   : Buy = {prob_buy:.1%}, Sell/Hold = {prob_sell:.1%}") # Simplified % format

# --- 6. Explain Prediction with SHAP (Text Only + Save Plot) ---
print("\n--- Prediction Explanation (Top Factors) ---")
try:
    explainer = shap.TreeExplainer(model)
    shap_values_instance = explainer.shap_values(data_final_predict.iloc[[last_pred_idx]])
    shap_df = pd.DataFrame({ 'feature': selected_features, 'feature_value_scaled': data_final_predict.iloc[last_pred_idx].values, 'shap_value': shap_values_instance[0] })
    shap_df['abs_shap_value'] = np.abs(shap_df['shap_value']); shap_df = shap_df.sort_values(by='abs_shap_value', ascending=False)

    print(f"Top {TOP_N_SHAP_FEATURES} factors influencing the '{signal}' prediction:")
    for idx, row in shap_df.head(TOP_N_SHAP_FEATURES).iterrows():
        direction = "->" if row['shap_value'] > 0 else "<-" # Simpler arrow indicator
        # Reduced detail in print statement
        print(f"  - {row['feature']:<22} (SHAP: {row['shap_value']:>6.2f} {direction})")

    # --- Generate & Save SHAP Plot ---
    try: # Nested try-except for plotting part only
        plt.figure(figsize=(16, 4)) # Keep size reasonable for saving
        force_plot = shap.force_plot(explainer.expected_value, shap_values_instance[0], data_final_predict.iloc[last_pred_idx], feature_names=selected_features, matplotlib=True, show=False)
        plt.title(f"SHAP Explanation {prediction_date:%Y-%m-%d} ({signal})", fontsize=11)
        plt.savefig("shap_force_plot.png", dpi=100, bbox_inches='tight') # Save figure
        plt.close() # Close figure to prevent display
        print(f"\n(SHAP graphical explanation saved to: shap_force_plot.png)")
    except Exception as plot_err:
        print(f"\nWarning: Could not save SHAP plot: {plot_err}")

except ImportError: print("SHAP requires matplotlib installed.")
except Exception as e: print(f"ERROR generating SHAP data: {e}")

# --- 7. Comparison Clarification (Concise) ---
print("\n--- Performance Context ---")
print(f"- Model Suggestion for {prediction_date.date()}: {signal}")
print("- Buy & Hold Strategy : Simply continues holding.")
print("- Overall Reliability : See Accuracy/F1/Profit Plot from the *training script's test set*.")

print("\n--- Script Finished ---")
# --- END OF SCRIPT ---