# --- START OF FINAL SCRIPT (Reverted v5 Logic, Simplified Output, Show Plots) ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# --- Filter Warnings ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# --- Imports ---
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib
import shap
import time
import xgboost as xgb

print("--- Bitcoin Trading Strategy Analysis ---")
if hasattr(shap, '__version__'): print(f"SHAP version: {shap.__version__}")

# --- Configuration ---
# <<< POINT THIS TO YOUR YFINANCE CSV >>>
btc_file = "btc_data_5y_daily_yf.csv"

# --- Model & Tuning Parameters ---
test_size_ratio = 0.2
top_n_features = 12 # From the version that gave ~56%
N_SPLITS_CV = 5
RUN_HYPERPARAM_SEARCH = True
N_SEARCH_ITERATIONS = 50

# --- Simulation Parameters ---
initial_balance = 10000

# --- 1. Data Loading (Specific to yfinance export with skiprows=3) ---
print(f"\n--- 1. Loading Data: '{btc_file}' ---")
expected_columns = ['date', 'close', 'high', 'low', 'open', 'volume']
try:
    btc_df = pd.read_csv(btc_file, skiprows=3, header=None, names=expected_columns)
    btc_df['date'] = pd.to_datetime(btc_df['date'], errors='coerce')
    required_cols = ['date', 'close', 'high', 'low', 'open', 'volume']
    missing_cols = [col for col in required_cols if col not in btc_df.columns];
    if missing_cols: exit(f"ERROR: Missing columns: {missing_cols}")
    for col in required_cols[1:]: # Numeric conversion
        if btc_df[col].dtype == 'object': btc_df[col] = btc_df[col].astype(str).str.replace(',', '', regex=False)
        btc_df[col] = pd.to_numeric(btc_df[col], errors='coerce')
    btc_df = btc_df.dropna(subset=required_cols)
    btc_df = btc_df.sort_values('date'); btc_df = btc_df.drop_duplicates(subset=['date'], keep='first')
    btc_df = btc_df.set_index('date', drop=False) # Set date index
    print(f"   Data Loaded & Preprocessed. Shape: {btc_df.shape}. Range: {btc_df.index.min().date()} to {btc_df.index.max().date()}")
except Exception as e: exit(f"ERROR loading/processing data: {e}")

# --- 2. Feature Engineering ---
print("\n--- 2. Engineering Features ---")
target = 'Trade_Signal'
# START Feature Calculation Block (No Sentiment Features Here)
btc_df['Next_Close'] = btc_df['close'].shift(-1); btc_df['Price_Change'] = btc_df['Next_Close'] - btc_df['close']
btc_df.dropna(subset=['Price_Change'], inplace=True); btc_df[target] = btc_df['Price_Change'].apply(lambda x: 1 if x > 0 else 0)
btc_df['prev_close'] = btc_df['close'].shift(1); btc_df['prev_price_change'] = btc_df['close'].diff()
btc_df['price_volatility_5d'] = btc_df['close'].pct_change().rolling(window=5, min_periods=1).std()
btc_df['moving_avg_5d'] = btc_df['close'].rolling(window=5, min_periods=1).mean(); btc_df['moving_avg_10d'] = btc_df['close'].rolling(window=10, min_periods=1).mean()
btc_df['volume_change'] = btc_df['volume'].diff(); btc_df['return_1d'] = btc_df['close'].pct_change(1); btc_df['return_3d'] = btc_df['close'].pct_change(3)
period_rsi = 14; delta = btc_df['close'].diff(1); gain = delta.where(delta > 0, 0).fillna(0); loss = -delta.where(delta < 0, 0).fillna(0); avg_gain = gain.rolling(window=period_rsi, min_periods=period_rsi).mean(); avg_loss = loss.rolling(window=period_rsi, min_periods=period_rsi).mean(); rs = avg_gain / avg_loss.replace(0, 1e-9); btc_df['rsi'] = 100 - (100 / (1 + rs))
exp1 = btc_df['close'].ewm(span=12, adjust=False).mean(); exp2 = btc_df['close'].ewm(span=26, adjust=False).mean(); btc_df['macd'] = exp1 - exp2; btc_df['macd_signal'] = btc_df['macd'].ewm(span=9, adjust=False).mean(); btc_df['macd_hist'] = btc_df['macd'] - btc_df['macd_signal']
btc_df['h-l'] = btc_df['high'] - btc_df['low']; btc_df['h-pc'] = np.abs(btc_df['high'] - btc_df['close'].shift(1)); btc_df['l-pc'] = np.abs(btc_df['low'] - btc_df['close'].shift(1)); tr_df = pd.DataFrame({'hl': btc_df['h-l'], 'hc': btc_df['h-pc'], 'lc': btc_df['l-pc']}); btc_df['tr'] = tr_df.max(axis=1, skipna=True)
period_atr = 14; btc_df['atr'] = btc_df['tr'].rolling(window=period_atr, min_periods=period_atr).mean()
period_roc = 9; btc_df['roc'] = btc_df['close'].pct_change(periods=period_roc) * 100
period_bollinger = 20; std_bollinger = 2; btc_df['bollinger_ma'] = btc_df['close'].rolling(window=period_bollinger, min_periods=period_bollinger).mean(); btc_df['bollinger_std'] = btc_df['close'].rolling(window=period_bollinger, min_periods=period_bollinger).std(); btc_df['bollinger_upper'] = btc_df['bollinger_ma'] + (btc_df['bollinger_std'] * std_bollinger); btc_df['bollinger_lower'] = btc_df['bollinger_ma'] - (btc_df['bollinger_std'] * std_bollinger); btc_df['bollinger_bandwidth'] = ((btc_df['bollinger_upper'] - btc_df['bollinger_lower']) / btc_df['bollinger_ma'].replace(0, np.nan)) * 100; btc_df['bollinger_bandwidth'].fillna(0, inplace=True)
if isinstance(btc_df.index, pd.DatetimeIndex): btc_df['day_of_week'] = btc_df.index.dayofweek; btc_df['month'] = btc_df.index.month
else: btc_df['day_of_week'] = -1; btc_df['month'] = -1
btc_df.drop(['h-l', 'h-pc', 'l-pc', 'tr', 'bollinger_ma', 'bollinger_std', 'Next_Close', 'Price_Change'], axis=1, inplace=True, errors='ignore')
# END Feature Calculation Block
potential_features = [col for col in btc_df.columns if col not in ['date', target]] # Define from engineered columns
initial_rows = btc_df.shape[0]; btc_df.dropna(subset=potential_features, inplace=True); final_rows = btc_df.shape[0]
print(f"   Features Engineered. Final shape: {btc_df.shape} (Dropped {initial_rows - final_rows} NaN rows).")
if btc_df.empty: exit("ERROR: No data after NaN removal.")
data_for_model = btc_df

# --- 3. Time Series Split & Setup ---
print("\n--- 3. Splitting Data ---")
X = data_for_model[potential_features]; y = data_for_model[target]
split_index = int(len(X) * (1 - test_size_ratio))
if split_index < N_SPLITS_CV * 2: exit("ERROR: Training set too small for CV.")
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
print(f"   Train: {X_train.shape[0]} samples ({X_train.index.min().date()} to {X_train.index.max().date()})")
print(f"   Test : {X_test.shape[0]} samples ({X_test.index.min().date()} to {X_test.index.max().date()})")
target_counts = y_train.value_counts(); target_norm = y_train.value_counts(normalize=True)
scale_pos_weight_calc = target_counts.get(0, 1) / target_counts.get(1, 1) # Handle missing class

# --- 4. Scaling ---
scaler = StandardScaler(); X_train_scaled = scaler.fit_transform(X_train); X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=potential_features, index=X_train.index); X_test_scaled = pd.DataFrame(X_test_scaled, columns=potential_features, index=X_test.index)

# --- 5. Feature Importance & Selection ---
print("\n--- 4. Feature Importance & Selection ---")
temp_model = XGBClassifier(random_state=42, use_label_encoder=False); temp_model.fit(X_train_scaled, y_train, verbose=False)
importances = temp_model.feature_importances_; indices = np.argsort(importances)[::-1]
selected_features = [potential_features[i] for i in indices[:top_n_features]]; selected_features = [f for i, f in enumerate(selected_features) if importances[indices[i]] > 1e-9]
if not selected_features: print("WARNING: No important features! Using top N."); selected_features = potential_features[:top_n_features] # Fallback
print(f"   Selected Top {len(selected_features)} Features: {selected_features}")
X_train_selected = X_train_scaled[selected_features]; X_test_selected = X_test_scaled[selected_features]

# --- 6. Hyperparameter Tuning (XGBoost) ---
print(f"\n--- 5. Hyperparameter Tuning (RandomizedSearch) ---")
final_ai_model = None
if RUN_HYPERPARAM_SEARCH:
    start_time = time.time()
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV); xgb_tune_model = XGBClassifier(random_state=42, use_label_encoder=False)
    param_dist = { 'n_estimators': [100, 200, 300, 400, 500], 'learning_rate': [0.01, 0.05, 0.1, 0.15], 'max_depth': [3, 5, 7, 9, 11], 'subsample': [0.7, 0.8, 0.9, 1.0], 'colsample_bytree': [0.7, 0.8, 0.9, 1.0], 'gamma': [0, 0.1, 0.2, 0.3, 0.4], 'min_child_weight': [1, 3, 5, 7], 'scale_pos_weight': [1, scale_pos_weight_calc] }
    search = RandomizedSearchCV( estimator=xgb_tune_model, param_distributions=param_dist, n_iter=N_SEARCH_ITERATIONS, cv=tscv, scoring='f1_weighted', n_jobs=1, verbose=0, refit=True, random_state=42 )
    try: search.fit(X_train_selected, y_train); print(f"   Tuning Complete ({time.time() - start_time:.1f}s). Best CV F1: {search.best_score_:.4f}"); print(f"   Best Params: {search.best_params_}"); final_ai_model = search.best_estimator_
    except Exception as e: print(f"   ERROR during Search: {e}. Using default."); RUN_HYPERPARAM_SEARCH = False
if not RUN_HYPERPARAM_SEARCH or final_ai_model is None:
    print("   Using Default Model Parameters..."); final_ai_model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42, scale_pos_weight=scale_pos_weight_calc, use_label_encoder=False, eval_metric='logloss', early_stopping_rounds=20)
    final_ai_model.fit(X_train_selected, y_train, eval_set=[(X_test_selected, y_test)], verbose=False)
if final_ai_model is None: exit("CRITICAL ERROR: No model.")

# --- 7. Evaluate Model (Predictive Metrics) ---
print("\n--- 6. FINAL MODEL EVALUATION (Test Set) ---")
y_pred_ai = final_ai_model.predict(X_test_selected); accuracy_ai = accuracy_score(y_test, y_pred_ai)
report_ai = classification_report(y_test, y_pred_ai, target_names=["SELL/HOLD (0)", "BUY (1)"], digits=3)
print(f"\n   >>> Accuracy: {accuracy_ai:.3f} <<<")
print("   Classification Report:\n", report_ai)

# --- 8. Explain Model (SHAP Summary) ---
print("\n--- 7. MODEL EXPLANATION (SHAP Summary) ---")
print("   Generating SHAP summary plot (might take a moment)...")
try:
    explainer = shap.TreeExplainer(final_ai_model); shap_values = explainer.shap_values(X_test_selected)
    plt.figure()
    shap.summary_plot(shap_values, X_test_selected, feature_names=selected_features, show=False, plot_size=(10, max(5, len(selected_features)*0.4)))
    plt.title(f'SHAP Summary - Test Set Feature Importance & Impact', fontsize=12); plt.tight_layout(pad=1.0)
    print("   >>> Displaying SHAP Summary Plot... <<<")
    plt.show(block=True) # Show SHAP summary
    plt.close()
except Exception as e: print(f"   ERROR generating SHAP summary plot: {e}")

# --- 9. Profitability Simulation & Plot ---
print("\n--- 8. PROFITABILITY SIMULATION (Test Period) ---")
test_data_profit = data_for_model.loc[X_test.index].copy(); test_data_profit['ai_signal'] = y_pred_ai
# Buy & Hold
btc_hold_start_price = test_data_profit['close'].iloc[0]; btc_hold_end_price = test_data_profit['close'].iloc[-1]; btc_hold_final = initial_balance * (btc_hold_end_price / btc_hold_start_price) if btc_hold_start_price > 0 else initial_balance; btc_hold_profit_pct = (btc_hold_final / initial_balance - 1) * 100
# AI Sim
ai_balance = initial_balance; ai_portfolio_value = [initial_balance]
for i in range(1, len(test_data_profit)):
    prev_signal = test_data_profit['ai_signal'].iloc[i-1]; price_prev_day = test_data_profit['close'].iloc[i-1]; price_curr_day = test_data_profit['close'].iloc[i]; daily_return = (price_curr_day / price_prev_day) - 1 if price_prev_day > 0 else 0
    if prev_signal == 1: ai_balance *= (1 + daily_return)
    ai_portfolio_value.append(ai_balance)
ai_final_balance = ai_balance; ai_profit_pct = (ai_final_balance / initial_balance - 1) * 100
# Summary Print
print("\n   *** Simulation Results ***")
profitability_results = pd.DataFrame([ {"Strategy": "Buy and Hold", "Final ($)": btc_hold_final, "Profit (%)": btc_hold_profit_pct}, {"Strategy": "AI Model", "Final ($)": ai_final_balance, "Profit (%)": ai_profit_pct} ])
profitability_results['Final ($)'] = profitability_results['Final ($)'].map('${:,.2f}'.format); profitability_results['Profit (%)'] = profitability_results['Profit (%)'].map('{:.2f}%'.format)
print(profitability_results.to_string(index=False))
# Plotting
print("\n   Generating Profitability Plot...")
plt.style.use('seaborn-v0_8-darkgrid'); plt.figure(figsize=(12, 6))
buy_hold_values = initial_balance * (test_data_profit['close'] / btc_hold_start_price) if btc_hold_start_price > 0 else pd.Series([initial_balance]*len(test_data_profit), index=test_data_profit.index)
plt.plot(test_data_profit.index, buy_hold_values, label="Buy and Hold", linestyle="--", color="dodgerblue", marker='.', markersize=4, alpha=0.7)
plt.plot(test_data_profit.index[1:], ai_portfolio_value[1:], label=f"AI Model (Top {len(selected_features)})", linestyle="-", color="forestgreen", marker='.', markersize=4, alpha=0.8)
plt.xlabel("Date (Test Period)"); plt.ylabel(f"Portfolio Value ($) - Start ${initial_balance:,.0f}")
plt.title("AI Model vs. Buy & Hold - Simulated Profitability (Test Period)"); plt.legend(fontsize=9)
plt.grid(True, linestyle=':', alpha=0.6); plt.xticks(rotation=30, ha='right'); plt.tight_layout()
print("   >>> Displaying Profitability Plot... <<<")
plt.show(block=True) # Show Profit plot
plt.close()

# --- 10. Save Artifacts ---
print("\n--- 9. Saving Final Model Components ---")
final_model_filename = "final_ai_model.pkl"; scaler_filename = "final_scaler.pkl"; selected_features_filename = "final_selected_features.joblib"
try: joblib.dump(final_ai_model, final_model_filename); joblib.dump(scaler, scaler_filename); joblib.dump(selected_features, selected_features_filename); print("   Model, Scaler, Features saved.")
except Exception as e: print(f"   ERROR saving components: {e}")

# --- Final Summary ---
print("\n" + "="*30 + " KEY RESULTS SUMMARY " + "="*30)
print(f"DATASET               : {btc_file}")
print(f"ANALYSIS RANGE        : {data_for_model.index.min().date()} to {data_for_model.index.max().date()}")
print(f"TEST PERIOD           : {X_test.index.min().date()} to {X_test.index.max().date()} ({X_test.shape[0]} days)")
print("--- METHODOLOGY ---")
print(f"FEATURE ENGINEERING   : Technical Indicators (MAs, Vol, RSI, MACD, ATR, ROC, BBands), Time")
print(f"FEATURE SELECTION   : Top {len(selected_features)} features via XGBoost Importance")
print(f"MODEL                 : XGBoost Classifier")
print(f"TUNING                : RandomizedSearchCV (f1_weighted scoring, TimeSeriesSplit CV)")
print(f"EXPLAINABILITY        : SHAP Summary Plot generated and displayed.")
print("--- PERFORMANCE (Test Set) ---")
print(f"ACCURACY              : {accuracy_ai:.3f}")
print("PRECISION/RECALL/F1   : See Classification Report above")
print("SIMULATED PROFIT (%):")
print(profitability_results[['Strategy', 'Profit (%)']].to_string(index=False))
print("="*80)


print("\n--- Analysis Script Complete ---")
# --- END OF FILE ---