import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import time

# 警告メッセージを非表示
warnings.filterwarnings("ignore")

# 1. テーブルデータ系モデルのライブラリ
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 2. 一般的な時系列予測モデルのライブラリ
# prophet, statsmodels, tbats をインストールしてください
# pip install prophet statsmodels tbats
from prophet import Prophet
from statsmodels.tsa.api import ExponentialSmoothing, ARIMA
from tbats import TBATS

# ───────────────────────────────────────────────
# 0. 定数設定
# ───────────────────────────────────────────────
TICKERS = ["AAPL", "^GSPC"]
START_DATE = "2010-01-01"
END_DATE = "2023-12-31"
TARGET_TICKER = "AAPL"
TARGET_COL = f"{TARGET_TICKER}_Close"
TEST_SIZE_RATIO = 0.3

# ───────────────────────────────────────────────
# 1. データ取得・前処理
# ───────────────────────────────────────────────
print("1. データ取得・前処理を開始...")
df = yf.download(TICKERS, start=START_DATE, end=END_DATE, progress=False)

# 特徴量エンジニアリング（テーブルデータ系モデル用）
data_for_tabular = pd.concat([
    df['Close'].rename(columns=lambda c: f"{c}_Close"),
    df['Volume'].rename(columns=lambda c: f"{c}_Volume")
], axis=1)

for t in TICKERS:
    data_for_tabular[f'{t}_Return1D'] = data_for_tabular[f'{t}_Close'].pct_change()
    data_for_tabular[f'{t}_MA20'] = data_for_tabular[f'{t}_Close'].rolling(20).mean()
    data_for_tabular[f'{t}_Volatility'] = data_for_tabular[f'{t}_Close'].pct_change().rolling(20).std()

data_for_tabular['Target_Dir'] = (data_for_tabular[TARGET_COL].shift(-1) > data_for_tabular[TARGET_COL]).astype(int)
data_for_tabular = data_for_tabular.dropna()

if "^GSPC_Volume" in data_for_tabular.columns:
    data_for_tabular = data_for_tabular.drop(columns="^GSPC_Volume")

# 学習・テストデータ分割
split_idx = int(len(data_for_tabular) * (1 - TEST_SIZE_RATIO))

# テーブルデータ系モデル用のデータ
X = data_for_tabular.drop(columns='Target_Dir')
y = data_for_tabular['Target_Dir']
X_tr, X_te = X.iloc[:split_idx], X.iloc[split_idx:]
y_tr, y_te = y.iloc[:split_idx], y.iloc[split_idx:]
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_te_scaled = scaler.transform(X_te)

# 時系列予測モデル用のデータ
ts_df = df['Close'][[TARGET_TICKER]].reset_index()
ts_df.columns = ['ds', 'y']
ts_train, ts_test = ts_df.iloc[:split_idx], ts_df.iloc[split_idx:]

print(f"データ準備完了。学習データ: {len(X_tr)} 件, テストデータ: {len(X_te)} 件")

# ───────────────────────────────────────────────
# 2. モデル評価の実行
# ───────────────────────────────────────────────
results_list = []

# --- 2-1. テーブルデータ系モデルの評価 ---
print("\n--- テーブルデータ系モデルの評価開始 ---")
pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
tabular_models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    "XGBoost": XGBClassifier(random_state=42, scale_pos_weight=pos_weight, n_estimators=200),
    "LightGBM": LGBMClassifier(random_state=42, is_unbalance=True, n_estimators=200),
    "CatBoost": CatBoostClassifier(random_state=42, scale_pos_weight=pos_weight, verbose=0, n_estimators=200)
}

for name, model in tabular_models.items():
    start_time = time.time()
    print(f"  Training {name}...")
    model.fit(X_tr_scaled, y_tr)
    pred_proba = model.predict_proba(X_te_scaled)[:, 1]
    pred_label = (pred_proba > 0.5).astype(int)
    acc = accuracy_score(y_te, pred_label)
    auc = roc_auc_score(y_te, pred_proba)
    duration = time.time() - start_time
    results_list.append({"Model": name, "Type": "Table-based", "Accuracy": acc, "AUC": auc, "Time (s)": duration})

# --- 2-2. 一般的な時系列予測モデルの評価 ---
print("\n--- 一般的な時系列予測モデルの評価開始 ---")
test_len = len(ts_test)
y_true_dir = (ts_test['y'].values > ts_test['y'].shift(1).values).astype(int)[1:]

# ホルトウィンターズ
start_time = time.time()
print("  Training Holt-Winters...")
model_hw = ExponentialSmoothing(ts_train['y'], trend='add', seasonal='add', seasonal_periods=252).fit()
preds_hw = model_hw.forecast(test_len)
pred_dir_hw = (preds_hw.values > ts_test['y'].shift(1).values).astype(int)[1:]
acc_hw = accuracy_score(y_true_dir, pred_dir_hw)
duration_hw = time.time() - start_time
results_list.append({"Model": "Holt-Winters", "Type": "Time Series", "Accuracy": acc_hw, "AUC": np.nan, "Time (s)": duration_hw})

# ARIMA
start_time = time.time()
print("  Training ARIMA...")
model_arima = ARIMA(ts_train['y'], order=(5, 1, 0)).fit()
preds_arima = model_arima.forecast(test_len)
pred_dir_arima = (preds_arima.values > ts_test['y'].shift(1).values).astype(int)[1:]
acc_arima = accuracy_score(y_true_dir, pred_dir_arima)
duration_arima = time.time() - start_time
results_list.append({"Model": "ARIMA", "Type": "Time Series", "Accuracy": acc_arima, "AUC": np.nan, "Time (s)": duration_arima})

# TBATS
start_time = time.time()
print("  Training TBATS...")
estimator = TBATS(seasonal_periods=(252,)) # 1年周期を仮定
model_tbats = estimator.fit(ts_train['y'])
preds_tbats = model_tbats.forecast(steps=test_len)
pred_dir_tbats = (preds_tbats > ts_test['y'].shift(1).values).astype(int)[1:]
acc_tbats = accuracy_score(y_true_dir, pred_dir_tbats)
duration_tbats = time.time() - start_time
results_list.append({"Model": "TBATS", "Type": "Time Series", "Accuracy": acc_tbats, "AUC": np.nan, "Time (s)": duration_tbats})

# Prophet
start_time = time.time()
print("  Training Prophet...")
model_prophet = Prophet()
model_prophet.fit(ts_train)
future = model_prophet.make_future_dataframe(periods=test_len)
forecast = model_prophet.predict(future)
preds_prophet = forecast['yhat'].iloc[-test_len:]
pred_dir_prophet = (preds_prophet.values > ts_test['y'].shift(1).values).astype(int)[1:]
acc_prophet = accuracy_score(y_true_dir, pred_dir_prophet)
duration_prophet = time.time() - start_time
results_list.append({"Model": "Prophet", "Type": "Time Series", "Accuracy": acc_prophet, "AUC": np.nan, "Time (s)": duration_prophet})

# ───────────────────────────────────────────────
# 3. 評価結果の表示
# ───────────────────────────────────────────────
print("\n3. 評価結果...")
results_df = pd.DataFrame(results_list).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
print("\n--- 全モデル性能比較 ---")
print(results_df.to_string())

best_model_info = results_df.iloc[0]
print("\n--- 結論 ---")
print(f"評価したモデルの中で、最も精度が高かったのは「{best_model_info['Model']}」であり、その正解率は {best_model_info['Accuracy']:.4f} でした。")