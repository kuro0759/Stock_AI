"""
──────────────────────────────────────────────────────────
XGBoost × SHAP / I-SHAP の比較に
  ▸ 追加評価指標
      – MAE   (Mean Absolute Error)
      – RMSE  (Root Mean Squared Error)
      – MAPE  (Mean Absolute Percentage Error)
      – R²    (決定係数)
      – DirAcc(方向一致率)
  ▸ Completeness / Fidelity も従来通り
  ▸ K = 0~14 で
      ・Fidelity の棒グラフを出力 (shap-ishap_fidelity_bar.png)
      ・Completeness / Fidelity のラインプロットを出力 (completeness_fidelity_vs_k.png)
  ▸ 予測値 vs 真値の推移を出力 (prediction_vs_truth.png)
  ▸ 上位要素をCSVに出力 (top_shap_ishap_elements.csv)
──────────────────────────────────────────────────────────
【実行前にインストール】
pip install yfinance xgboost shap numpy pandas scikit-learn matplotlib
"""

import yfinance as yf
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import csv
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ───────────────────────────────────────────────
# 0. 定数設定
# ───────────────────────────────────────────────
TICKERS       = ["AAPL", "MSFT", "GOOG"]
START_DATE    = "2000-01-01"
END_DATE      = "2025-05-31"
TARGET_TICKER = "AAPL"
TARGET_COL    = f"{TARGET_TICKER}_Return"
TEST_SIZE     = 0.2
MAX_K         = 15
KMEANS_N      = 100  # 背景データ要約サンプル数

# ───────────────────────────────────────────────
# 1. データ取得・前処理
# ───────────────────────────────────────────────
print("1. データ取得・前処理を開始...")
df = yf.download(
    TICKERS,
    start=START_DATE,
    end=END_DATE,
    interval="1d",
    auto_adjust=True,
    threads=True
)

price = df["Close"].rename(columns=lambda c: f"{c}_Close")
vol   = df["Volume"].rename(columns=lambda c: f"{c}_Volume")
data  = pd.concat([price, vol], axis=1).dropna()

for t in TICKERS:
    data[f"{t}_MA5"] = data[f"{t}_Close"].rolling(5).mean()
    d = data[f"{t}_Close"].diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    mean_up = up.rolling(14).mean()
    mean_dn = dn.rolling(14).mean()
    rs = mean_up / mean_dn.replace(0, np.nan)
    data[f"{t}_RSI"] = 100 - 100/(1 + rs)

data[TARGET_COL] = data[f"{TARGET_TICKER}_Close"].pct_change().shift(-1)
data = data.dropna()

feature_names = list(data.drop(columns=TARGET_COL).columns)
X = data[feature_names].values
y = data[TARGET_COL].values

split_idx = int(len(X) * (1 - TEST_SIZE))
X_tr, X_te = X[:split_idx], X[split_idx:]
y_tr, y_te = y[:split_idx], y[split_idx:]
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)
print(f"学習データ: {len(X_tr)} 件, テストデータ: {len(X_te)} 件")

# ───────────────────────────────────────────────
# 2. モデル学習・予測
# ───────────────────────────────────────────────
print("\n2. モデル学習・予測を開始...")

# ハイパーパラメータチューニング
param_dist = {
    "n_estimators": [200, 300, 400],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}
base_model = XGBRegressor(random_state=42, tree_method="hist")
cv = TimeSeriesSplit(n_splits=3)
search = RandomizedSearchCV(
    base_model,
    param_distributions=param_dist,
    n_iter=10,
    scoring="neg_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    verbose=0,
    random_state=42,
)
search.fit(X_tr, y_tr)
best_params = search.best_params_
print(f"Best params: {best_params}")

model = XGBRegressor(
    **best_params,
    random_state=42,
    tree_method="hist",
    early_stopping_rounds=10,
)
model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
pred = model.predict(X_te)
print("学習・予測が完了しました。")

# ───────────────────────────────────────────────
# 3. モデル性能評価
# ───────────────────────────────────────────────
print("\n3. モデル性能評価:")
mae = mean_absolute_error(y_te, pred)
mse = mean_squared_error(y_te, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_te, pred)
dir_acc = ((pred * y_te) > 0).mean()
mask = (y_te != 0)
mape = np.mean(np.abs((y_te[mask] - pred[mask]) / y_te[mask])) * 100 if mask.sum()>0 else np.nan

print(f"MAE     : {mae:.5f}")
print(f"RMSE    : {rmse:.5f}")
print(f"MAPE    : {mape:.3f}%")
print(f"R²      : {r2:.3f}")
print(f"DirAcc  : {dir_acc:.3f}")

# ───────────────────────────────────────────────
# 4. SHAP / I-SHAP 計算
# ───────────────────────────────────────────────
print("\n4. SHAP / I-SHAP の計算を開始...（時間がかかる場合があります）")

# ─ kmeans で要約した背景データ（numpy array）を取得 ─
kmeans_obj = shap.kmeans(X_tr, KMEANS_N)
# shap 0.42 では kmeans_obj.data, 古いバージョンではそのまま ndarray の場合があるため
background = kmeans_obj.data if hasattr(kmeans_obj, "data") else np.array(kmeans_obj)

# 従来SHAP／I-SHAP ともに interventional モード
explainer_shap = shap.TreeExplainer(
    model,
    data=background,
    feature_perturbation="interventional"
)
shap_vals  = explainer_shap.shap_values(X_te)

explainer_ishap = shap.TreeExplainer(
    model,
    data=background,
    feature_perturbation="interventional"
)
ishap_vals = explainer_ishap.shap_interaction_values(X_te)

mean_abs_shap  = np.abs(shap_vals).mean(axis=0)
mean_abs_ishap = np.abs(ishap_vals).mean(axis=0)

baseline   = X_tr.mean(axis=0)
orig_preds = model.predict(X_te)

def fidelity(mask_idx):
    if len(mask_idx)==0:
        return 0.0
    Xc = X_te.copy()
    Xc[:, mask_idx] = baseline[mask_idx]
    return mean_absolute_error(orig_preds, model.predict(Xc))

# Completeness / Fidelity を K=0~MAX_K-1 で算出
results = []
for K in range(MAX_K):
    top_sh = np.argsort(mean_abs_shap)[-K:] if K>0 else []
    comp_s = mean_abs_shap[top_sh].sum() / mean_abs_shap.sum() if mean_abs_shap.sum()>0 else 0
    bot_sh = [i for i in range(len(mean_abs_shap)) if i not in top_sh]
    fid_s = fidelity(bot_sh)

    top_ish = np.argsort(mean_abs_ishap)[-K:] if K>0 else []
    comp_i  = mean_abs_ishap[top_ish].sum() / mean_abs_ishap.sum() if mean_abs_ishap.sum()>0 else 0
    bot_ish = [i for i in range(len(mean_abs_ishap)) if i not in top_ish]
    fid_i = fidelity(bot_ish)

    results.append({
        "K": K,
        "Comp_SHAP": comp_s, "Fid_SHAP": fid_s,
        "Comp_ISHAP": comp_i, "Fid_ISHAP": fid_i
    })

df_res = pd.DataFrame(results)
print("計算が完了しました。")

# ───────────────────────────────────────────────
# 5. 結果表示 & グラフ保存
# ───────────────────────────────────────────────
print("\n5. グラフとCSVファイルを出力します...")

# 5-1. 棒グラフ (Fidelity)
k_list = [5, 10, MAX_K-1]
df_plot = df_res[df_res['K'].isin(k_list)]
x = np.arange(len(k_list))

plt.figure(figsize=(8,4))
bar_w = 0.35
plt.bar(x - bar_w/2, df_plot['Fid_SHAP'], bar_w, label="Fidelity (SHAP)")
plt.bar(x + bar_w/2, df_plot['Fid_ISHAP'], bar_w, label="Fidelity (I-SHAP)")
plt.xticks(x, k_list)
plt.xlabel("K (Top features count)")
plt.ylabel("Fidelity (MAE)")
plt.title("Fidelity Comparison: SHAP vs I-SHAP")
plt.legend()
plt.tight_layout()
plt.savefig("shap-ishap_fidelity_bar.png", dpi=300)
plt.close()

# 5-2. ラインプロット (Completeness & Fidelity vs K)
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,8), sharex=True)
ax1.plot(df_res['K'], df_res['Comp_SHAP'], '-o', label="Completeness (SHAP)")
ax1.plot(df_res['K'], df_res['Comp_ISHAP'], '--s', label="Completeness (I-SHAP)")
ax1.set_ylabel("Completeness")
ax1.legend(); ax1.grid(True)

ax2.plot(df_res['K'], df_res['Fid_SHAP'], '-o', label="Fidelity (SHAP)")
ax2.plot(df_res['K'], df_res['Fid_ISHAP'], '--s', label="Fidelity (I-SHAP)")
ax2.set_ylabel("Fidelity (MAE)")
ax2.set_xlabel("K (Top features count)")
ax2.legend(); ax2.grid(True)

plt.tight_layout()
plt.savefig("completeness_fidelity_vs_k.png", dpi=300)
plt.close()

# 5-3. 予測推移プロット
pred_tr = model.predict(X_tr)
plt.figure(figsize=(10,5))
plt.plot(np.arange(len(y)), y, color="gray", label="Truth")
plt.plot(np.arange(len(pred_tr)), pred_tr, color="green", label="Prediction (Train)")
plt.plot(np.arange(len(pred_tr), len(y)), pred, color="red", label="Prediction (Test)")
plt.axvline(x=split_idx, color="blue", linestyle="--", label="Train/Test Split")
plt.title(f"Prediction vs Truth (Test R²: {r2:.3f})")
plt.xlabel("Time index")
plt.ylabel("Return")
plt.legend(); plt.tight_layout()
plt.savefig("prediction_vs_truth.png", dpi=300)
plt.close()

# 5-4. 上位要素を CSV に書き出し
csv_path = os.path.join(os.path.dirname(__file__), 'top_shap_ishap_elements.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['K', 'SHAP_Top_Features', 'I-SHAP_Top_Features'])
    for K in range(MAX_K):
        top_sh = np.argsort(mean_abs_shap)[-K:][::-1] if K>0 else []
        top_ish = np.argsort(mean_abs_ishap)[-K:][::-1] if K>0 else []
        sh_feats  = "; ".join(feature_names[i] for i in top_sh)
        ish_feats = "; ".join(feature_names[i] for i in top_ish)
        writer.writerow([K, sh_feats, ish_feats])

print("すべての処理が完了しました。")
