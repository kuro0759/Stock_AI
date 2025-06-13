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

# 各種ライブラリのインポート
import yfinance as yf
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import csv
import os
import warnings
import logging

# SHAPのログを非表示にする
logging.getLogger("shap").setLevel(logging.ERROR)

# 環境依存の警告を無視
warnings.filterwarnings(
    "ignore",
    message=r"Could not find the number of physical cores.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"FEATURE_DEPENDENCE::independent.*",
)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

# ───────────────────────────────────────────────
# 0. 定数設定
# ───────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "GOOG", "^VIX", "^GSPC", "^IXIC", "^TNX"]
START_DATE = "2016-01-01"
END_DATE = "2022-12-01"
TARGET_TICKER = "AAPL"
TARGET_COL = "Target_Dir"
TEST_SIZE = 0.3
MAX_K = 15
KMEANS_N = 20  # 背景データ要約サンプル数
# 分析モード: "effect_vs_interaction" または "obs_vs_intervention"
#ANALYSIS_MODE = "effect_vs_interaction"
ANALYSIS_MODE = "obs_vs_intervention"
# SHAP解析のモードを設定

# ───────────────────────────────────────────────
# 1. データ取得・前処理
# ───────────────────────────────────────────────
print("1. データ取得・前処理を開始...")

# yfinance で株価データを取得
df = yf.download(
    TICKERS,
    start=START_DATE,
    end=END_DATE,
    interval="1d",
    auto_adjust=True,
    threads=True,
)

open_p = df["Open"].rename(columns=lambda c: f"{c}_Open")
high_p = df["High"].rename(columns=lambda c: f"{c}_High")
low_p = df["Low"].rename(columns=lambda c: f"{c}_Low")
close_p = df["Close"].rename(columns=lambda c: f"{c}_Close")
vol_p = df.get("Volume")
if vol_p is not None:
    vol_p = vol_p.rename(columns=lambda c: f"{c}_Volume")

frames = [open_p, high_p, low_p, close_p]
if vol_p is not None:
    frames.append(vol_p)

# 各価格データをまとめる
data = pd.concat(frames, axis=1).dropna()

# 各銘柄ごとのテクニカル指標を計算
for t in TICKERS:
    close_col = f"{t}_Close"
    high_col = f"{t}_High"
    low_col = f"{t}_Low"

    data[f"{t}_MA5"] = data[close_col].rolling(5).mean()
    data[f"{t}_MA20"] = data[close_col].rolling(20).mean()
    data[f"{t}_Volatility"] = data[close_col].pct_change().rolling(14).std()
    data[f"{t}_Return1D"] = data[close_col].pct_change()
    d = data[close_col].diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    mean_up = up.rolling(14).mean()
    mean_dn = dn.rolling(14).mean()
    rs = mean_up / mean_dn.replace(0, np.nan)
    data[f"{t}_RSI"] = 100 - 100 / (1 + rs)

    ema12 = data[close_col].ewm(span=12, adjust=False).mean()
    ema26 = data[close_col].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    data[f"{t}_MACD"] = macd
    data[f"{t}_MACD_signal"] = signal
    data[f"{t}_MACD_hist"] = macd - signal

    ma20 = data[close_col].rolling(20).mean()
    std20 = data[close_col].rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    data[f"{t}_BB_pct"] = (data[close_col] - lower) / (upper - lower)

    high14 = data[high_col].rolling(14).max()
    low14 = data[low_col].rolling(14).min()
    k = (data[close_col] - low14) / (high14 - low14)
    data[f"{t}_Stoch_K"] = k
    data[f"{t}_Stoch_D"] = k.rolling(3).mean()

    tr = pd.concat([
        data[high_col] - data[low_col],
        (data[high_col] - data[close_col].shift()).abs(),
        (data[low_col] - data[close_col].shift()).abs(),
    ], axis=1).max(axis=1)
    data[f"{t}_ATR"] = tr.rolling(14).mean()

    tenkan = (data[high_col].rolling(9).max() + data[low_col].rolling(9).min()) / 2
    kijun = (data[high_col].rolling(26).max() + data[low_col].rolling(26).min()) / 2
    data[f"{t}_Tenkan"] = tenkan
    data[f"{t}_Kijun"] = kijun

# 過去リターンを遅行特徴量として追加
for lag in [1, 2, 3, 5, 10]:
    data[f"{TARGET_TICKER}_Return_lag_{lag}"] = data[f"{TARGET_TICKER}_Return1D"].shift(lag)

WINDOW_SIZE = 20
data[f"{TARGET_TICKER}_Volatility_{WINDOW_SIZE}d"] = data[f"{TARGET_TICKER}_Close"].rolling(WINDOW_SIZE).std()
data[f"{TARGET_TICKER}_High_vs_{WINDOW_SIZE}d"] = data[f"{TARGET_TICKER}_Close"] / data[f"{TARGET_TICKER}_Close"].rolling(WINDOW_SIZE).max()
# 翌日の株価が上がるかどうかをターゲットに
target_ret = data[f"{TARGET_TICKER}_Close"].pct_change().shift(-1)
data[TARGET_COL] = (target_ret > 0).astype(int)

data = data.dropna()

feature_names = list(data.drop(columns=TARGET_COL).columns)
X = data[feature_names].values
y = data[TARGET_COL].values

split_idx = int(len(X) * (1 - TEST_SIZE))
X_tr, X_te = X[:split_idx], X[split_idx:]
y_tr, y_te = y[:split_idx], y[split_idx:]

# 特徴量を標準化
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)
print(f"学習データ: {len(X_tr)} 件, テストデータ: {len(X_te)} 件")

# クラス分布の確認と scale_pos_weight 設定
class_counts = pd.Series(y_tr).value_counts()
pos_weight = (
    class_counts.get(0, 0) / class_counts.get(1, 1)
    if class_counts.get(1, 0) != 0
    else 1.0
)
print(f"クラス分布: {class_counts.to_dict()}")
print(f"scale_pos_weight: {pos_weight:.2f}")

# ───────────────────────────────────────────────
# 2. モデル学習・予測
# ───────────────────────────────────────────────
print("\n2. モデル学習・予測を開始...")

# ハイパーパラメータ探索範囲
param_dist = {
    "n_estimators": [300, 400, 500, 600, 700],
    "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7, 9],
    "gamma": [0, 0.05, 0.1, 0.15, 0.2],
}
base_model = XGBClassifier(
    random_state=42,
    tree_method="hist",
    eval_metric="logloss",
    scale_pos_weight=pos_weight,
)
cv = TimeSeriesSplit(n_splits=5)
search = RandomizedSearchCV(
    base_model,
    param_distributions=param_dist,
    n_iter=20,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    random_state=42,
    verbose=0,
)
# ランダムサーチで最適パラメータを探索
search.fit(X_tr, y_tr)
best_params = search.best_params_
best_score = search.best_score_
print(f"Best params: {best_params}")
print(f"Cross-val AUC: {best_score:.3f}")

model = XGBClassifier(
    **best_params,
    random_state=42,
    tree_method="hist",
    eval_metric="logloss",
    early_stopping_rounds=10,
    scale_pos_weight=pos_weight,
)
model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
pred_proba = model.predict_proba(X_te)[:, 1]
pred_label = (pred_proba > 0.5).astype(int)
print("学習・予測が完了しました。")

# ───────────────────────────────────────────────
# 3. モデル性能評価
# ───────────────────────────────────────────────
print("\n3. モデル性能評価:")
acc = accuracy_score(y_te, pred_label)
auc = roc_auc_score(y_te, pred_proba)
precision = precision_score(y_te, pred_label)
recall = recall_score(y_te, pred_label)
f1 = f1_score(y_te, pred_label)
print(f"Accuracy : {acc:.3f}")
print(f"AUC      : {auc:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1       : {f1:.3f}")

# ───────────────────────────────────────────────
# 4. SHAP / I-SHAP 計算
# ───────────────────────────────────────────────
print("\n4. SHAP / I-SHAP の計算を開始...（時間がかかる場合があります）")

# 背景データをK-meansで要約
kmeans_obj = shap.kmeans(X_tr, KMEANS_N)
background = getattr(kmeans_obj, "data", np.array(kmeans_obj))
# SHAP/ISHAP のExplainerを生成

if ANALYSIS_MODE == "effect_vs_interaction":
    explainer_shap = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
    )
    explainer_ishap = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
    )
else:  # "obs_vs_intervention"
    explainer_shap = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="tree_path_dependent",
    )
    explainer_ishap = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
    )

import contextlib
# 計算時の警告を抑制
with open(os.devnull, "w") as fnull:
    with contextlib.redirect_stderr(fnull):
        shap_vals = explainer_shap.shap_values(X_te)
        if ANALYSIS_MODE == "effect_vs_interaction":
            ishap_vals = explainer_ishap.shap_interaction_values(X_te)
        else:
            ishap_vals = explainer_ishap.shap_values(X_te)

if isinstance(shap_vals, list):
    shap_vals = shap_vals[0]
if isinstance(ishap_vals, list):
    ishap_vals = ishap_vals[0]

mean_abs_shap = np.abs(shap_vals).mean(axis=0)
if ANALYSIS_MODE == "effect_vs_interaction":
    mean_abs_ishap = np.abs(ishap_vals).sum(axis=2).mean(axis=0)
else:
    mean_abs_ishap = np.abs(ishap_vals).mean(axis=0)

baseline = X_tr.mean(axis=0)
orig_preds = pred_proba
# 選択外の特徴量を無効化した際の差異を計算

def fidelity(mask_idx):
    if len(mask_idx) == 0:
        return 0.0
    Xc = X_te.copy()
    Xc[:, mask_idx] = baseline[mask_idx]
    return np.mean(np.abs(orig_preds - model.predict_proba(Xc)[:, 1]))

# Completeness と Fidelity の結果を保持
results = []
for K in range(MAX_K):
    top_sh = np.argsort(mean_abs_shap)[-K:] if K > 0 else []
    comp_s = mean_abs_shap[top_sh].sum() / mean_abs_shap.sum() if mean_abs_shap.sum() > 0 else 0
    bot_sh = [i for i in range(len(mean_abs_shap)) if i not in top_sh]
    fid_s = fidelity(bot_sh)

    top_ish = np.argsort(mean_abs_ishap)[-K:] if K > 0 else []
    comp_i = mean_abs_ishap[top_ish].sum() / mean_abs_ishap.sum() if mean_abs_ishap.sum() > 0 else 0
    bot_ish = [i for i in range(len(mean_abs_ishap)) if i not in top_ish]
    fid_i = fidelity(bot_ish)

    results.append({
        "K": K,
        "Comp_SHAP": comp_s,
        "Fid_SHAP": fid_s,
        "Comp_ISHAP": comp_i,
        "Fid_ISHAP": fid_i,
    })

df_res = pd.DataFrame(results)
print("計算が完了しました。")

# ───────────────────────────────────────────────
# 5. 結果表示 & グラフ保存
# ───────────────────────────────────────────────
print("\n5. グラフとCSVファイルを出力します...")

k_list = [5, 10, MAX_K - 1]
df_plot = df_res[df_res["K"].isin(k_list)]
# 結果をグラフ用に抽出
x = np.arange(len(k_list))

plt.figure(figsize=(8, 4))
bar_w = 0.35
plt.bar(x - bar_w / 2, df_plot["Fid_SHAP"], bar_w, label="Fidelity (SHAP)")
plt.bar(x + bar_w / 2, df_plot["Fid_ISHAP"], bar_w, label="Fidelity (I-SHAP)")
plt.xticks(x, k_list)
plt.xlabel("K (Top features count)")
plt.ylabel("Fidelity (MAE)")
plt.title("Fidelity Comparison: SHAP vs I-SHAP")
plt.legend()
plt.tight_layout()
plt.savefig("shap-ishap_fidelity_bar.png", dpi=300)
plt.close()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
ax1.plot(df_res["K"], df_res["Comp_SHAP"], "-o", label="Completeness (SHAP)")
ax1.plot(df_res["K"], df_res["Comp_ISHAP"], "--s", label="Completeness (I-SHAP)")
ax1.set_ylabel("Completeness")
ax1.legend(); ax1.grid(True)

ax2.plot(df_res["K"], df_res["Fid_SHAP"], "-o", label="Fidelity (SHAP)")
ax2.plot(df_res["K"], df_res["Fid_ISHAP"], "--s", label="Fidelity (I-SHAP)")
ax2.set_ylabel("Fidelity (MAE)")
ax2.set_xlabel("K (Top features count)")
ax2.legend(); ax2.grid(True)

plt.tight_layout()
plt.savefig("completeness_fidelity_vs_k.png", dpi=300)
plt.close()

pred_tr = model.predict_proba(X_tr)[:, 1]
plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(y)), y, color="gray", label="Truth")
plt.plot(np.arange(len(pred_tr)), pred_tr, color="green", label="Prediction (Train)")
plt.plot(np.arange(len(pred_tr), len(y)), pred_proba, color="red", label="Prediction (Test)")
plt.axvline(x=split_idx, color="blue", linestyle="--", label="Train/Test Split")
plt.title(f"Prediction vs Truth (Test AUC: {auc:.3f})")
plt.xlabel("Time index")
plt.ylabel("Probability")
plt.legend(); plt.tight_layout()
plt.savefig("prediction_vs_truth.png", dpi=300)
plt.close()

# 上位特徴量をCSVに保存
csv_path = os.path.join(os.path.dirname(__file__), "top_shap_ishap_elements.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["K", "SHAP_Top_Features", "I-SHAP_Top_Features"])
    for K in range(MAX_K):
        top_sh = np.argsort(mean_abs_shap)[-K:][::-1] if K > 0 else []
        top_ish = np.argsort(mean_abs_ishap)[-K:][::-1] if K > 0 else []
        sh_feats = "; ".join(feature_names[i] for i in top_sh)
        ish_feats = "; ".join(feature_names[i] for i in top_ish)
        writer.writerow([K, sh_feats, ish_feats])

print("すべての処理が完了しました。")
