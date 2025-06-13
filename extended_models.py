import warnings
warnings.filterwarnings('ignore')

# データ取得と前処理
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

# 時系列モデルライブラリ
try:
    from prophet import Prophet
    _prophet_available = True
except Exception as e:
    Prophet = None
    _prophet_available = False
    _prophet_error = e

try:
    from statsmodels.tsa.api import ExponentialSmoothing, ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _statsmodels_available = True
except Exception as e:
    ExponentialSmoothing = ARIMA = SARIMAX = None
    _statsmodels_available = False
    _statsmodels_error = e

try:
    from tbats import TBATS
    _tbats_available = True
except Exception as e:
    TBATS = None
    _tbats_available = False
    _tbats_error = e

# 深層時系列モデル（使用可能なら）
try:
    from neuralprophet import NeuralProphet
    _neuralprophet_available = True
except Exception as e:
    NeuralProphet = None
    _neuralprophet_available = False
    _neuralprophet_error = e

# 以下のモデル群は gluonts 等が必要なため、import に失敗したらスキップ
try:
    from gluonts.model.deepar import DeepAREstimator
    from gluonts.model.deep_factor import DeepFactorEstimator
    from gluonts.model.seq2seq import MQCNNEstimator
    from gluonts.mx.trainer import Trainer
    _gluonts_available = True
except Exception as e:
    DeepAREstimator = DeepFactorEstimator = MQCNNEstimator = Trainer = None
    _gluonts_available = False
    _gluonts_error = e

# 定数設定
TICKERS = ['AAPL', '^GSPC']
START_DATE = '2010-01-01'
END_DATE = '2023-12-31'
TARGET_TICKER = 'AAPL'
TARGET_COL = f"{TARGET_TICKER}_Close"
TEST_SIZE_RATIO = 0.3

print('1. データ取得と前処理 ...')
# 株価データ取得
df = yf.download(TICKERS, start=START_DATE, end=END_DATE, progress=False)

# テーブルデータ系特徴量作成
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
if '^GSPC_Volume' in data_for_tabular.columns:
    data_for_tabular = data_for_tabular.drop(columns='^GSPC_Volume')

split_idx = int(len(data_for_tabular) * (1 - TEST_SIZE_RATIO))
X = data_for_tabular.drop(columns='Target_Dir')
y = data_for_tabular['Target_Dir']
X_tr, X_te = X.iloc[:split_idx], X.iloc[split_idx:]
y_tr, y_te = y.iloc[:split_idx], y.iloc[split_idx:]
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_te_scaled = scaler.transform(X_te)

# 学習データをさらに学習用と評価用に分割
X_train_part, X_val, y_train_part, y_val = train_test_split(
    X_tr_scaled, y_tr, test_size=0.2, random_state=42
)

# 時系列予測用データ
ts_df = df['Close'][[TARGET_TICKER]].reset_index()
ts_df.columns = ['ds', 'y']
ts_train, ts_test = ts_df.iloc[:split_idx], ts_df.iloc[split_idx:]

y_true_dir = (ts_test['y'].values > ts_test['y'].shift(1).values).astype(int)[1:]

# 結果格納リスト
results_list = []

print('2. テーブルデータ系モデルのチューニングと評価 ...')
pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
cv = TimeSeriesSplit(n_splits=3)

# それぞれ簡単なハイパーパラメータ空間を定義
lr_params = {'C': [0.1, 1, 10]}
xgb_params = {'n_estimators': [100, 200], 'max_depth': [3, 5]}
lgbm_params = {'n_estimators': [100, 200], 'num_leaves': [31, 63]}
cat_params = {'depth': [4, 6], 'learning_rate': [0.03, 0.1]}

models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000, class_weight='balanced'), lr_params),
    ('XGBoost', XGBClassifier(random_state=42, scale_pos_weight=pos_weight), xgb_params),
    ('LightGBM', LGBMClassifier(random_state=42, is_unbalance=True), lgbm_params),
    ('CatBoost', CatBoostClassifier(random_state=42, scale_pos_weight=pos_weight, verbose=0), cat_params),
]

for name, base_model, params in models:
    search = RandomizedSearchCV(base_model, params, n_iter=2, cv=cv, scoring='accuracy', random_state=42)
    search.fit(X_train_part, y_train_part)
    best_model = search.best_estimator_

    # 評価用データを指定して途中経過を表示しながら学習
    if name == 'XGBoost':
        print("--- XGBoostの学習 ---")
        best_model.fit(
            X_train_part,
            y_train_part,
            eval_set=[(X_val, y_val)],
            verbose=100,
            early_stopping_rounds=20,
        )
    elif name == 'LightGBM':
        print("\n--- LightGBMの学習 ---")
        best_model.fit(
            X_train_part,
            y_train_part,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(period=100)],
        )
    elif name == 'CatBoost':
        print("\n--- CatBoostの学習 ---")
        best_model.fit(
            X_train_part,
            y_train_part,
            eval_set=[(X_val, y_val)],
            verbose=100,
            early_stopping_rounds=20,
        )
    else:
        best_model.fit(X_train_part, y_train_part)

    pred = best_model.predict(X_te_scaled)
    acc = accuracy_score(y_te, pred)
    try:
        proba = best_model.predict_proba(X_te_scaled)[:,1]
        auc = roc_auc_score(y_te, proba)
    except Exception:
        auc = np.nan
    results_list.append({'Model': name, 'Accuracy': acc, 'AUC': auc})

print('3. 一般的な時系列モデルの評価 ...')
if _statsmodels_available:
    # Holt-Winters
    hw_model = ExponentialSmoothing(ts_train['y'], trend='add', seasonal='add', seasonal_periods=252).fit()
    hw_pred = hw_model.forecast(len(ts_test))
    hw_dir = (hw_pred.values > ts_test['y'].shift(1).values).astype(int)[1:]
    acc_hw = accuracy_score(y_true_dir, hw_dir)
    results_list.append({'Model': 'Holt-Winters', 'Accuracy': acc_hw, 'AUC': np.nan})

    # ARIMA パラメータチューニング (簡易的に p と q を 0-2 で探索)
    best_aic = float('inf')
    best_order = None
    for p in range(3):
        for q in range(3):
            try:
                m = ARIMA(ts_train['y'], order=(p,1,q)).fit()
                if m.aic < best_aic:
                    best_aic = m.aic
                    best_order = (p,1,q)
            except Exception:
                continue
    if best_order:
        arima_model = ARIMA(ts_train['y'], order=best_order).fit()
        arima_pred = arima_model.forecast(len(ts_test))
        arima_dir = (arima_pred.values > ts_test['y'].shift(1).values).astype(int)[1:]
        acc_arima = accuracy_score(y_true_dir, arima_dir)
        results_list.append({'Model': f'ARIMA{best_order}', 'Accuracy': acc_arima, 'AUC': np.nan})

    # SARIMA/SARIMAX も同様にパラメータ簡易探索
    best_aic = float('inf')
    best_order = None
    for p in range(2):
        for q in range(2):
            try:
                m = SARIMAX(ts_train['y'], order=(p,1,q), seasonal_order=(p,0,q,252)).fit(disp=False)
                if m.aic < best_aic:
                    best_aic = m.aic
                    best_order = (p,1,q)
            except Exception:
                continue
    if best_order:
        sarimax_model = SARIMAX(ts_train['y'], order=best_order, seasonal_order=(best_order[0],0,best_order[2],252)).fit(disp=False)
        sarimax_pred = sarimax_model.forecast(len(ts_test))
        sarimax_dir = (sarimax_pred.values > ts_test['y'].shift(1).values).astype(int)[1:]
        acc_sarimax = accuracy_score(y_true_dir, sarimax_dir)
        results_list.append({'Model': f'SARIMAX{best_order}', 'Accuracy': acc_sarimax, 'AUC': np.nan})
else:
    print(f'statsmodels の読み込みに失敗: {_statsmodels_error}')

if _tbats_available:
    tbats_est = TBATS(seasonal_periods=(252,))
    tbats_model = tbats_est.fit(ts_train['y'])
    tbats_pred = tbats_model.forecast(steps=len(ts_test))
    tbats_dir = (tbats_pred > ts_test['y'].shift(1).values).astype(int)[1:]
    acc_tbats = accuracy_score(y_true_dir, tbats_dir)
    results_list.append({'Model': 'TBATS', 'Accuracy': acc_tbats, 'AUC': np.nan})
else:
    print(f'TBATS の読み込みに失敗: {_tbats_error}')

if _prophet_available:
    prophet_model = Prophet()
    prophet_model.fit(ts_train)
    future = prophet_model.make_future_dataframe(periods=len(ts_test))
    fcst = prophet_model.predict(future)
    prophet_pred = fcst['yhat'].iloc[-len(ts_test):]
    prophet_dir = (prophet_pred.values > ts_test['y'].shift(1).values).astype(int)[1:]
    acc_prophet = accuracy_score(y_true_dir, prophet_dir)
    results_list.append({'Model': 'Prophet', 'Accuracy': acc_prophet, 'AUC': np.nan})
else:
    print(f'Prophet の読み込みに失敗: {_prophet_error}')

if _neuralprophet_available:
    nprophet = NeuralProphet()
    nprophet.fit(ts_train, freq='D')
    forecast = nprophet.predict(nprophet.make_future_dataframe(ts_train, periods=len(ts_test)))
    nprophet_pred = forecast['yhat1'].iloc[-len(ts_test):]
    nprophet_dir = (nprophet_pred.values > ts_test['y'].shift(1).values).astype(int)[1:]
    acc_nprophet = accuracy_score(y_true_dir, nprophet_dir)
    results_list.append({'Model': 'NeuralProphet', 'Accuracy': acc_nprophet, 'AUC': np.nan})
else:
    print(f'NeuralProphet の読み込みに失敗: {_neuralprophet_error}')

if _gluonts_available:
    # GluonTS 系モデルは実装サンプルのみ。実行負荷が大きいので epochs=1 とする。
    from gluonts.dataset.common import ListDataset
    train_ds = ListDataset([{'start': ts_train['ds'].iloc[0], 'target': ts_train['y'].values}], freq='D')
    test_ds = ListDataset([{'start': ts_test['ds'].iloc[0], 'target': ts_test['y'].values}], freq='D')
    trainer = Trainer(epochs=1)
    deepar_est = DeepAREstimator(freq='D', prediction_length=len(ts_test), trainer=trainer)
    deepar_model = deepar_est.train(train_ds)
    for entry, forecast in zip(test_ds, deepar_model.predict(test_ds)):
        deepar_pred = forecast.mean
    deepar_dir = (deepar_pred > ts_test['y'].shift(1).values).astype(int)[1:]
    acc_deepar = accuracy_score(y_true_dir, deepar_dir)
    results_list.append({'Model': 'DeepAR', 'Accuracy': acc_deepar, 'AUC': np.nan})
else:
    print(f'GluonTS 系モデルの読み込みに失敗: {_gluonts_error}')

print('\n4. 評価結果')
res_df = pd.DataFrame(results_list)
print(res_df)

