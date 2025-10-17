"""
order_imbalance_strategy.py

Single-file prototype pipeline implementing:
- DataLoader
- FeatureEngineer (MPB, VOI, OIR + rolling stats)
- ModelWrapper (LinearRegression, SVM, RandomForest)
- Trainer (CV, persistence)
- Backtester (simple execution model)
- Strategy (glue)

Author: kschoenberger1
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
import matplotlib.pyplot as plt

# ---------------------
# DataLoader
# ---------------------
class DataLoader:
    """
    Loads data from CSV or pandas DataFrame; expects aggregated bars at minimal:
      - timestamp
      - price_open, price_high, price_low, price_close
      - buy_volume, sell_volume, aggressive_buy_count, aggressive_sell_count (if available)
    If only trades/ticks are available, user should pre-aggregate externally into bars.
    """
    def __init__(self, source: Optional[str]=None, df: Optional[pd.DataFrame]=None, time_col: str="timestamp"):
        if source is None and df is None:
            raise ValueError("Either source (CSV path) or df (DataFrame) must be provided.")
        self.source = source
        self.df = df
        self.time_col = time_col

    def load(self) -> pd.DataFrame:
        if self.df is not None:
            df = self.df.copy()
        else:
            df = pd.read_csv(self.source)
        if self.time_col in df.columns:
            try:
                df[self.time_col] = pd.to_datetime(df[self.time_col])
                df = df.sort_values(self.time_col).reset_index(drop=True)
            except Exception:
                pass
        return df

# ---------------------
# FeatureEngineer
# ---------------------
class FeatureEngineer:
    """
    Computes MPB, VOI, OIR and rolling features.
    Usage:
      fe = FeatureEngineer(mpb_window=1, voi_window=10, oir_eps=1e-6)
      df_feat = fe.transform(df)
    """
    def __init__(self, mpb_window:int=1, voi_window:int=10, oi_ratio_window:int=10, oir_eps:float=1e-6):
        self.mpb_window = mpb_window
        self.voi_window = voi_window
        self.oi_ratio_window = oi_ratio_window
        self.oir_eps = oir_eps

    def _compute_mpb(self, df: pd.DataFrame) -> pd.Series:
        # MPB = (buy_volume - sell_volume) / total_volume
        bv = df.get('buy_volume')
        sv = df.get('sell_volume')
        if bv is None or sv is None:
            # fallback: use close price movement as proxy for direction * volume
            # Not ideal, but preserves pipeline for data that lacks buy/sell split.
            vol = df.get('volume', np.nan)
            direction = np.sign(df['price_close'].diff().fillna(0))
            signed_vol = direction * vol
            mpb = signed_vol / (vol.replace(0, np.nan))
            return mpb.fillna(0)
        total = (bv + sv).replace(0, np.nan)
        mpb = (bv - sv) / total
        return mpb.fillna(0)

    def _compute_voi(self, df: pd.DataFrame) -> pd.Series:
        # VOI: rolling sum of signed volume normalized by rolling sum of volume
        bv = df.get('buy_volume')
        sv = df.get('sell_volume')
        vol = df.get('volume')
        if (bv is not None) and (sv is not None):
            signed = bv - sv
            denom = (bv + sv)
        elif vol is not None:
            # fallback: sign from price change times volume
            signed = np.sign(df['price_close'].diff().fillna(0)) * vol
            denom = vol
        else:
            # no data -> zeros
            return pd.Series(0.0, index=df.index)
        rolling_signed = signed.rolling(self.voi_window, min_periods=1).sum()
        rolling_denom = denom.rolling(self.voi_window, min_periods=1).sum().replace(0, np.nan)
        voi = rolling_signed / rolling_denom
        return voi.fillna(0)

    def _compute_oir(self, df: pd.DataFrame) -> pd.Series:
        # OIR: ratio of aggressive buy count to aggressive sell count
        ab = df.get('aggressive_buy_count')
        asell = df.get('aggressive_sell_count')
        if ab is None or asell is None:
            # fallback: construct from buy/sell volumes using counts=1 per bar approximation
            ab = df.get('buy_volume', pd.Series(0, index=df.index))
            asell = df.get('sell_volume', pd.Series(0, index=df.index))
        rolling_ab = ab.rolling(self.oi_ratio_window, min_periods=1).sum()
        rolling_as = asell.rolling(self.oi_ratio_window, min_periods=1).sum()
        oir = (rolling_ab + self.oir_eps) / (rolling_as + self.oir_eps)
        # return log ratio
        return np.log(oir)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Ensure price_close exists
        if 'price_close' not in df.columns and 'close' in df.columns:
            df['price_close'] = df['close']
        df['MPB'] = self._compute_mpb(df)
        df['VOI'] = self._compute_voi(df)
        df['OIR_log'] = self._compute_oir(df)
        # rolling statistics
        df['MPB_roll_mean'] = df['MPB'].rolling(self.voi_window, min_periods=1).mean()
        df['MPB_roll_std'] = df['MPB'].rolling(self.voi_window, min_periods=1).std().fillna(0)
        df['VOI_roll_mean'] = df['VOI'].rolling(self.voi_window, min_periods=1).mean()
        df['OIR_roll_mean'] = df['OIR_log'].rolling(self.oi_ratio_window, min_periods=1).mean()
        # Price returns
        df['ret_1'] = df['price_close'].pct_change().fillna(0)
        df['ret_5'] = df['price_close'].pct_change(5).fillna(0)
        # create target: binary next-bar direction (1 price up, 0 price down or flat)
        df['target_up'] = (df['price_close'].shift(-1) > df['price_close']).astype(int)
        # drop rows with NaN in core features
        feature_cols = ['MPB','VOI','OIR_log','MPB_roll_mean','MPB_roll_std','VOI_roll_mean','OIR_roll_mean','ret_1','ret_5']
        df = df.dropna(subset=feature_cols)
        return df

# ---------------------
# ModelWrapper
# ---------------------
class ModelWrapper:
    """
    Wrap common models behind a unified interface.
    model_type: 'logreg' (LogisticRegression), 'svm', 'rf', 'linreg' (LinearRegression)
    """
    def __init__(self, model_type: str='rf', random_state: int=42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._init_model()

    def _init_model(self):
        if self.model_type == 'logreg':
            return LogisticRegression(max_iter=1000, random_state=self.random_state)
        elif self.model_type == 'svm':
            return SVC(probability=True, random_state=self.random_state)
        elif self.model_type == 'rf':
            return RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        elif self.model_type == 'linreg':
            return LinearRegression()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        else:
            raise RuntimeError("Model has no predict method")

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'decision_function'):
            # fallback: use decision function scaled
            df = self.model.decision_function(X)
            # convert to two-column proba-like
            probs = np.vstack([1 - (df - df.min())/(df.max()-df.min()+1e-9), (df - df.min())/(df.max()-df.min()+1e-9)]).T
            return probs
        else:
            return None

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)

# ---------------------
# Trainer
# ---------------------
class Trainer:
    def __init__(self, model_wrapper: ModelWrapper, feature_cols: list, test_size:float=0.2, random_state:int=42):
        self.model_wrapper = model_wrapper
        self.feature_cols = feature_cols
        self.test_size = test_size
        self.random_state = random_state
        self.trained = False

    def train(self, df: pd.DataFrame):
        X = df[self.feature_cols]
        y = df['target_up']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, shuffle=False
        )
        self.model_wrapper.fit(X_train, y_train)
        self.trained = True
        # Evaluate
        preds = self.model_wrapper.predict(X_test)
        proba = self.model_wrapper.predict_proba(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, preds),
            'precision': precision_score(y_test, preds, zero_division=0),
            'recall': recall_score(y_test, preds, zero_division=0),
            'f1': f1_score(y_test, preds, zero_division=0)
        }
        # try AUC
        try:
            if proba is not None:
                auc = roc_auc_score(y_test, proba[:,1])
            else:
                auc = None
        except Exception:
            auc = None
        metrics['auc'] = auc
        self.metrics = metrics
        self.X_test = X_test
        self.y_test = y_test
        self.preds = preds
        return metrics

    def cross_validate(self, df: pd.DataFrame, cv: int=5):
        X = df[self.feature_cols]
        y = df['target_up']
        scores = cross_val_score(self.model_wrapper.model, X, y, cv=cv, scoring='accuracy')
        return {'cv_accuracy_mean': scores.mean(), 'cv_accuracy_std': scores.std()}

# ---------------------
# Backtester (very simple)
# ---------------------
class Backtester:
    """
    Very simple backtester:
      - Signals: 1 -> go long next bar, 0 -> go flat/short depending on short_allowed
      - Basic PnL: next bar return * position_size
      - Slippage & fee per trade are supported
    """
    def __init__(self, price_col='price_close', slippage=0.0, fee=0.0, short_allowed=False):
        self.price_col = price_col
        self.slippage = slippage
        self.fee = fee
        self.short_allowed = short_allowed

    def run(self, df: pd.DataFrame, signals: np.ndarray) -> pd.DataFrame:
        df = df.copy().reset_index(drop=True)
        df['signal'] = 0
        df.loc[df.index[:len(signals)], 'signal'] = signals  # align
        # position: 1 for long, -1 for short (if allowed)
        pos = df['signal'].shift(0).fillna(0)
        if self.short_allowed:
            pos = pos.replace({0:0, 1:1})
        else:
            pos = pos.replace({0:0, 1:1})
        # Simple returns: next_bar_return = (next_price / price - 1)
        next_price = df[self.price_col].shift(-1)
        returns = (next_price - df[self.price_col]) / df[self.price_col]
        # PnL per step
        pnl = pos * returns - (pos.diff().abs().fillna(0) * self.fee) - (pos.abs() * self.slippage * 0)
        df['returns'] = returns
        df['pos'] = pos
        df['pnl'] = pnl
        df['cum_pnl'] = df['pnl'].cumsum()
        return df

# ---------------------
# Strategy (glue)
# ---------------------
class Strategy:
    def __init__(self, model_type='rf', fe_params=None, model_params=None, feature_cols=None):
        self.fe_params = fe_params or {}
        self.model_wrapper = ModelWrapper(model_type=model_type, **(model_params or {}))
        self.fe = FeatureEngineer(**self.fe_params)
        # default features
        self.feature_cols = feature_cols or ['MPB','VOI','OIR_log','MPB_roll_mean','MPB_roll_std','VOI_roll_mean','OIR_roll_mean','ret_1','ret_5']

    def run_pipeline(self, df: pd.DataFrame, persist_model_path: Optional[str] = None) -> Dict[str, Any]:
        df_feat = self.fe.transform(df)
        trainer = Trainer(self.model_wrapper, feature_cols=self.feature_cols)
        metrics = trainer.train(df_feat)
        if persist_model_path:
            self.model_wrapper.save(persist_model_path)
        # Backtest
        proba = self.model_wrapper.predict_proba(trainer.X_test)
        if proba is not None:
            signals = (proba[:,1] > 0.5).astype(int)
        else:
            signals = trainer.preds
        backtester = Backtester(price_col='price_close', slippage=0.0, fee=0.0, short_allowed=False)
        # Backtest only on test slice
        df_test = df_feat.loc[trainer.X_test.index]
        df_bt = backtester.run(df_test, signals)
        return {
            'metrics': metrics,
            'df_features': df_feat,
            'df_backtest': df_bt,
            'trainer': trainer
        }

# ---------------------
# Example usage & synthetic data generator
# ---------------------
def generate_synthetic_data(n=2000, seed=42):
    np.random.seed(seed)
    # Simulate price as random walk
    prices = 100 + np.cumsum(np.random.normal(0, 0.05, size=n))
    # Simulate volume and buy/sell splits
    volume = np.random.poisson(100, size=n) + 10
    # Construct buy/sell volumes with occasional imbalances
    imbalance = np.random.normal(0, 0.3, size=n)
    buy_vol = ((1 + imbalance) * volume / 2).clip(min=0).astype(float)
    sell_vol = (volume - buy_vol)
    # aggressive counts
    ab = (np.random.binomial(5, p=0.5 + 0.1 * imbalance.clip(-0.4,0.4))).astype(int)
    asell = (np.random.binomial(5, p=0.5 - 0.1 * imbalance.clip(-0.4,0.4))).astype(int)
    df = pd.DataFrame({
        'timestamp': pd.date_range("2025-01-01", periods=n, freq='T'),
        'price_close': prices,
        'volume': volume,
        'buy_volume': buy_vol,
        'sell_volume': sell_vol,
        'aggressive_buy_count': ab,
        'aggressive_sell_count': asell
    })
    return df

# ---------------------
# If run as script
# ---------------------
if __name__ == "__main__":
    # User can replace this with DataLoader("mydata.csv").load()
    df = generate_synthetic_data(n=3000)
    strat = Strategy(model_type='rf', fe_params={'voi_window': 20, 'oi_ratio_window': 20})
    results = strat.run_pipeline(df)
    print("Training metrics:", results['metrics'])
    # Report classification
    trainer = results['trainer']
    print("Classification report on test set:\n", classification_report(trainer.y_test, trainer.preds, zero_division=0))
    # Simple plot of cumulative PnL
    bt = results['df_backtest']
    plt.figure(figsize=(10,4))
    plt.plot(bt['cum_pnl'].values)
    plt.title("Cumulative PnL (test set)")
    plt.xlabel("Bar index (test slice)")
    plt.ylabel("Cumulative PnL")
    plt.grid(True)
    plt.show()

