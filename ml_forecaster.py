"""
ml_forecaster.py — Machine Learning Price Forecasting (Pillar 1)
=================================================================

Uses gradient boosted trees (XGBoost) and Random Forest to forecast
next-hour electricity prices from engineered tabular features.

WHAT THIS MODULE DOES
---------------------
Given a history of hourly electricity prices (e.g., the past 365 days),
it engineers a rich set of tabular features (lag prices, rolling statistics,
cyclical time encodings, peak-period indicators) and trains two gradient
boosting models to predict the *next* hour's price.

WHY TABULAR ML (NOT JUST DEEP LEARNING)?
-----------------------------------------
Electricity price follows strong, regular patterns: overnight lows, midday
solar dips, and evening peaks repeat with high consistency. These patterns
map naturally onto hand-crafted features that tree-based models exploit well.
On this type of structured, feature-rich tabular data, XGBoost and Random
Forest often match or outperform LSTM — which must learn the same patterns
from scratch. This module demonstrates that "the right tool depends on the
data, not the hype."

INPUT → OUTPUT
--------------
  Input:  A DataFrame of hourly electricity prices with metadata columns
          (datetime, price_mwh, hour, day_of_week, month, is_weekend).
  Output: Predicted price for the *next* hour (scalar), trained on the
          engineering features derived from the price history.

  Crucially, target = price[t+1] and features are all derived from
  price[t] and earlier — no lookahead bias.

STUDENT WORK
------------
Students improve upon this baseline in main.ipynb by writing their own
feature engineering function and/or tuning hyperparameters. The .py file
itself is provided infrastructure; do not modify it directly.

IEOR E4010: AI for Operations Research and Financial Engineering
Columbia University, Spring 2026
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

from config import Config, DEFAULT_CONFIG

# Try importing xgboost; fall back to sklearn GBR
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("xgboost not installed — falling back to sklearn GradientBoostingRegressor. "
                  "Install with: pip install xgboost")


# ============================================================
# Feature Engineering
# ============================================================
def engineer_features(
    df: pd.DataFrame,
    cfg: Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Engineer features for price forecasting from raw price data.

    FEATURE CATEGORIES
    ------------------
    1. Cyclical time encodings (sin/cos of hour, day-of-week, month):
       Rather than treating hour=23 as "far from" hour=0, sine/cosine
       transforms put adjacent hours close together in feature space.

    2. Lag features (price at t-1, t-2, t-3, t-6, t-12, t-24, t-168):
       Yesterday's same-hour price and last-week's same-hour price are
       typically the strongest predictors of tomorrow's price.

    3. Rolling statistics (mean/std over 6h, 24h, 168h windows):
       Capture recent price level and volatility.

    4. Price change features (1h and 24h differences):
       Momentum indicators — is the price currently rising or falling?

    5. Peak period indicators (is_peak, is_solar, is_overnight):
       Binary flags for electricity market structural periods.

    TARGET: price[t+1] — the next hour's price.
    All features are computed from price[t] and earlier, so there is
    no lookahead bias. Rows with NaN (due to lags at the start of the
    dataset) are dropped.

    Args:
        df:  DataFrame with columns: price_mwh, hour, day_of_week, month, is_weekend
        cfg: Configuration

    Returns:
        DataFrame with engineered features and target column
    """
    feat = df.copy()

    # ---- Time-based features ----
    feat["hour_sin"] = np.sin(2 * np.pi * feat["hour"] / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * feat["hour"] / 24)
    feat["dow_sin"] = np.sin(2 * np.pi * feat["day_of_week"] / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * feat["day_of_week"] / 7)
    feat["month_sin"] = np.sin(2 * np.pi * feat["month"] / 12)
    feat["month_cos"] = np.cos(2 * np.pi * feat["month"] / 12)
    feat["is_weekend"] = feat["is_weekend"].astype(int)

    # ---- Lag features ----
    # Previous hours' prices (most important features typically)
    for lag in [1, 2, 3, 6, 12, 24]:
        feat[f"price_lag_{lag}"] = feat["price_mwh"].shift(lag)

    # Same hour yesterday and last week
    feat["price_same_hour_yesterday"] = feat["price_mwh"].shift(24)
    feat["price_same_hour_last_week"] = feat["price_mwh"].shift(168)

    # ---- Rolling statistics ----
    for window in [6, 24, 168]:
        feat[f"price_rolling_mean_{window}"] = (
            feat["price_mwh"].rolling(window=window, min_periods=1).mean()
        )
        feat[f"price_rolling_std_{window}"] = (
            feat["price_mwh"].rolling(window=window, min_periods=1).std().fillna(0)
        )

    # ---- Price change features ----
    feat["price_diff_1"] = feat["price_mwh"].diff(1)
    feat["price_diff_24"] = feat["price_mwh"].diff(24)

    # ---- Peak period indicator ----
    feat["is_peak"] = ((feat["hour"] >= 17) & (feat["hour"] < 21)).astype(int)
    feat["is_solar"] = ((feat["hour"] >= 10) & (feat["hour"] < 15)).astype(int)
    feat["is_overnight"] = ((feat["hour"] >= 0) & (feat["hour"] < 6)).astype(int)

    # ---- Target: next-hour price ----
    feat["target"] = feat["price_mwh"].shift(-1)

    # Drop rows with NaN from lags/shifts
    feat.dropna(inplace=True)
    feat.reset_index(drop=True, inplace=True)

    return feat


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the list of feature columns (everything except metadata and target)."""
    exclude = {"datetime", "price_mwh", "target", "day_index", "date"}
    return [c for c in df.columns if c not in exclude]


# ============================================================
# Model Training
# ============================================================
def train_ml_models(
    train_df: pd.DataFrame,
    cfg: Config = DEFAULT_CONFIG,
) -> Dict[str, Any]:
    """Train XGBoost and Random Forest models.

    Trains two models on the same feature set:
    - XGBoost (or sklearn GradientBoosting as fallback): a sequential
      ensemble of shallow trees where each tree corrects the residual
      errors of the previous. The learning_rate and n_estimators jointly
      control the bias-variance trade-off.
    - Random Forest: a parallel ensemble of deep trees trained on random
      feature subsets. Typically lower variance but higher bias than GBM.

    Hyperparameter defaults are set conservatively to work well out-of-the-box.
    Students can improve results by tuning via grid or random search.

    Args:
        train_df: DataFrame with engineered features and target
        cfg:      Configuration

    Returns:
        Dict with trained models and feature info
    """
    feature_cols = get_feature_columns(train_df)
    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values

    models = {}

    # --- XGBoost / Gradient Boosting ---
    if HAS_XGBOOST:
        print("Training XGBoost regressor...")
        xgb_model = XGBRegressor(
            n_estimators=cfg.forecast.xgb_n_estimators,
            max_depth=cfg.forecast.xgb_max_depth,
            learning_rate=cfg.forecast.xgb_learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=cfg.seed,
            verbosity=0,
        )
        xgb_model.fit(X_train, y_train)
        models["xgboost"] = xgb_model
    else:
        print("Training GradientBoosting regressor (sklearn fallback)...")
        gbr_model = GradientBoostingRegressor(
            n_estimators=cfg.forecast.xgb_n_estimators,
            max_depth=cfg.forecast.xgb_max_depth,
            learning_rate=cfg.forecast.xgb_learning_rate,
            subsample=0.8,
            random_state=cfg.seed,
        )
        gbr_model.fit(X_train, y_train)
        models["xgboost"] = gbr_model

    # --- Random Forest ---
    print("Training Random Forest regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=cfg.seed,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    models["random_forest"] = rf_model

    return {
        "models": models,
        "feature_columns": feature_cols,
        "train_size": len(X_train),
    }


# ============================================================
# Evaluation
# ============================================================
def evaluate_ml_models(
    models_dict: Dict[str, Any],
    test_df: pd.DataFrame,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Evaluate trained models on test data.

    Args:
        models_dict: Output from train_ml_models()
        test_df:     DataFrame with engineered features and target
        verbose:     Print metrics

    Returns:
        Dict mapping model_name → {mae, rmse, r2, predictions}
    """
    feature_cols = models_dict["feature_columns"]
    X_test = test_df[feature_cols].values
    y_test = test_df["target"].values

    if len(X_test) == 0:
        if verbose:
            print("WARNING: Test set is empty after feature engineering. "
                  "Use more historical days (cfg.price.num_days_history).")
        return {}

    results = {}

    for name, model in models_dict["models"].items():
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "predictions": y_pred,
            "actuals": y_test,
        }

        if verbose:
            print(f"\n{name.upper()} Results:")
            print(f"  MAE:  ${mae:.2f}/MWh")
            print(f"  RMSE: ${rmse:.2f}/MWh")
            print(f"  R²:   {r2:.4f}")

    return results


def get_feature_importance(
    models_dict: Dict[str, Any],
    top_n: int = 15,
) -> pd.DataFrame:
    """Extract and rank feature importances.

    Feature importance in tree models measures how much each feature
    reduces prediction error across all splits in all trees. High
    importance on lag_1 means "last hour's price is the best predictor
    of next hour's price," which aligns with electricity market intuition.

    When analyzing results, consider:
    - Are short-lag features (lag_1, lag_2) dominant? (autocorrelation)
    - Do same-hour-yesterday / last-week features rank highly? (seasonality)
    - Do rolling stats add information beyond the raw lags?
    - How much do cyclical time features (hour_sin/cos) contribute vs.
      peak-period indicator flags?

    Args:
        models_dict: Output from train_ml_models()
        top_n:       Number of top features to return

    Returns:
        DataFrame with feature names and importance scores
    """
    feature_cols = models_dict["feature_columns"]
    importances = {}

    for name, model in models_dict["models"].items():
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            importances[name] = imp

    if not importances:
        return pd.DataFrame()

    # Use first available model's importances
    first_model = list(importances.keys())[0]
    imp = importances[first_model]

    df = pd.DataFrame({
        "feature": feature_cols,
        "importance": imp,
    }).sort_values("importance", ascending=False).head(top_n)

    return df


# ============================================================
# Prediction for Environment Integration
# ============================================================
def predict_next_24h(
    model,
    recent_prices: np.ndarray,
    current_hour: int,
    current_dow: int,
    current_month: int,
    cfg: Config = DEFAULT_CONFIG,
) -> np.ndarray:
    """Forecast the next 24 hours of prices using the trained model.

    This is the function the agentic AI and RL agent can call.

    Args:
        model:         Trained sklearn/xgboost model
        recent_prices: Last 168+ hours of prices (for lag/rolling features)
        current_hour:  Current hour of day (0-23)
        current_dow:   Current day of week (0=Mon)
        current_month: Current month (1-12)
        cfg:           Configuration

    Returns:
        np.ndarray of shape (24,) with predicted prices $/MWh
    """
    predictions = []

    # Build a mini-DataFrame for each future hour
    prices_buffer = list(recent_prices)

    for h_ahead in range(24):
        hour = (current_hour + h_ahead + 1) % 24
        dow = current_dow
        if hour < current_hour:
            dow = (dow + 1) % 7
        month = current_month
        is_weekend = 1 if dow >= 5 else 0

        # Build feature dict
        feats = {}
        feats["hour"] = hour
        feats["day_of_week"] = dow
        feats["month"] = month
        feats["is_weekend"] = is_weekend
        feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        feats["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        feats["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        feats["month_sin"] = np.sin(2 * np.pi * month / 12)
        feats["month_cos"] = np.cos(2 * np.pi * month / 12)

        # Lag features from buffer
        n = len(prices_buffer)
        for lag in [1, 2, 3, 6, 12, 24]:
            idx = n - lag
            feats[f"price_lag_{lag}"] = prices_buffer[idx] if idx >= 0 else prices_buffer[0]

        feats["price_same_hour_yesterday"] = (
            prices_buffer[n - 24] if n >= 24 else prices_buffer[0]
        )
        feats["price_same_hour_last_week"] = (
            prices_buffer[n - 168] if n >= 168 else prices_buffer[0]
        )

        # Rolling stats
        for window in [6, 24, 168]:
            window_data = prices_buffer[-min(window, n):]
            feats[f"price_rolling_mean_{window}"] = np.mean(window_data)
            feats[f"price_rolling_std_{window}"] = np.std(window_data) if len(window_data) > 1 else 0

        # Price diffs
        feats["price_diff_1"] = prices_buffer[-1] - prices_buffer[-2] if n >= 2 else 0
        feats["price_diff_24"] = prices_buffer[-1] - prices_buffer[-24] if n >= 24 else 0

        # Peak indicators
        feats["is_peak"] = 1 if 17 <= hour < 21 else 0
        feats["is_solar"] = 1 if 10 <= hour < 15 else 0
        feats["is_overnight"] = 1 if 0 <= hour < 6 else 0

        # Predict
        X = np.array([[feats[c] for c in sorted(feats.keys())]])
        # We need features in the same order as training — use model
        try:
            if hasattr(model, "feature_names_in_"):
                X = np.array([[feats[c] for c in model.feature_names_in_]])
            pred = float(model.predict(X)[0])
        except Exception:
            # Fallback: use last known price
            pred = float(prices_buffer[-1])

        pred = max(0, pred)
        predictions.append(pred)
        prices_buffer.append(pred)

    return np.array(predictions)


# ============================================================
# Model Persistence (Save / Load)
# ============================================================
def save_ml_model(models_dict: Dict[str, Any], path: str = "submission/ml_model.pkl") -> None:
    """Save trained ML models and feature metadata to disk.

    Saves the complete models_dict (models + feature_columns + train_size)
    using joblib. The saved file can be loaded with load_ml_model() and
    evaluated on any dataset that has the same feature columns.

    Args:
        models_dict: Output from train_ml_models()
        path:        Save path (e.g., "submission/ml_model.pkl")
    """
    import joblib
    import os

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    joblib.dump(models_dict, path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"ML model saved to '{path}' ({size_mb:.1f} MB)")
    print(f"  Models: {list(models_dict['models'].keys())}")
    print(f"  Features: {len(models_dict['feature_columns'])}")
    print(f"  Trained on: {models_dict['train_size']:,} samples")


def load_ml_model(path: str = "submission/ml_model.pkl") -> Dict[str, Any]:
    """Load a saved ML model from disk.

    Args:
        path: Path to the saved .pkl file

    Returns:
        models_dict compatible with evaluate_ml_models() and predict_next_24h()
    """
    import joblib

    models_dict = joblib.load(path)
    print(f"ML model loaded from '{path}'")
    print(f"  Models: {list(models_dict['models'].keys())}")
    print(f"  Features: {len(models_dict['feature_columns'])}")
    return models_dict


# ============================================================
# Visualization
# ============================================================
def plot_ml_results(
    results: Dict[str, Dict[str, float]],
    test_df: pd.DataFrame,
    num_days: int = 7,
):
    """Plot ML forecasting results.

    Args:
        results:  Output from evaluate_ml_models()
        test_df:  Test DataFrame with actuals
        num_days: Number of days to plot
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("ML Price Forecasting Results", fontsize=14)

    hours_to_plot = num_days * 24

    # Panel 1: Actual vs Predicted (time series)
    ax1 = axes[0, 0]
    actuals = list(results.values())[0]["actuals"][:hours_to_plot]
    ax1.plot(actuals, "k-", linewidth=0.8, alpha=0.7, label="Actual")
    for name, res in results.items():
        ax1.plot(res["predictions"][:hours_to_plot], linewidth=0.8,
                alpha=0.8, label=f"{name} (MAE=${res['mae']:.1f})")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Price ($/MWh)")
    ax1.set_title(f"Actual vs. Predicted ({num_days} days)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Scatter plot
    ax2 = axes[0, 1]
    for name, res in results.items():
        ax2.scatter(res["actuals"][:500], res["predictions"][:500],
                   s=5, alpha=0.3, label=name)
    lim = max(np.max(actuals), 100)
    ax2.plot([0, lim], [0, lim], "r--", linewidth=1, label="Perfect")
    ax2.set_xlabel("Actual ($/MWh)")
    ax2.set_ylabel("Predicted ($/MWh)")
    ax2.set_title("Prediction Scatter Plot")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Error distribution
    ax3 = axes[1, 0]
    for name, res in results.items():
        errors = res["predictions"] - res["actuals"]
        ax3.hist(errors, bins=50, alpha=0.5, label=f"{name} (σ={np.std(errors):.1f})")
    ax3.set_xlabel("Prediction Error ($/MWh)")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Error Distribution")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Feature importance
    ax4 = axes[1, 1]
    # We'll plot this from outside if models_dict is available
    ax4.text(0.5, 0.5, "See get_feature_importance()",
            ha="center", va="center", transform=ax4.transAxes, fontsize=12)
    ax4.set_title("Feature Importance (call separately)")

    plt.tight_layout()
    plt.savefig("ml_forecasting_results.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_importance(importance_df: pd.DataFrame):
    """Plot feature importance bar chart."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        importance_df["feature"].values[::-1],
        importance_df["importance"].values[::-1],
        color="#0D9488",
    )
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importances")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    from data_utils import generate_synthetic_prices, train_test_split_prices

    cfg = Config()

    # Generate data
    print("Generating synthetic price data...")
    prices_df = generate_synthetic_prices(cfg)

    # Split
    train_df, test_df = train_test_split_prices(prices_df, cfg)

    # Engineer features
    print("\nEngineering features...")
    train_feat = engineer_features(train_df, cfg)
    test_feat = engineer_features(test_df, cfg)
    print(f"Features: {len(get_feature_columns(train_feat))}")
    print(f"Train samples: {len(train_feat)}, Test samples: {len(test_feat)}")

    # Train
    print("\nTraining models...")
    models_dict = train_ml_models(train_feat, cfg)

    # Evaluate
    print("\nEvaluating on test set...")
    results = evaluate_ml_models(models_dict, test_feat)

    # Feature importance
    print("\nFeature importance:")
    imp_df = get_feature_importance(models_dict)
    print(imp_df.to_string(index=False))

    # Plot
    try:
        plot_ml_results(results, test_feat)
        plot_feature_importance(imp_df)
    except ImportError:
        print("matplotlib not available — skipping plots")
