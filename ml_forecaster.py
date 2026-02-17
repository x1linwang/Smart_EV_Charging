"""
ml_forecaster.py — Machine Learning Price Forecasting (Pillar 1)
=================================================================

Uses gradient boosted trees (XGBoost) and Random Forest to forecast
next-hour electricity prices from engineered tabular features.

This module demonstrates that for structured, tabular data with known
temporal patterns, classical ML often matches or beats deep learning —
a key lesson for IEOR students.

Student TODOs are marked with ★. The provided implementations are
complete and runnable; students can improve them.

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

    ★ STUDENT TODO: Improve or extend this feature engineering pipeline.
    
    Ideas to try:
    - Add more lag features (t-2, t-3, t-12, t-48)
    - Add rolling statistics with different windows
    - Add price change features (returns, momentum)
    - Add interaction features (hour × is_weekend)
    - Add Fourier features for cyclical encoding of time
    - Add exponentially weighted moving averages
    - Try polynomial features of existing ones

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

    ★ STUDENT TODO: Tune hyperparameters to improve performance.
    
    Ideas to try:
    - Grid search or random search over hyperparameters
    - Adjust n_estimators, max_depth, learning_rate
    - Try different min_samples_leaf / min_child_weight
    - Experiment with subsample ratios
    - Compare with other models (LightGBM, CatBoost, Ridge, etc.)

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

    ★ STUDENT TODO: Analyze which features matter most.
    
    Questions to answer:
    - Which lag feature is most important?
    - Do cyclical time features beat raw hour/month?
    - How important are rolling statistics?

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
