#!/usr/bin/env python3
"""
autograder.py — CA Benchmark Autograder for EV Smart Charging Project
======================================================================

OVERVIEW
--------
This script evaluates all student-submitted model files against a hidden
holdout dataset and produces a ranked leaderboard. It is intended for
course staff (CAs/instructors) only; students do not run this file.

USAGE
-----
  python autograder.py --submissions_dir ./submissions [OPTIONS]

  Required:
    --submissions_dir DIR   Directory containing one subfolder per student.

  Optional:
    --holdout_seed    INT   Secret seed for holdout data (default: 7777).
                            Change this each semester to prevent hardcoding.
    --holdout_days    INT   Days of holdout price data to generate (default: 30).
    --rl_episodes     INT   Number of episodes per RL evaluation (default: 20).
    --output          FILE  Output CSV path (default: leaderboard.csv).
    --verbose               Print detailed per-student results.

SUBMISSION STRUCTURE
--------------------
Each student submission folder should contain:

  submissions/
  └── student_UNI/              ← one folder per student
      ├── ml_model.pkl          ← saved with save_ml_model() from ml_forecaster.py
      ├── lstm_model.pth        ← saved with save_lstm_model() from dl_forecaster.py
      ├── rl_agent.zip          ← saved with PPO.save() from stable-baselines3
      └── student_info.json     ← {"name": "Jane Smith", "uni": "js1234"}

Students who did not submit a particular model receive NaN for that metric.

EVALUATION CRITERIA
-------------------
All three pillars are evaluated on the same holdout scenario:

  1. ML Model (ml_model.pkl):
     - Generates holdout prices with holdout_seed (unseen during training)
     - Engineers features using the baseline engineer_features()
     - Evaluates: MAE ($/MWh), RMSE ($/MWh), R² on next-hour price prediction
     - Lower MAE/RMSE = better; higher R² = better.

  2. LSTM Model (lstm_model.pth):
     - Same holdout price data as ML
     - Uses saved normalization statistics from the student's training
     - Evaluates: MAE ($/MWh), RMSE ($/MWh), R²
     - Lower MAE/RMSE = better.

  3. RL Agent (rl_agent.zip):
     - Runs rl_episodes episodes on a fixed holdout charging scenario
       (EV schedules and price curve from holdout_seed)
     - Evaluates: mean net charging cost ($), % EVs meeting departure target
     - Lower cost + higher target compliance = better.

EXTRA CREDIT SCORING
--------------------
A combined score is computed as:

  ml_score   = baseline_ml_mae / student_ml_mae         (>1 means student beats baseline)
  lstm_score = baseline_lstm_mae / student_lstm_mae
  rl_score   = baseline_rl_cost / student_rl_cost       (>1 means student beats baseline)

  combined_score = 0.3 × ml_score + 0.3 × lstm_score + 0.4 × rl_score

  combined_score > 1.0 → student beats all baselines (earns extra credit)
  combined_score ≈ 1.0 → on par with provided baselines
  combined_score < 1.0 → below baseline on average

The combined score is used for ranking. Grading policy is left to the
instructor's discretion.

STEP-BY-STEP GUIDE FOR CAs
--------------------------
1. Collect all student submission folders into one directory (e.g., ./submissions/).
   Each student's folder should be named by their UNI (e.g., submissions/js1234/).

2. Run the autograder:
     python autograder.py --submissions_dir ./submissions --verbose

3. The autograder will:
   a. Generate holdout data using the secret seed (students never see this data).
   b. Train baseline models on holdout data (~5-10 min for ML+LSTM+RL).
   c. Evaluate each student's models against the holdout data.
   d. Output a leaderboard CSV and print results to the console.

4. The leaderboard CSV contains one row per student plus a baseline row.
   Share the leaderboard with students (minus the baseline seed info).

5. For extra credit: students with combined_score > 1.0 beat all baselines.
   The instructor decides how much extra credit to award per score increment.

COMMON CA ISSUES
----------------
- "Student model fails to load": Usually means they modified the architecture
  but didn't save/load correctly. They get NaN for that component.
- "Baseline LSTM training takes too long": The autograder trains a quick
  baseline (20 epochs). Total autograder runtime is ~10-20 minutes for
  a class of 30 students.
- "Student's ML MAE is NaN": They probably didn't save the model or
  their custom feature engineering created columns that don't exist in
  the holdout feature set. The autograder handles this gracefully.
- "RL evaluation episodes vary a lot": This is normal. 20 episodes
  provides a reasonable estimate. Increase --rl_episodes for more precision.

IEOR E4010: AI for Operations Research and Financial Engineering
Columbia University, Spring 2026
"""

import os
import sys
import json
import argparse
import warnings
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

# Suppress noisy warnings during batch evaluation
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# Configuration — CA/Instructor Settings
# ============================================================
# IMPORTANT FOR CAs: Change the holdout seed each semester.
# If students discover the seed, they can overfit their models
# to the holdout distribution, defeating the purpose of evaluation.
# The seed controls both the holdout price data and the RL
# evaluation scenario (EV schedules + price curve).
DEFAULT_HOLDOUT_SEED = 7777

# Number of holdout price days to generate
DEFAULT_HOLDOUT_DAYS = 30

# Number of RL evaluation episodes (more = more reliable estimate)
DEFAULT_RL_EPISODES = 20

# Day index from holdout prices to use for RL evaluation.
# This selects which day's price curve is used for the RL scenario.
# Day 5 is chosen to avoid edge effects at the start of the series.
RL_EVAL_DAY_INDEX = 5


# ============================================================
# Baseline Computation
# ============================================================
def compute_baselines(
    cfg,
    holdout_train_df,
    holdout_test_df,
    holdout_test_prices: np.ndarray,
    holdout_schedules,
    holdout_price_curve: np.ndarray,
    n_rl_episodes: int,
    verbose: bool = True,
) -> Dict[str, float]:
    """Compute baseline metrics from the provided implementations.

    Runs the default ml_forecaster, dl_forecaster, and rl_agent on
    the holdout data to establish reference performance levels.
    All student scores are normalized against these baselines.

    Returns:
        Dict with keys: baseline_ml_mae, baseline_lstm_mae, baseline_rl_cost
    """
    from ml_forecaster import engineer_features, train_ml_models, evaluate_ml_models
    from dl_forecaster import train_lstm, evaluate_lstm
    from rl_agent import train_rl_agent, evaluate_rl_agent

    baselines = {}

    # --- Baseline ML ---
    if verbose:
        print("\n[Baseline] Training default ML model on holdout train data...")
    try:
        train_feat = engineer_features(holdout_train_df, cfg)
        test_feat = engineer_features(holdout_test_df, cfg)
        baseline_ml = train_ml_models(train_feat, cfg)
        ml_res = evaluate_ml_models(baseline_ml, test_feat, verbose=False)
        model_name = list(ml_res.keys())[0]
        baselines["baseline_ml_mae"] = ml_res[model_name]["mae"]
        baselines["baseline_ml_rmse"] = ml_res[model_name]["rmse"]
        baselines["baseline_ml_r2"] = ml_res[model_name]["r2"]
        if verbose:
            print(f"  Baseline ML MAE:  ${baselines['baseline_ml_mae']:.3f}/MWh")
    except Exception as e:
        print(f"  WARNING: Baseline ML failed: {e}")
        baselines["baseline_ml_mae"] = float("nan")
        baselines["baseline_ml_rmse"] = float("nan")
        baselines["baseline_ml_r2"] = float("nan")

    # --- Baseline LSTM ---
    if verbose:
        print("[Baseline] Training default LSTM on holdout train data (quick)...")
    try:
        # Use a short training run for the baseline to keep autograder fast
        from config import Config
        fast_cfg = Config()
        fast_cfg.forecast.lstm_epochs = 20
        train_prices = holdout_train_df["price_mwh"].values
        val_prices = holdout_test_prices[:1000]
        baseline_lstm = train_lstm(train_prices, val_prices=val_prices,
                                   cfg=fast_cfg, verbose=False)
        lstm_res = evaluate_lstm(baseline_lstm, holdout_test_prices, verbose=False)
        baselines["baseline_lstm_mae"] = lstm_res["mae"]
        baselines["baseline_lstm_rmse"] = lstm_res["rmse"]
        baselines["baseline_lstm_r2"] = lstm_res["r2"]
        if verbose:
            print(f"  Baseline LSTM MAE: ${baselines['baseline_lstm_mae']:.3f}/MWh")
    except Exception as e:
        print(f"  WARNING: Baseline LSTM failed: {e}")
        baselines["baseline_lstm_mae"] = float("nan")
        baselines["baseline_lstm_rmse"] = float("nan")
        baselines["baseline_lstm_r2"] = float("nan")

    # --- Baseline RL ---
    if verbose:
        print("[Baseline] Training default RL agent on holdout scenario (quick)...")
    try:
        from config import Config
        fast_cfg = Config()
        fast_cfg.rl.total_timesteps = 50_000  # Quick baseline run
        rl_result = train_rl_agent(fast_cfg, verbose=False)
        rl_eval = evaluate_rl_agent(
            rl_result["model"], cfg,
            schedules=holdout_schedules,
            price_curve=holdout_price_curve,
            n_episodes=n_rl_episodes,
            verbose=False,
        )
        baselines["baseline_rl_cost"] = rl_eval["net_cost_mean"]
        baselines["baseline_rl_targets"] = rl_eval["targets_met_mean"]
        if verbose:
            print(f"  Baseline RL cost: ${baselines['baseline_rl_cost']:.2f}")
    except Exception as e:
        print(f"  WARNING: Baseline RL failed: {e}")
        baselines["baseline_rl_cost"] = float("nan")
        baselines["baseline_rl_targets"] = float("nan")

    return baselines


# ============================================================
# Per-Student Evaluation Functions
# ============================================================
def evaluate_ml_submission(
    ml_path: str,
    holdout_test_df: pd.DataFrame,
    cfg,
    verbose: bool = False,
) -> Dict[str, float]:
    """Evaluate a student's saved ML model on holdout test data.

    HOW THIS WORKS:
    The student's saved .pkl file contains the trained model objects AND
    a list of feature column names they used during training. We call
    the baseline engineer_features() on the holdout data to generate
    the standard feature columns. If the student added custom features
    that don't exist in the holdout data, those columns are skipped
    (with a warning). The model is evaluated only on available features.

    This means students who add custom features in a SEPARATE function
    will still be evaluated fairly — the model uses whatever baseline
    features are available. Students who modified engineer_features()
    directly (which they shouldn't) may see degraded performance.

    Returns:
        Dict with mae, rmse, r2 (NaN on error)
    """
    from ml_forecaster import engineer_features, evaluate_ml_models

    try:
        import joblib
        models_dict = joblib.load(ml_path)
        if verbose:
            print(f"    Loaded ML model: {list(models_dict['models'].keys())}")

        # Engineer features — students may have extended this; we call baseline
        # to get the standard columns the model was likely trained on.
        # If the student used custom features, only columns present in both
        # the model's feature_columns AND the test set will be used.
        test_feat = engineer_features(holdout_test_df, cfg)

        # Check that required feature columns are available
        feature_cols = models_dict.get("feature_columns", [])
        available = [c for c in feature_cols if c in test_feat.columns]
        missing = [c for c in feature_cols if c not in test_feat.columns]

        if missing:
            if verbose:
                print(f"    WARNING: {len(missing)} custom feature columns missing from "
                      f"holdout data: {missing[:5]}... Using only available features.")
            # Rebuild models_dict with available feature columns only
            models_dict_eval = dict(models_dict)
            models_dict_eval["feature_columns"] = available
        else:
            models_dict_eval = models_dict

        results = evaluate_ml_models(models_dict_eval, test_feat, verbose=False)
        if not results:
            return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}

        # Use the first model's results (or pick best)
        best_name = min(results.keys(), key=lambda k: results[k]["mae"])
        return {
            "mae": results[best_name]["mae"],
            "rmse": results[best_name]["rmse"],
            "r2": results[best_name]["r2"],
            "model_type": best_name,
        }

    except Exception as e:
        if verbose:
            print(f"    ERROR evaluating ML model: {e}")
            traceback.print_exc()
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"),
                "error": str(e)}


def evaluate_lstm_submission(
    lstm_path: str,
    holdout_test_prices: np.ndarray,
    verbose: bool = False,
) -> Dict[str, float]:
    """Evaluate a student's saved LSTM model on holdout price data.

    Loads the state dict using the baseline PriceLSTM architecture.
    If the student used a custom architecture with different hidden_size
    or num_layers, the saved metadata in the .pth file handles this.

    Returns:
        Dict with mae, rmse, r2 (NaN on error)
    """
    from dl_forecaster import load_lstm_model, evaluate_lstm

    try:
        lstm_dict = load_lstm_model(lstm_path, verbose=False if not verbose else True)
        results = evaluate_lstm(lstm_dict, holdout_test_prices, verbose=False)
        return {
            "mae": results["mae"],
            "rmse": results["rmse"],
            "r2": results["r2"],
        }

    except Exception as e:
        if verbose:
            print(f"    ERROR evaluating LSTM: {e}")
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"),
                "error": str(e)}


def evaluate_rl_submission(
    rl_path: str,
    cfg,
    holdout_schedules,
    holdout_price_curve: np.ndarray,
    n_episodes: int = 20,
    verbose: bool = False,
) -> Dict[str, float]:
    """Evaluate a student's saved RL agent on the holdout charging scenario.

    HOW THIS WORKS:
    Loads the student's PPO model (.zip file saved by Stable-Baselines3)
    and runs n_episodes deterministic rollouts on the fixed holdout
    scenario (same EV schedules and price curve for every student).

    IMPORTANT: The environment uses the DEFAULT reward function from
    environment.py, regardless of what custom reward the student used
    during training. This ensures all students are compared on the
    same cost metric. A student's custom reward might emphasize
    different tradeoffs (e.g., penalize violations more heavily),
    but the final evaluation metric is always net electricity cost.

    Returns:
        Dict with net_cost_mean, net_cost_std, targets_met_mean (NaN on error)
    """
    try:
        from stable_baselines3 import PPO
        from rl_agent import evaluate_rl_agent

        model = PPO.load(rl_path)
        if verbose:
            print(f"    RL model loaded (policy net: {model.policy.net_arch})")

        results = evaluate_rl_agent(
            model, cfg,
            schedules=holdout_schedules,
            price_curve=holdout_price_curve,
            n_episodes=n_episodes,
            verbose=False,
        )
        return {
            "net_cost_mean": results["net_cost_mean"],
            "net_cost_std": results["net_cost_std"],
            "targets_met_mean": results["targets_met_mean"],
            "targets_met_min": results["targets_met_min"],
            "v2g_revenue_mean": results["v2g_revenue_mean"],
            "penalties_mean": results["penalties_mean"],
        }

    except ImportError:
        if verbose:
            print("    stable-baselines3 not available for RL evaluation.")
        return {"net_cost_mean": float("nan"), "net_cost_std": float("nan"),
                "targets_met_mean": float("nan"), "error": "stable-baselines3 not installed"}
    except Exception as e:
        if verbose:
            print(f"    ERROR evaluating RL agent: {e}")
        return {"net_cost_mean": float("nan"), "net_cost_std": float("nan"),
                "targets_met_mean": float("nan"), "error": str(e)}


# ============================================================
# Combined Score Computation
# ============================================================
def compute_combined_score(
    student_metrics: Dict[str, float],
    baselines: Dict[str, float],
) -> float:
    """Compute the normalized combined score for extra credit ranking.

    HOW SCORING WORKS:
    Each component score is the ratio baseline_metric / student_metric.
    A ratio > 1.0 means the student's model outperformed the baseline.

    Example: If baseline ML MAE = $3.50 and student ML MAE = $2.80,
    then ml_score = 3.50 / 2.80 = 1.25 (25% better than baseline).

    The three component scores are combined with weights:
      ML=30%, LSTM=30%, RL=40%
    RL is weighted highest because the reward function design task
    is the most open-ended and has the largest impact on performance.

    Score interpretation:
      > 1.0: student beats all baselines on weighted average (extra credit)
      ≈ 1.0: on par with provided baselines
      < 1.0: below baseline on average (no extra credit, but may still pass)

    If a student didn't submit one model type, the weights are
    renormalized among the submitted components.
    """
    scores = []
    weights = []

    # ML score: baseline_mae / student_mae (higher = better)
    if (not np.isnan(student_metrics.get("ml_mae", float("nan"))) and
            not np.isnan(baselines.get("baseline_ml_mae", float("nan"))) and
            student_metrics["ml_mae"] > 0):
        ml_score = baselines["baseline_ml_mae"] / student_metrics["ml_mae"]
        scores.append(ml_score)
        weights.append(0.30)

    # LSTM score: baseline_mae / student_mae
    if (not np.isnan(student_metrics.get("lstm_mae", float("nan"))) and
            not np.isnan(baselines.get("baseline_lstm_mae", float("nan"))) and
            student_metrics["lstm_mae"] > 0):
        lstm_score = baselines["baseline_lstm_mae"] / student_metrics["lstm_mae"]
        scores.append(lstm_score)
        weights.append(0.30)

    # RL score: baseline_cost / student_cost (higher = better)
    if (not np.isnan(student_metrics.get("rl_cost", float("nan"))) and
            not np.isnan(baselines.get("baseline_rl_cost", float("nan"))) and
            student_metrics["rl_cost"] > 0):
        rl_score = baselines["baseline_rl_cost"] / student_metrics["rl_cost"]
        scores.append(rl_score)
        weights.append(0.40)

    if not scores:
        return float("nan")

    # Normalize weights to sum to 1
    total_w = sum(weights)
    weighted_score = sum(s * w / total_w for s, w in zip(scores, weights))
    return weighted_score


# ============================================================
# Main Autograder
# ============================================================
def run_autograder(
    submissions_dir: str,
    holdout_seed: int = DEFAULT_HOLDOUT_SEED,
    holdout_days: int = DEFAULT_HOLDOUT_DAYS,
    n_rl_episodes: int = DEFAULT_RL_EPISODES,
    output_csv: str = "leaderboard.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the full autograder on all submissions.

    Args:
        submissions_dir: Path to directory containing student submission folders.
        holdout_seed:    Seed for generating holdout data (keep secret from students).
        holdout_days:    Number of days in holdout price dataset.
        n_rl_episodes:   Episodes per RL evaluation.
        output_csv:      Path to save the leaderboard CSV.
        verbose:         Print detailed progress.

    Returns:
        DataFrame with one row per student, sorted by combined_score descending.
    """
    from config import Config
    from data_utils import (
        generate_synthetic_prices,
        generate_ev_schedules,
        get_daily_price_curve,
        train_test_split_prices,
    )

    # ----------------------------------------------------------
    # 1. Generate holdout data (hidden from students)
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"IEOR E4010 — EV Charging Project Autograder")
    print(f"{'='*60}")
    print(f"Submissions directory: {submissions_dir}")
    print(f"Holdout seed: {holdout_seed} (keep confidential)")
    print(f"Holdout days: {holdout_days}")
    print(f"RL evaluation episodes: {n_rl_episodes}")
    print(f"{'='*60}\n")

    cfg = Config()

    print("Generating holdout dataset...")
    holdout_prices_df = generate_synthetic_prices(cfg, num_days=holdout_days, seed=holdout_seed)

    # Use last 20% of holdout data as test set
    holdout_train_df, holdout_test_df = train_test_split_prices(holdout_prices_df, cfg)
    holdout_test_prices = holdout_test_df["price_mwh"].values

    # Fixed EV scenario for RL evaluation
    holdout_schedules = generate_ev_schedules(cfg, seed=holdout_seed)
    holdout_price_curve = get_daily_price_curve(holdout_prices_df,
                                                day_index=RL_EVAL_DAY_INDEX, cfg=cfg)
    print(f"  Holdout: {len(holdout_prices_df)} hours total, "
          f"{len(holdout_test_df)} test hours\n")

    # ----------------------------------------------------------
    # 2. Compute baselines
    # ----------------------------------------------------------
    print("Computing baseline metrics (this may take a few minutes)...")
    baselines = compute_baselines(
        cfg,
        holdout_train_df,
        holdout_test_df,
        holdout_test_prices,
        holdout_schedules,
        holdout_price_curve,
        n_rl_episodes=n_rl_episodes,
        verbose=verbose,
    )
    print(f"\nBaseline Summary:")
    print(f"  ML  MAE:  ${baselines.get('baseline_ml_mae', float('nan')):.3f}/MWh")
    print(f"  LSTM MAE: ${baselines.get('baseline_lstm_mae', float('nan')):.3f}/MWh")
    print(f"  RL cost:  ${baselines.get('baseline_rl_cost', float('nan')):.2f}")

    # ----------------------------------------------------------
    # 3. Discover submission folders
    # ----------------------------------------------------------
    submissions_path = Path(submissions_dir)
    if not submissions_path.exists():
        print(f"\nERROR: Submissions directory '{submissions_dir}' not found.")
        return pd.DataFrame()

    student_dirs = sorted([
        d for d in submissions_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if not student_dirs:
        print(f"\nERROR: No submission folders found in '{submissions_dir}'.")
        print("Expected structure: submissions/<student_uni>/[ml_model.pkl, lstm_model.pth, rl_agent.zip]")
        return pd.DataFrame()

    print(f"\nFound {len(student_dirs)} submission(s).")

    # ----------------------------------------------------------
    # 4. Evaluate each student
    # ----------------------------------------------------------
    results = []

    for student_dir in student_dirs:
        folder_name = student_dir.name
        print(f"\n{'─'*50}")
        print(f"Evaluating: {folder_name}")

        # Load student info
        info_path = student_dir / "student_info.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            student_name = info.get("name", folder_name)
            student_uni = info.get("uni", folder_name)
        else:
            student_name = folder_name
            student_uni = folder_name

        row = {
            "folder": folder_name,
            "name": student_name,
            "uni": student_uni,
        }

        # --- ML Evaluation ---
        ml_path = student_dir / "ml_model.pkl"
        if ml_path.exists():
            print(f"  [ML] Evaluating ml_model.pkl...")
            ml_metrics = evaluate_ml_submission(
                str(ml_path), holdout_test_df, cfg, verbose=verbose
            )
            row["ml_mae"] = ml_metrics.get("mae", float("nan"))
            row["ml_rmse"] = ml_metrics.get("rmse", float("nan"))
            row["ml_r2"] = ml_metrics.get("r2", float("nan"))
            if "error" in ml_metrics:
                row["ml_error"] = ml_metrics["error"]
                print(f"    ERROR: {ml_metrics['error'][:80]}")
            else:
                print(f"    MAE: ${row['ml_mae']:.3f}/MWh  |  "
                      f"RMSE: ${row['ml_rmse']:.3f}  |  R²: {row['ml_r2']:.4f}")
        else:
            print(f"  [ML] ml_model.pkl not found — skipping.")
            row["ml_mae"] = float("nan")
            row["ml_rmse"] = float("nan")
            row["ml_r2"] = float("nan")

        # --- LSTM Evaluation ---
        lstm_path = student_dir / "lstm_model.pth"
        if lstm_path.exists():
            print(f"  [LSTM] Evaluating lstm_model.pth...")
            lstm_metrics = evaluate_lstm_submission(
                str(lstm_path), holdout_test_prices, verbose=verbose
            )
            row["lstm_mae"] = lstm_metrics.get("mae", float("nan"))
            row["lstm_rmse"] = lstm_metrics.get("rmse", float("nan"))
            row["lstm_r2"] = lstm_metrics.get("r2", float("nan"))
            if "error" in lstm_metrics:
                row["lstm_error"] = lstm_metrics["error"]
                print(f"    ERROR: {lstm_metrics['error'][:80]}")
            else:
                print(f"    MAE: ${row['lstm_mae']:.3f}/MWh  |  "
                      f"RMSE: ${row['lstm_rmse']:.3f}  |  R²: {row['lstm_r2']:.4f}")
        else:
            print(f"  [LSTM] lstm_model.pth not found — skipping.")
            row["lstm_mae"] = float("nan")
            row["lstm_rmse"] = float("nan")
            row["lstm_r2"] = float("nan")

        # --- RL Evaluation ---
        rl_path = student_dir / "rl_agent.zip"
        if rl_path.exists():
            print(f"  [RL] Evaluating rl_agent.zip ({n_rl_episodes} episodes)...")
            rl_metrics = evaluate_rl_submission(
                str(rl_path), cfg,
                holdout_schedules, holdout_price_curve,
                n_episodes=n_rl_episodes,
                verbose=verbose,
            )
            row["rl_cost"] = rl_metrics.get("net_cost_mean", float("nan"))
            row["rl_cost_std"] = rl_metrics.get("net_cost_std", float("nan"))
            row["rl_targets_pct"] = rl_metrics.get("targets_met_mean", float("nan"))
            row["rl_v2g_revenue"] = rl_metrics.get("v2g_revenue_mean", float("nan"))
            if "error" in rl_metrics:
                row["rl_error"] = rl_metrics["error"]
                print(f"    ERROR: {rl_metrics['error'][:80]}")
            else:
                print(f"    Net cost: ${row['rl_cost']:.2f} ± ${row['rl_cost_std']:.2f}  |  "
                      f"Targets: {row['rl_targets_pct']:.1f}/{cfg.fleet.num_evs}")
        else:
            print(f"  [RL] rl_agent.zip not found — skipping.")
            row["rl_cost"] = float("nan")
            row["rl_cost_std"] = float("nan")
            row["rl_targets_pct"] = float("nan")

        # --- Combined Score ---
        row["combined_score"] = compute_combined_score(row, baselines)
        print(f"  Combined score: {row['combined_score']:.4f}  "
              f"({'BEATS BASELINE ✓' if row['combined_score'] > 1.0 else 'below baseline'})")

        results.append(row)

    # ----------------------------------------------------------
    # 5. Build and display leaderboard
    # ----------------------------------------------------------
    if not results:
        print("\nNo results to display.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)
    df.index += 1  # 1-indexed ranking

    # ----------------------------------------------------------
    # 6. Print final leaderboard
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("FINAL LEADERBOARD")
    print(f"{'='*60}")
    print(f"\nBaselines: ML MAE=${baselines.get('baseline_ml_mae', float('nan')):.2f} | "
          f"LSTM MAE=${baselines.get('baseline_lstm_mae', float('nan')):.2f} | "
          f"RL cost=${baselines.get('baseline_rl_cost', float('nan')):.2f}\n")

    display_cols = ["name", "uni", "ml_mae", "lstm_mae", "rl_cost",
                    "rl_targets_pct", "combined_score"]
    display_cols = [c for c in display_cols if c in df.columns]

    with pd.option_context("display.float_format", "{:.3f}".format,
                           "display.max_columns", 15,
                           "display.width", 100):
        print(df[display_cols].to_string())

    print(f"\nStudents beating baseline: "
          f"{(df['combined_score'] > 1.0).sum()} / {len(df)}")

    # ----------------------------------------------------------
    # 7. Save leaderboard
    # ----------------------------------------------------------
    # Add baseline row for reference
    baseline_row = {
        "folder": "PROVIDED_BASELINE",
        "name": "Baseline (provided code)",
        "uni": "—",
        "ml_mae": baselines.get("baseline_ml_mae"),
        "ml_rmse": baselines.get("baseline_ml_rmse"),
        "ml_r2": baselines.get("baseline_ml_r2"),
        "lstm_mae": baselines.get("baseline_lstm_mae"),
        "lstm_rmse": baselines.get("baseline_lstm_rmse"),
        "lstm_r2": baselines.get("baseline_lstm_r2"),
        "rl_cost": baselines.get("baseline_rl_cost"),
        "rl_targets_pct": baselines.get("baseline_rl_targets"),
        "combined_score": 1.0,
    }
    # Prepend baseline row (unranked)
    baseline_df = pd.DataFrame([baseline_row])
    output_df = pd.concat([baseline_df, df.reset_index(drop=True)], ignore_index=True)
    output_df.to_csv(output_csv, index=False)
    print(f"\nLeaderboard saved to '{output_csv}'")

    return df


# ============================================================
# CLI Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="IEOR E4010 EV Charging Project Autograder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--submissions_dir", required=True,
        help="Directory containing student submission subfolders",
    )
    parser.add_argument(
        "--holdout_seed", type=int, default=DEFAULT_HOLDOUT_SEED,
        help=f"Holdout data seed (default: {DEFAULT_HOLDOUT_SEED}). Keep confidential.",
    )
    parser.add_argument(
        "--holdout_days", type=int, default=DEFAULT_HOLDOUT_DAYS,
        help=f"Days of holdout price data (default: {DEFAULT_HOLDOUT_DAYS})",
    )
    parser.add_argument(
        "--rl_episodes", type=int, default=DEFAULT_RL_EPISODES,
        help=f"RL evaluation episodes per student (default: {DEFAULT_RL_EPISODES})",
    )
    parser.add_argument(
        "--output", default="leaderboard.csv",
        help="Output CSV path (default: leaderboard.csv)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed per-student evaluation output",
    )

    args = parser.parse_args()

    leaderboard = run_autograder(
        submissions_dir=args.submissions_dir,
        holdout_seed=args.holdout_seed,
        holdout_days=args.holdout_days,
        n_rl_episodes=args.rl_episodes,
        output_csv=args.output,
        verbose=args.verbose,
    )

    return 0 if leaderboard is not None and len(leaderboard) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
