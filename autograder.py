#!/usr/bin/env python3
"""
autograder.py — CA Benchmark Autograder for EV Smart Charging Project
======================================================================

OVERVIEW
--------
This script evaluates all student-submitted model files against a hidden
holdout dataset and produces a ranked leaderboard. It is intended for
course staff (CAs/instructors) only; students do not run this file.

QUICK START FOR CAs
-------------------
1. Collect student submission zips from Courseworks.
   Unzip each into a subfolder named after the student's UNI:

     submissions/
     ├── js1234/
     │   ├── ml_model.pkl
     │   ├── lstm_model.pth
     │   ├── rl_agent.zip
     │   └── student_info.json
     └── ab5678/
         └── ...

   NOTE: The submission packaging cell (Section 10) in main.ipynb creates
   this structure automatically. Students submit a zip that extracts to
   a folder named submission_UNI/ — rename that folder to the UNI for clarity.

2. Run the autograder (evaluates all three pillars, ~60-90 min for 30 students):

     python autograder.py --submissions_dir ./submissions --verbose

3. Results are saved to leaderboard.csv:
   - combined_score > 1.0 → student beats all baselines (extra credit)
   - combined_score ≈ 1.0 → on par with provided baselines
   - combined_score < 1.0 → below baseline performance

4. To evaluate only one pillar (useful for staged deadlines):

     python autograder.py --submissions_dir ./submissions --pillar ml
     python autograder.py --submissions_dir ./submissions --pillar lstm
     python autograder.py --submissions_dir ./submissions --pillar rl

5. To check submission structure without running models (fast dry run):

     python autograder.py --submissions_dir ./submissions --dry_run

6. Change the holdout seed each semester to prevent students from hardcoding
   to the holdout distribution:

     python autograder.py --submissions_dir ./submissions --holdout_seed 9999

TIMING ESTIMATES (on a standard laptop CPU)
--------------------------------------------
  Baseline computation: ~5-10 min (ML + LSTM + RL training)
  Per-student ML eval:  ~30 sec
  Per-student LSTM eval: ~30 sec
  Per-student RL eval:  ~2 min (20 episodes × 96 steps)
  Total for 30 students: ~75-90 min

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
    --pillar          STR   Evaluate only one pillar: ml, lstm, rl, or all (default: all).
    --dry_run               Check submission structure without running models.
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
# Configuration
# ============================================================
# Holdout seed is known only to course staff.
# Change between semesters to prevent students from overfitting
# to the holdout distribution.
DEFAULT_HOLDOUT_SEED = 7777

# Number of holdout price days to generate
DEFAULT_HOLDOUT_DAYS = 30

# Number of RL evaluation episodes (more = more reliable estimate)
DEFAULT_RL_EPISODES = 20

# Day index from holdout prices to use for RL evaluation
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

    The student's model may have been trained with custom features.
    We use the feature_columns stored in the pickle to extract the
    right columns from the holdout test data.

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
        # NOTE: load_lstm_model() does not accept a verbose parameter — it always
        # prints a brief "Loaded from ..." message. This is intentional behavior.
        lstm_dict = load_lstm_model(lstm_path)
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

    Loads the PPO model and runs n_episodes deterministic rollouts on
    the fixed holdout scenario. Uses the DEFAULT reward function from
    environment.py regardless of what reward function the student used
    during training — this ensures a fair cost comparison.

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

    Score > 1.0: student beats all baselines (earns extra credit)
    Score ≈ 1.0: on par with provided baselines
    Score < 1.0: below baseline performance

    Weights: ML=30%, LSTM=30%, RL=40%
    (RL weighted highest as it is the most open-ended task)
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
    pillar: str = "all",
    dry_run: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the full autograder on all submissions.

    Args:
        submissions_dir: Path to directory containing student submission folders.
        holdout_seed:    Seed for generating holdout data (keep secret from students).
        holdout_days:    Number of days in holdout price dataset.
        n_rl_episodes:   Episodes per RL evaluation.
        output_csv:      Path to save the leaderboard CSV.
        pillar:          Which pillar(s) to evaluate: "ml", "lstm", "rl", or "all".
                         Use for staged deadlines where not all pillars are due together.
        dry_run:         If True, check submission structure only (no model evaluation).
                         Useful for confirming correct file names before the deadline.
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
    import datetime

    # Validate pillar argument
    valid_pillars = {"ml", "lstm", "rl", "all"}
    if pillar not in valid_pillars:
        raise ValueError(f"pillar must be one of {valid_pillars}, got '{pillar}'")

    eval_ml   = pillar in ("ml",   "all")
    eval_lstm = pillar in ("lstm", "all")
    eval_rl   = pillar in ("rl",   "all")

    print(f"\n{'='*60}")
    print(f"IEOR E4010 — EV Charging Project Autograder")
    print(f"{'='*60}")
    print(f"Run time:              {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Submissions directory: {submissions_dir}")
    print(f"Holdout seed:          {holdout_seed} (keep confidential)")
    print(f"Holdout days:          {holdout_days}")
    print(f"Pillar(s) evaluated:   {pillar}")
    print(f"RL evaluation episodes:{n_rl_episodes}")
    if dry_run:
        print(f"Mode:                  DRY RUN (structure check only — no models evaluated)")
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
    # 2. Compute baselines (skip for dry_run — baselines not needed for structure check)
    # ----------------------------------------------------------
    baselines = {}
    if not dry_run:
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
    # 3b. Dry-run mode: just check file presence, no evaluation
    # ----------------------------------------------------------
    if dry_run:
        print("\n--- DRY RUN: Checking submission structure ---")
        required = {
            "ml":   "ml_model.pkl",
            "lstm": "lstm_model.pth",
            "rl":   "rl_agent.zip",
            "info": "student_info.json",
        }
        all_ok = True
        for student_dir in student_dirs:
            folder_name = student_dir.name
            missing = [name for key, name in required.items()
                       if not (student_dir / name).exists()]
            status = "OK" if not missing else f"MISSING: {missing}"
            print(f"  {folder_name:30s} {status}")
            if missing:
                all_ok = False
        print(f"\nDry-run complete. {'All submissions complete.' if all_ok else 'Some files missing (see above).'}")
        return pd.DataFrame()

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
            print(f"  WARNING: student_info.json not found — using folder name as identifier.")

        row = {
            "folder": folder_name,
            "name": student_name,
            "uni": student_uni,
        }

        # --- ML Evaluation ---
        ml_path = student_dir / "ml_model.pkl"
        if eval_ml:
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
        else:
            print(f"  [ML] Skipped (--pillar {pillar})")
            row["ml_mae"] = float("nan")
            row["ml_rmse"] = float("nan")
            row["ml_r2"] = float("nan")

        # --- LSTM Evaluation ---
        lstm_path = student_dir / "lstm_model.pth"
        if eval_lstm:
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
        else:
            print(f"  [LSTM] Skipped (--pillar {pillar})")
            row["lstm_mae"] = float("nan")
            row["lstm_rmse"] = float("nan")
            row["lstm_r2"] = float("nan")

        # --- RL Evaluation ---
        rl_path = student_dir / "rl_agent.zip"
        if eval_rl:
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
        else:
            print(f"  [RL] Skipped (--pillar {pillar})")
            row["rl_cost"] = float("nan")
            row["rl_cost_std"] = float("nan")
            row["rl_targets_pct"] = float("nan")

        # --- Combined Score ---
        row["combined_score"] = compute_combined_score(row, baselines)
        beat = row["combined_score"] > 1.0
        if not np.isnan(row["combined_score"]):
            print(f"  Combined score: {row['combined_score']:.4f}  "
                  f"({'BEATS BASELINE ✓' if beat else 'below baseline'})")

        # --- Save per-student JSON (useful for detailed CA records) ---
        student_json_path = student_dir / "autograder_result.json"
        try:
            result_to_save = {k: (float(v) if isinstance(v, float) else v)
                              for k, v in row.items()}
            result_to_save["baselines"] = {
                k: (float(v) if isinstance(v, float) else v)
                for k, v in baselines.items()
            }
            result_to_save["autograder_timestamp"] = __import__('datetime').datetime.now().isoformat()
            with open(student_json_path, "w") as f:
                json.dump(result_to_save, f, indent=2)
        except Exception as e:
            pass  # Non-critical — don't fail the whole run

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
        "--pillar", default="all", choices=["ml", "lstm", "rl", "all"],
        help="Evaluate only one pillar (ml/lstm/rl) or all (default: all). "
             "Useful for staged deadlines.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Check submission file structure only — do not run models. "
             "Run this before the deadline to catch missing files.",
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
        pillar=args.pillar,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    return 0 if leaderboard is not None and len(leaderboard) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
