# Large-Scale EV Smart Charging: A Four-Pillar AI Project

**IEOR E4010 — Artificial Intelligence for Operations Research and Financial Engineering**
Columbia University · Spring 2026 · Final Project

---

## For Students: Quick Start

**Primary workflow: Google Colab + GitHub**

1. Open `main.ipynb` in Google Colab (link provided on Courseworks)
2. Run the Colab setup cells at the top (git clone + pip install)
3. Work through sections 0–9 from top to bottom
4. Complete the three **★ Student TODO** sections (Sections 4, 5, 6)
5. Save your trained models and submit via Courseworks (Section 10)

**Local workflow (alternative):**
```bash
git clone https://github.com/COURSE_REPO/Smart_EV_Charging.git
cd Smart_EV_Charging
pip install -r requirements.txt
jupyter notebook main.ipynb
```

> **Important:** All student work belongs in `main.ipynb`. The `.py` files are
> provided infrastructure — read them to understand the implementations, but
> do not modify them. The notebook uses extension hooks (custom feature functions,
> `custom_reward_fn`, configurable hyperparameters) so you can improve performance
> entirely within the notebook.

---

## Project Overview

You are the fleet operations manager at a delivery company depot. Every evening,
**20 electric vans** return and must be charged to ≥90% battery by their morning
departure — but a shared **150 kW transformer** means you cannot charge them all at once.

Your job: **minimize electricity cost** while meeting every vehicle's deadline.

This project integrates four AI pillars into one working system:

| Pillar | Module | What It Does |
|--------|--------|--------------|
| **1 — Machine Learning** | `ml_forecaster.py` | XGBoost forecasts next-hour electricity prices from tabular features |
| **2 — Deep Learning** | `dl_forecaster.py` | LSTM learns price patterns from raw sequential data |
| **3 — Reinforcement Learning** | `rl_agent.py` | PPO agent learns a real-time charging policy through simulation |
| **4 — Agentic AI** | `agentic_ai.py` | GPT-4o orchestrator with tool-calling ties all modules together |

---

## Problem Specification

### Core Parameters

| Parameter | Value | Why This Value |
|-----------|-------|----------------|
| Fleet size | 20 EVs | Large enough that coupling constraints dominate (20×11=220 kW > 150 kW limit) |
| Battery capacity | 60 kWh | Ford E-Transit commercial van specification |
| Charge/discharge rate | ±11 kW | Level 2 AC bidirectional (supports V2G) |
| **Transformer limit** | **150 kW** | 150 < 220 kW theoretical max → must coordinate charging |
| Target departure SoC | ≥90% | Operations requirement for a full day's delivery route |
| Arrival SoC | 20–60% (μ=35%) | Stochastic — varies by vehicle and daily delivery distance |
| Arrival window | 2 PM – 8 PM | Afternoon return from delivery routes |
| Departure window | 5 AM – 9 AM | Morning dispatch |
| Time resolution | 15 minutes | 96 steps per 24-hour episode |
| Simulation horizon | 24 hours | One overnight charging cycle |
| V2G degradation cost | $0.04/kWh | Battery wear cost for bidirectional operation |

### Why the Transformer Limit Creates an Interesting Problem

At 150 kW, the depot can simultaneously charge at most **13.6 EVs** at full power
(150 ÷ 11 = 13.6). With up to 20 EVs plugged in at once, the controller must
*choose* which EVs to charge each step, and at what rate. This coupling between
vehicles is what makes per-EV greedy strategies suboptimal and why coordination matters.

### Vehicle-to-Grid (V2G)

EVs can also *discharge* energy back to the grid (action < 0). This earns V2G
revenue when electricity prices are high, but causes battery degradation ($0.04/kWh).
An intelligent agent learns when V2G is profitable enough to offset the degradation
cost — roughly when `price > degradation_cost × 1000 = $40/MWh`.

---

## How the Data Works

### Electricity Prices (Synthetic CAISO-Style)

We generate **365 days of synthetic hourly prices** calibrated to California's CAISO
(California Independent System Operator) wholesale market structure:

| Time Period | Typical Price | Cause |
|-------------|--------------|-------|
| Overnight (midnight–6 AM) | $15–25/MWh | Low demand; baseload surplus |
| Morning ramp (6–9 AM) | $40–80/MWh | Commuter demand, industrial startup |
| Solar dip (10 AM–2 PM) | $10–20/MWh | California's "duck curve" — solar overproduction |
| Evening peak (5–9 PM) | $60–120/MWh | Solar drops off, demand surges; gas peakers fire |

Additional variability:
- Seasonal patterns: ~15% higher in summer (AC cooling) and winter (heating)
- Weekend/weekday difference: ~10% lower on weekends
- Random noise: ±$5/MWh Gaussian
- Rare price spikes (~5% of hours): $150–300/MWh extreme events
- Occasional negative prices (~2%): excess renewable generation

**Train/test split:** 80/20 chronological split. The first 292 days are training data,
the last 73 days are the test set. **Never shuffle time series data** — shuffling causes
data leakage (future prices appear in training features), producing unrealistically
optimistic metrics.

### EV Fleet Schedules (Stochastic)

Each episode generates a fresh set of 20 EV schedules with randomized:
- **Arrival times:** Uniform in [2 PM, 8 PM] (steps 8–32 from noon start)
- **Arrival SoC:** Truncated normal, μ=35%, σ=10%, range [20%, 60%]
- **Departure times:** Uniform in [5 AM, 9 AM] (steps 68–84)

The stochastic schedules ensure the RL agent trains on diverse scenarios
(not just one fixed scenario), improving generalization.

---

## How the LP Benchmark Works

### Why Complete Foresight?

The LP optimizer (`optimizer.py`) solves the charging problem to **mathematical
global optimality** — but it requires knowing all future electricity prices,
all EV arrival times, and all arrival SoC values **before the episode begins**.

We use it as the gold standard benchmark for two reasons:
1. **Theoretical ceiling:** The LP gives the minimum achievable cost on a scenario.
   Any real-time algorithm can only do equal to or worse.
2. **Cost of uncertainty:** `gap = (RL_cost − LP_cost) / |LP_cost|` measures how
   much the real-time agent pays for not knowing future prices.

### The LP Formulation

**Decision variables:**
- `p_charge[i,t]` — power drawn from grid for EV `i` at step `t` (kW), ≥ 0
- `p_discharge[i,t]` — power injected into grid by EV `i` at step `t` (V2G, kW), ≥ 0

**Objective (minimize):**
```
Σ_t Σ_i  price[t]/1000 × p_charge[i,t] × Δt          ← electricity cost ($)
        − price[t]/1000 × p_discharge[i,t] × Δt       ← V2G revenue ($)
        + $0.04 × p_discharge[i,t] × Δt               ← degradation cost ($)
```
where Δt = 0.25 hours (15-minute steps).

**Constraints:**
- Power bounds: 0 ≤ p_charge[i,t] ≤ 11 kW (only when EV i is connected)
- SoC dynamics: SoC[i,t+1] = SoC[i,t] + (η_c · p_charge − p_discharge/η_d) · Δt / capacity_kWh
- SoC bounds: 5% ≤ SoC[i,t] ≤ 100%
- Departure target: SoC[i, departure_step] ≥ 90%
- **Transformer limit:** Σ_i p_charge[i,t] ≤ 150 kW at every time step

Solved with `scipy.optimize.linprog` (HiGHS backend), typically in < 1 second.

---

## How the ML Forecaster Works

### Input Features (25+ baseline features)

The ML model (`ml_forecaster.py`) predicts `price[t+1]` — the next hour's price.
All features are computed from `price[t]` and earlier (no lookahead).

| Feature Category | Examples | Count |
|------------------|---------|-------|
| Cyclical time encodings | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `month_sin/cos` | 6 |
| Short lags | `price_lag_1h`, `..._2h`, `..._3h`, `..._6h`, `..._12h`, `..._24h` | 6 |
| Seasonal lags | `price_same_hour_yesterday` (lag-24), `price_same_hour_last_week` (lag-168) | 2 |
| Rolling statistics | mean and std over 6h, 24h, 168h windows | 6 |
| Price momentum | `price_diff_1h`, `price_diff_24h` | 2 |
| Market structure | `is_morning_peak`, `is_solar_dip`, `is_evening_peak`, `is_overnight` | 4 |
| Calendar | `is_weekend` | 1 |

**Why cyclical encodings?** Raw `hour` treats hour=23 and hour=0 as "far apart"
(distance=23). Sine/cosine encoding maps the 24-hour cycle onto a unit circle so
adjacent hours are nearby in feature space: `hour_sin = sin(2π × hour / 24)`.

### Model

We train **XGBoost** (200 trees, max_depth=6, lr=0.1) and **Random Forest** (200 trees,
max_depth=10). Both are gradient-boosted or bagged ensembles of decision trees that
handle tabular features well without normalization.

Evaluation: **MAE** ($/MWh), **RMSE**, **R²** on the chronological test set.

### Student Task: Extend Feature Engineering (Section 4)

Write `my_engineer_features(df, cfg)` in the notebook. Always start by calling
`engineer_features(df, cfg)` to get the 25 baseline features, then add your own.
The autograder handles custom features gracefully — columns not present in holdout
data are silently dropped, so the baseline features always act as a performance floor.

---

## How the LSTM Forecaster Works

### Sequential Input

Unlike XGBoost (tabular features), the LSTM reads **raw price sequences**:

```
Input:  sliding window of 168 consecutive hourly prices → shape (batch, 168, 1)
Target: price at step 169
```

**Why 168 steps (1 week)?** Captures both daily cycles (24h) and weekly seasonality
(weekday vs. weekend price patterns). Longer windows generally help but increase
training time.

### Architecture (baseline)

```
Input: (batch, 168, 1)  ← 1 week of hourly prices, 1 feature
→ LSTM Layer 1 (hidden_size=64)
→ LSTM Layer 2 (hidden_size=64)
→ Dropout(0.2)
→ Linear(64 → 32) → ReLU
→ Linear(32 → 1)
Output: scalar price prediction ($/MWh), denormalized
```

### Training Details

- **Normalization:** prices z-scored as `(price − μ_train) / σ_train`. The saved model
  stores `price_mean` and `price_std` (training set statistics). These are used at
  inference time to denormalize predictions.
- **Optimizer:** Adam (lr=0.001) with ReduceLROnPlateau scheduler
- **Early stopping:** patience=10 epochs on validation loss
- **Gradient clipping:** max_norm=1.0

### 24-Hour Autoregressive Forecasting

For planning purposes, the model predicts 24 hours ahead by **feeding predictions
back as input**: predict t+1, append to window, predict t+2, etc. Errors compound
over 24 steps, so long-horizon accuracy is lower than 1-step accuracy.

### Student Task: Improve Architecture (Section 5)

Modify `cfg.forecast.lstm_hidden_size`, `lstm_num_layers`, `lstm_seq_len`, etc.,
or subclass `PriceLSTM` to change the architecture. Pass your class to
`train_lstm(model_class=YourClass)`.

---

## How the RL Agent Works

### MDP Formulation

The RL agent controls the charging problem as a Markov Decision Process (MDP)
over 96 time steps (one 24-hour episode).

### State Space (63-dimensional)

At each time step, the agent observes a 63-dim vector:

**Per-EV features (3 × 20 = 60 values, EV i at indices 3i..3i+2):**

| Feature | Range | Meaning |
|---------|-------|---------|
| `current_soc` | [0, 1] | Current battery level |
| `time_to_depart` | [0, 1] | Urgency: 1=just arrived, 0=departing now |
| `is_connected` | {0, 1} | Whether EV is currently plugged in |

**Global features (indices 60–62):**

| Feature | Range | Meaning |
|---------|-------|---------|
| `norm_time` | [0, 1] | Fraction of day elapsed — correlates with price patterns |
| `norm_price` | [0, 1] | Current price / max observed price — today's relative price |
| `load_fraction` | [0, 1] | Last step's total power / 150 kW transformer limit |

**Critical:** The agent sees `norm_price` (current price) but **NOT future prices**.
It must learn from `norm_time` that "it's 2 AM, prices are typically low." This is
what distinguishes RL from the LP (which has complete price foresight).

### Action Space (20-dimensional continuous)

One scalar per EV, clipped to **[-1, +1]**:

| Value | Interpretation |
|-------|---------------|
| +1.0 | Charge at full rate (+11 kW) |
| 0.0 | Idle |
| -1.0 | Discharge at full rate (V2G, -11 kW) |

If total charging power > 150 kW, the environment **proportionally scales all
charging actions down** to exactly hit the transformer limit.

### Reward Function (default)

```
reward = -(price_weight × step_cost) - deadline_penalty - overload_penalty

step_cost = charging_cost - v2g_revenue + degradation_cost
          = Σ_i charging_energy_i × price/1000
            - Σ_i discharging_energy_i × price/1000
            + Σ_i discharging_energy_i × $0.04

deadline_penalty: fires when EV departs with SoC < 90%
                  = penalty_weight × (0.90 - actual_soc)  [sparse: fires only at departure]
overload_penalty: fires if total power > 150 kW after scaling (rare with proper reward)
```

**Default weights** (from `config.py`):
- `reward_price_weight = 1.0`
- `reward_deadline_penalty = 100.0`
- `reward_overload_penalty = 50.0`

The `deadline_penalty` is **sparse** — it fires only at the moment of departure.
This makes learning harder but is more realistic than a dense progress-based reward.

### PPO Algorithm

**PPO (Proximal Policy Optimization)** trains two networks jointly:
- **Actor (policy):** state (63-dim) → Gaussian distribution over actions (20-dim)
  - Architecture: Linear(63→256) → ReLU → Linear(256→128) → ReLU → Linear(128→20)
- **Critic (value function):** state → scalar expected return (same architecture)

PPO uses a "clipped surrogate objective" to limit how much the policy changes per
update, preventing destabilizing large gradient steps.

**Training:** 100k–500k timesteps, ~5–30 min on CPU. Each rollout collects 2048
environment steps across random scenarios; 10 mini-batch gradient updates follow.

### Expected Strategy Ranking (net cost)

```
LP Optimal ≤ RL Agent ≤ Round Robin ≤ ALAP ≤ ASAP  (lower cost = better)
```

The gap between LP and RL is the **cost of uncertainty** — the economic value
of perfect 24-hour price foresight.

### Student Task: Reward Design + Hyperparameter Tuning (Section 6)

Define `my_custom_reward(step_cost, deadline_penalty, ...)` and pass it to
`train_rl_agent(cfg, custom_reward_fn=my_custom_reward)`. You do NOT need to
edit `environment.py`. Also experiment with `cfg.rl.total_timesteps`,
`learning_rate`, and network architecture.

---

## Project Structure

```
Smart_EV_Charging/
├── main.ipynb           <- Student work goes here (run this in Colab)
│
├── config.py            <- All parameters in one place (dataclasses)
├── data_utils.py        <- Synthetic + realistic data generation
├── environment.py       <- Gymnasium environment (state, action, reward)
├── heuristics.py        <- ASAP, ALAP, Round Robin baselines
├── optimizer.py         <- LP optimal solution (perfect foresight)
├── ml_forecaster.py     <- Pillar 1: XGBoost / Random Forest
├── dl_forecaster.py     <- Pillar 2: LSTM
├── rl_agent.py          <- Pillar 3: PPO via Stable-Baselines3
├── agentic_ai.py        <- Pillar 4: OpenAI tool-calling orchestrator
│
├── autograder.py        <- Course staff only: CA evaluation script
├── requirements.txt     <- Python dependencies
└── README.md            <- This file
```

**The `.py` files are provided reference implementations.** Students read them to
understand how things work, but implement their improvements in `main.ipynb`.

---

## Student Tasks (Summary)

The notebook runs end-to-end with working baselines. These TODOs mark where to improve:

### TODO 1 (Section 4): Extend ML Feature Engineering
Write `my_engineer_features(df, cfg)`. Start from `engineer_features()` (25+ baseline
features) and add at least 3 new feature categories. Ideas: additional price lags,
interaction terms, exponentially weighted moving averages.

### TODO 2 (Section 5): Improve the LSTM
Option A: Modify `cfg.forecast` hyperparameters (hidden_size, num_layers, seq_len).
Option B: Subclass `PriceLSTM` and pass to `train_lstm(model_class=YourClass)`.

### TODO 3 (Section 6): Design a Better Reward + Tune PPO
Write `my_custom_reward(step_cost, deadline_penalty, ...)` and configure PPO
hyperparameters. The autograder always evaluates on actual charging cost and
departure compliance — not your custom reward.

### TODO 4 (Section 9): Analysis Report
Answer four analysis questions comparing strategies, documenting your design
choices and experimental results.

---

## Submission Instructions

After completing all TODOs and training your models, run **Section 10** in the
notebook to package your submission. Upload to Courseworks by the deadline.

**Required files:**
```
submission_YOUR_UNI.zip
├── ml_model.pkl          <- saved with save_ml_model()
├── lstm_model.pth        <- saved with save_lstm_model()
├── rl_agent.zip          <- saved with PPO.save()
└── student_info.json     <- {"name": "Your Name", "uni": "your_uni"}
```

---

## Grading and Extra Credit

### Standard Grading

Completion of the Analysis Report (Section 9) is the primary deliverable. Running
the notebook end-to-end and answering all four analysis questions thoroughly is
required for full credit.

### Extra Credit: Model Performance Leaderboard

The course staff runs an automated evaluation on a **secret holdout dataset**.
Students whose models outperform the provided baselines earn extra credit.

**Scoring formula:**
```
ml_score   = baseline_ml_mae   / your_ml_mae        (> 1.0 means you beat baseline)
lstm_score = baseline_lstm_mae / your_lstm_mae
rl_score   = baseline_rl_cost  / your_rl_cost

combined_score = 0.30 × ml_score + 0.30 × lstm_score + 0.40 × rl_score
```

- `combined_score > 1.0` → beats all baselines → extra credit
- `combined_score ≈ 1.0` → on par with provided code
- `combined_score < 1.0` → below baseline

RL is weighted 40% (highest) because reward design + hyperparameter tuning is
the most open-ended task. Missing a pillar has its weight redistributed to
submitted pillars.

---

## Optional: Realistic Price Data

```bash
pip install gridstatus
```

```python
from data_utils import load_realistic_prices
prices_df = load_realistic_prices("CAISO", "2024-01-01", "2024-12-31")
```

Supported ISOs: CAISO, ERCOT, PJM, MISO, NYISO, ISONE, SPP.

---

## Optional: Live LLM Mode (Section 8)

Section 8 defaults to **mock mode** — no API key needed. For live GPT-4o:

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.24 | Array operations |
| pandas | ≥2.0 | Data manipulation |
| scipy | ≥1.10 | LP solver (HiGHS) |
| scikit-learn | ≥1.3 | ML models, metrics |
| matplotlib | ≥3.7 | Visualization |
| xgboost | ≥2.0 | Gradient boosting |
| torch | ≥2.0 | LSTM training |
| gymnasium | ≥0.29 | RL environment |
| stable-baselines3 | ≥2.2 | PPO implementation |
| openai | ≥1.0 | LLM API (optional) |
| gridstatus | ≥0.25 | Real price data (optional) |

---

*IEOR E4010: AI for Operations Research and Financial Engineering*
*Columbia University, Spring 2026*
