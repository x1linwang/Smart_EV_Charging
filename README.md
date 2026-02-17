# 🔌 Large-Scale EV Smart Charging: A Four-Pillar AI Project

**IEOR E4010 — Artificial Intelligence for Operations Research and Financial Engineering**
Columbia University · Spring 2026 · Final Project

---

## Overview

You are the fleet operations manager at a delivery company depot.
Every evening, 20 electric vans return and must be charged to ≥90% by their morning departure — but a shared 150 kW transformer means you can't charge them all at once.

Your job: **minimize electricity cost** while meeting every vehicle's deadline.

This project integrates four AI pillars into one working system:

| Pillar | Module | What It Does |
|--------|--------|-------------|
| **Machine Learning** | `ml_forecaster.py` | XGBoost forecasts next-hour electricity prices from tabular features |
| **Deep Learning** | `dl_forecaster.py` | LSTM learns price patterns from raw sequential data |
| **Reinforcement Learning** | `rl_agent.py` | PPO agent learns a real-time charging policy through simulation |
| **Agentic AI** | `agentic_ai.py` | GPT-4o orchestrator with tool-calling ties everything together |

---

## Quick Start

### 1. Environment Setup

```bash
# Create a virtual environment (recommended)
python -m venv ev_env
source ev_env/bin/activate        # Linux/Mac
# ev_env\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

**Option A — Jupyter Notebook (recommended):**
```bash
jupyter notebook main.ipynb
```
Run cells top-to-bottom. The notebook guides you through every step.

**Option B — Individual Modules:**
```bash
# Test data generation
python data_utils.py

# Run heuristic baselines
python heuristics.py

# Solve LP optimal
python optimizer.py

# Train ML forecaster
python ml_forecaster.py

# Train LSTM forecaster
python dl_forecaster.py

# Train RL agent (takes ~5-10 min)
python rl_agent.py

# Run agentic AI demo (works in mock mode without API key)
python agentic_ai.py
```

### 3. Enable Live LLM Mode (Optional)

The agentic AI module defaults to **mock mode** (no API key needed).
To enable live GPT-4o responses:

```bash
export OPENAI_API_KEY="sk-..."
python agentic_ai.py
```

Or in Python / Jupyter:
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

### 4. Use Realistic Price Data (Optional)

By default, the project uses synthetic prices calibrated to CAISO patterns.
To use real wholesale market data:

```bash
pip install gridstatus
```

Then in your code or notebook:
```python
from data_utils import load_realistic_prices
prices_df = load_realistic_prices("CAISO", "2024-01-01", "2024-12-31")
```

Supported ISOs: CAISO, ERCOT, PJM, MISO, NYISO, ISONE, SPP.

---

## Project Structure

```
ev_smart_charging/
├── config.py            # All parameters in one place (dataclasses)
├── data_utils.py        # Synthetic + realistic data generation
├── environment.py       # Gymnasium environment for the charging problem
├── heuristics.py        # ASAP, ALAP, Round Robin baselines
├── optimizer.py         # LP optimal solution (perfect foresight)
├── ml_forecaster.py     # Pillar 1: XGBoost / Random Forest
├── dl_forecaster.py     # Pillar 2: LSTM
├── rl_agent.py          # Pillar 3: PPO via Stable-Baselines3
├── agentic_ai.py        # Pillar 4: OpenAI tool-calling orchestrator
├── main.ipynb           # Master notebook (run this)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Problem Specification

| Parameter | Value | Why |
|-----------|-------|-----|
| Fleet size | 20 EVs | Large enough for coupling constraints to matter |
| Battery capacity | 60 kWh | Typical commercial van (e.g., Ford E-Transit) |
| Charge/discharge rate | ±11 kW | Level 2 AC, bidirectional for V2G |
| Transformer limit | 150 kW | 20×11=220 kW demand vs. 150 kW supply → must coordinate |
| Target departure SoC | ≥90% | Hard constraint — missed deadlines are penalized |
| Arrival SoC | 20–60% (μ=35%) | Stochastic — varies by vehicle each day |
| Arrivals | 2 PM – 8 PM | Afternoon return from delivery routes |
| Departures | 5 AM – 9 AM | Morning dispatch for next day's deliveries |
| Time resolution | 15 minutes | 96 steps per 24-hour day |
| V2G degradation | $0.04/kWh | Battery wear cost for discharging |

---

## Student Tasks (★ TODOs)

The code is **complete and runnable** as provided. These TODOs mark where students can improve upon the working baselines:

### TODO 1: ML Feature Engineering (`ml_forecaster.py`)
**Function:** `engineer_features()`

Improve the feature pipeline for XGBoost price forecasting. Ideas:
- Additional lag features (t-48, t-72)
- Interaction features (hour × is_weekend)
- Fourier encoding of cyclical time
- Exponentially weighted moving averages

### TODO 2: LSTM Architecture & Training (`dl_forecaster.py`)
**Class:** `PriceLSTM` and **Function:** `train_lstm()`

Improve the deep learning forecaster. Ideas:
- Experiment with hidden_size (32, 64, 128, 256)
- Try GRU instead of LSTM
- Add attention mechanism
- Use multi-feature input (price + hour + dow)

### TODO 3: RL Reward Function Design (`environment.py`)
**Location:** `step()` method, Section 8

Design a better reward function for the PPO agent. Ideas:
- Change penalty magnitudes
- Add reward for smooth power profiles
- Reward V2G revenue explicitly
- Lexicographic priorities instead of weighted sum

### TODO 4: Analysis Report (`main.ipynb`)
**Location:** Section 9 of the notebook

Answer the analysis questions comparing all approaches:
- ML vs. DL forecasting performance
- RL vs. LP: the "cost of uncertainty"
- Overall strategy ranking and deployment recommendation

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

## Troubleshooting

**`ModuleNotFoundError: No module named 'xgboost'`**
→ `pip install xgboost`

**`ModuleNotFoundError: No module named 'stable_baselines3'`**
→ `pip install stable-baselines3`

**RL training is slow**
→ Reduce `cfg.rl.total_timesteps` to 50,000 for quick tests. Use 200,000+ for good policies.

**LSTM not converging**
→ Reduce `cfg.forecast.lstm_sequence_length` to 48 (2 days). Check that prices are not all identical.

**`openai.AuthenticationError`**
→ Set `OPENAI_API_KEY` env variable, or use mock mode (default).
