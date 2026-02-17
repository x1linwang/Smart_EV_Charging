"""
config.py — Centralized Configuration for EV Smart Charging Project
=====================================================================

All simulation parameters in one place. Students can modify these
to explore different scenarios without touching the core logic.

IEOR E4010: AI for Operations Research and Financial Engineering
Columbia University, Spring 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# Fleet & Battery Parameters
# ============================================================
@dataclass
class BatteryConfig:
    """Physical parameters for EV batteries."""
    capacity_kwh: float = 60.0          # Battery capacity (kWh)
    max_charge_rate_kw: float = 11.0    # Max Level 2 AC charging (kW)
    max_discharge_rate_kw: float = 11.0 # Max V2G discharge rate (kW)
    charge_efficiency: float = 0.95     # Charging efficiency (η)
    discharge_efficiency: float = 0.95  # Discharging efficiency (η)
    soc_min: float = 0.10              # Minimum SoC (10%)
    soc_max: float = 1.00              # Maximum SoC (100%)
    degradation_cost_per_kwh: float = 0.04  # Battery wear cost for V2G ($/kWh)


@dataclass
class FleetConfig:
    """Fleet composition and schedule parameters."""
    num_evs: int = 20                   # Number of EVs in the depot
    target_soc: float = 0.90           # Required SoC at departure (90%)

    # Arrival window (hours of day, 24h format)
    arrival_start_hour: float = 14.0    # Earliest arrival: 2 PM
    arrival_end_hour: float = 20.0      # Latest arrival: 8 PM

    # Departure window (hours of day, 24h format)
    departure_start_hour: float = 5.0   # Earliest departure: 5 AM
    departure_end_hour: float = 9.0     # Latest departure: 9 AM

    # Arrival SoC distribution
    arrival_soc_min: float = 0.20       # Minimum arrival SoC (20%)
    arrival_soc_max: float = 0.60       # Maximum arrival SoC (60%)
    arrival_soc_mean: float = 0.35      # Mean arrival SoC
    arrival_soc_std: float = 0.10       # Std dev of arrival SoC


# ============================================================
# Grid & Transformer Parameters
# ============================================================
@dataclass
class GridConfig:
    """Electrical grid and transformer constraints."""
    transformer_limit_kw: float = 150.0  # Max aggregate power (kW)
    # With 20 EVs × 11 kW = 220 kW theoretical max,
    # the 150 kW limit forces coordination (ratio ≈ 0.68)


# ============================================================
# Time Parameters
# ============================================================
@dataclass
class TimeConfig:
    """Simulation time discretization."""
    time_step_minutes: int = 15         # Decision interval (minutes)
    steps_per_hour: int = 4             # 60 / 15 = 4
    simulation_hours: int = 24          # One full day
    total_steps: int = 96               # 24 × 4 = 96

    # Simulation start time (hour of day)
    start_hour: float = 12.0            # Noon — captures afternoon arrivals
    # through next-morning departures

    @property
    def dt_hours(self) -> float:
        """Time step duration in hours."""
        return self.time_step_minutes / 60.0


# ============================================================
# Electricity Price Parameters
# ============================================================
@dataclass
class PriceConfig:
    """Electricity price model parameters.

    Default values are calibrated to resemble CAISO (California)
    day-ahead market patterns:
    - Low overnight:   ~$15-25/MWh  (off-peak)
    - Midday dip:      ~$10-20/MWh  (solar surplus, "duck curve")
    - Evening peak:    ~$50-100/MWh (6-9 PM ramp)

    Prices are in $/MWh. To convert to $/kWh, divide by 1000.
    """
    # Base price components ($/MWh)
    base_price: float = 30.0            # Average base price
    overnight_discount: float = 15.0    # How much cheaper overnight is
    solar_dip: float = 10.0             # Midday solar suppression
    evening_peak: float = 45.0          # Evening peak premium
    noise_std: float = 5.0              # Random noise std dev

    # Historical data parameters (for ML/DL training)
    num_days_history: int = 365         # Days of historical data to generate
    train_ratio: float = 0.8            # Train/test split ratio

    # Peak hours (24h format)
    peak_start: float = 17.0            # 5 PM
    peak_end: float = 21.0              # 9 PM
    solar_peak_start: float = 10.0      # 10 AM
    solar_peak_end: float = 15.0        # 3 PM


# ============================================================
# RL Training Parameters
# ============================================================
@dataclass
class RLConfig:
    """Reinforcement learning training configuration."""
    algorithm: str = "PPO"              # RL algorithm
    total_timesteps: int = 200_000      # Training steps
    learning_rate: float = 3e-4         # Learning rate
    n_steps: int = 2048                 # Steps per update
    batch_size: int = 64                # Mini-batch size
    n_epochs: int = 10                  # Epochs per update
    gamma: float = 0.99                 # Discount factor
    gae_lambda: float = 0.95           # GAE lambda
    clip_range: float = 0.2            # PPO clip range
    ent_coef: float = 0.01             # Entropy coefficient

    # Reward weights (students will tune these)
    reward_price_weight: float = 1.0       # Weight on electricity cost
    reward_deadline_penalty: float = 100.0 # Penalty per EV missing target
    reward_overload_penalty: float = 50.0  # Penalty for exceeding transformer
    reward_degradation_weight: float = 1.0 # Weight on degradation cost

    # Training settings
    n_eval_episodes: int = 10           # Episodes for evaluation
    eval_freq: int = 10_000             # Evaluate every N steps
    seed: int = 42                      # Random seed


# ============================================================
# Forecasting Parameters
# ============================================================
@dataclass
class ForecastConfig:
    """ML/DL forecasting model configuration."""
    # ML (XGBoost) parameters
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1

    # DL (LSTM) parameters
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_sequence_length: int = 168     # 1 week of hourly data
    lstm_epochs: int = 50
    lstm_batch_size: int = 32
    lstm_learning_rate: float = 1e-3


# ============================================================
# Agentic AI Parameters
# ============================================================
@dataclass
class AgentConfig:
    """LLM orchestrator configuration."""
    provider: str = "openai"            # LLM provider: "openai" or "mock"
    model: str = "gpt-4o"              # Model name
    temperature: float = 0.3            # Low temperature for tool-calling
    max_tokens: int = 1024              # Max response tokens
    api_key_env_var: str = "OPENAI_API_KEY"  # Env var for API key


# ============================================================
# Master Configuration
# ============================================================
@dataclass
class Config:
    """Master configuration combining all sub-configs.

    Usage:
        cfg = Config()                   # All defaults
        cfg.fleet.num_evs = 30           # Override specific values
        cfg.price.evening_peak = 60.0
    """
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    fleet: FleetConfig = field(default_factory=FleetConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    price: PriceConfig = field(default_factory=PriceConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    seed: int = 42                      # Global random seed

    def __post_init__(self):
        """Validate configuration consistency."""
        # Transformer must accommodate at least one EV
        assert self.grid.transformer_limit_kw >= self.battery.max_charge_rate_kw, \
            "Transformer limit must be >= max charge rate of one EV"
        # Target SoC must be within battery bounds
        assert self.battery.soc_min < self.fleet.target_soc <= self.battery.soc_max, \
            "Target SoC must be within [soc_min, soc_max]"
        # Time consistency
        assert self.time.total_steps == self.time.simulation_hours * self.time.steps_per_hour

    def summary(self) -> str:
        """Print a human-readable summary of key parameters."""
        max_simultaneous = self.grid.transformer_limit_kw / self.battery.max_charge_rate_kw
        utilization = (self.fleet.num_evs * self.battery.max_charge_rate_kw
                       / self.grid.transformer_limit_kw)
        lines = [
            "=" * 60,
            "EV Smart Charging Configuration Summary",
            "=" * 60,
            f"Fleet:        {self.fleet.num_evs} EVs × {self.battery.capacity_kwh} kWh",
            f"Charge rate:  ±{self.battery.max_charge_rate_kw} kW (V2G enabled)",
            f"Transformer:  {self.grid.transformer_limit_kw} kW limit",
            f"  → Max simultaneous full-rate: {max_simultaneous:.1f} EVs",
            f"  → Over-subscription ratio:    {utilization:.2f}x",
            f"Time:         {self.time.total_steps} steps × {self.time.time_step_minutes} min "
            f"= {self.time.simulation_hours}h",
            f"Target SoC:   ≥{self.fleet.target_soc*100:.0f}% at departure",
            f"Arrival SoC:  {self.fleet.arrival_soc_min*100:.0f}%–"
            f"{self.fleet.arrival_soc_max*100:.0f}% "
            f"(μ={self.fleet.arrival_soc_mean*100:.0f}%)",
            f"Degradation:  ${self.battery.degradation_cost_per_kwh:.2f}/kWh discharged",
            f"Price range:  ~${self.price.base_price - self.price.overnight_discount:.0f}–"
            f"${self.price.base_price + self.price.evening_peak:.0f}/MWh",
            "=" * 60,
        ]
        return "\n".join(lines)


# Convenience: default config instance
DEFAULT_CONFIG = Config()


if __name__ == "__main__":
    cfg = Config()
    print(cfg.summary())
