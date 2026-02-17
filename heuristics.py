"""
heuristics.py — Baseline Charging Strategies
==============================================

Simple rule-based strategies that serve as performance baselines.
Students compare their ML/DL/RL results against these to understand
what "smart" charging actually buys you.

Three baselines:
1. ASAP (As Soon As Possible): Charge every EV at max rate immediately
2. ALAP (As Late As Possible): Delay charging until the last safe moment
3. Round Robin: Rotate charging slots across EVs to respect transformer

IEOR E4010: AI for Operations Research and Financial Engineering
Columbia University, Spring 2026
"""

import numpy as np
from typing import List, Dict, Any, Optional

from config import Config, DEFAULT_CONFIG
from data_utils import EVSchedule
from environment import EVChargingEnv, make_env


def run_heuristic(
    env: EVChargingEnv,
    strategy: str = "asap",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a heuristic strategy on the environment.

    Args:
        env:      EVChargingEnv instance (will be reset)
        strategy: One of "asap", "alap", "round_robin"
        verbose:  Print step-by-step info

    Returns:
        Dict with episode results
    """
    strategy = strategy.lower()
    if strategy not in ("asap", "alap", "round_robin"):
        raise ValueError(f"Unknown strategy: {strategy}. Use 'asap', 'alap', or 'round_robin'.")

    obs, info = env.reset()
    cfg = env.cfg
    schedules = env.get_schedules()
    price_curve = env.get_price_curve()
    total_reward = 0.0

    # Pre-compute ALAP start times
    if strategy == "alap":
        alap_start = _compute_alap_starts(schedules, cfg)

    for step in range(cfg.time.total_steps):
        action = np.zeros(cfg.fleet.num_evs, dtype=np.float32)

        for i, ev in enumerate(schedules):
            if not (ev.arrival_step <= step < ev.departure_step):
                continue  # EV not connected

            current_soc = env.soc[i]

            if current_soc >= ev.target_soc:
                action[i] = 0.0  # Already at target
                continue

            if strategy == "asap":
                # Charge at max rate as soon as plugged in
                action[i] = 1.0

            elif strategy == "alap":
                # Only start charging when we must
                if step >= alap_start.get(i, ev.arrival_step):
                    action[i] = 1.0
                else:
                    action[i] = 0.0

            elif strategy == "round_robin":
                # Give each EV a fair time slice
                # Active EVs take turns in groups sized to transformer limit
                action[i] = 1.0  # Will be scaled by transformer enforcement

        # For round_robin, prioritize EVs with least time remaining
        if strategy == "round_robin":
            action = _apply_round_robin(action, step, schedules, env.soc, cfg)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if verbose and step % 8 == 0:
            hour = (cfg.time.start_hour + step * cfg.time.dt_hours) % 24
            print(f"  Step {step:3d} ({int(hour):02d}:{int((hour%1)*60):02d}) | "
                  f"Connected: {info['num_connected']:2d} | "
                  f"Avg SoC: {info['avg_soc']:.1%} | "
                  f"Power: {info['total_power_kw']:.0f} kW")

        if terminated or truncated:
            break

    return {
        "strategy": strategy,
        "total_reward": total_reward,
        "total_cost": info["episode_cost"],
        "v2g_revenue": info["episode_v2g_revenue"],
        "degradation_cost": info["episode_degradation"],
        "net_cost": info["episode_net_cost"],
        "penalties": info["episode_penalties"],
        "evs_meeting_target": info["evs_meeting_target"],
        "total_evs": info["total_evs"],
    }


def _compute_alap_starts(
    schedules: List[EVSchedule],
    cfg: Config,
) -> Dict[int, int]:
    """Compute the latest time step each EV can start charging
    and still reach target SoC by departure.

    Returns:
        Dict mapping ev_id -> latest_start_step
    """
    starts = {}
    for ev in schedules:
        energy_needed = (ev.target_soc - ev.arrival_soc) * ev.battery_kwh
        hours_needed = energy_needed / (ev.max_charge_kw * cfg.battery.charge_efficiency)
        steps_needed = int(np.ceil(hours_needed / cfg.time.dt_hours))
        # Add 1-step safety margin
        latest_start = ev.departure_step - steps_needed - 1
        starts[ev.ev_id] = max(latest_start, ev.arrival_step)
    return starts


def _apply_round_robin(
    action: np.ndarray,
    step: int,
    schedules: List[EVSchedule],
    soc: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """Apply round-robin scheduling that respects transformer limit.

    Priority: EVs with least time until departure go first.
    """
    max_simultaneous = int(cfg.grid.transformer_limit_kw / cfg.battery.max_charge_rate_kw)

    # Find connected EVs that need charging
    candidates = []
    for i, ev in enumerate(schedules):
        if ev.arrival_step <= step < ev.departure_step and soc[i] < ev.target_soc:
            time_left = ev.departure_step - step
            urgency = (ev.target_soc - soc[i]) / max(time_left * cfg.time.dt_hours, 0.01)
            candidates.append((i, urgency))

    # Sort by urgency (most urgent first)
    candidates.sort(key=lambda x: -x[1])

    # Zero out all, then enable top-priority EVs
    result = np.zeros_like(action)
    for rank, (ev_idx, _) in enumerate(candidates):
        if rank < max_simultaneous:
            result[ev_idx] = 1.0

    return result


def run_all_heuristics(
    cfg: Config = DEFAULT_CONFIG,
    schedules: Optional[List[EVSchedule]] = None,
    price_curve: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Run all heuristic strategies and return results.

    Args:
        cfg:         Configuration
        schedules:   Pre-generated schedules (same for all strategies)
        price_curve: Pre-generated price curve (same for all strategies)
        verbose:     Print summary table

    Returns:
        List of result dicts, one per strategy
    """
    results = []

    for strategy in ["asap", "alap", "round_robin"]:
        env = make_env(cfg, schedules=schedules, price_curve=price_curve)
        result = run_heuristic(env, strategy=strategy)
        results.append(result)

        if verbose:
            print(f"\n{'='*50}")
            print(f"Strategy: {strategy.upper()}")
            print(f"  Net cost:          ${result['net_cost']:.2f}")
            print(f"  Charging cost:     ${result['total_cost']:.2f}")
            print(f"  V2G revenue:       ${result['v2g_revenue']:.2f}")
            print(f"  Degradation:       ${result['degradation_cost']:.2f}")
            print(f"  Penalties:         ${result['penalties']:.2f}")
            print(f"  EVs at target:     {result['evs_meeting_target']}/{result['total_evs']}")

    return results


if __name__ == "__main__":
    from data_utils import generate_synthetic_prices, get_daily_price_curve, generate_ev_schedules

    cfg = Config()
    prices_df = generate_synthetic_prices(cfg, num_days=30)
    price_curve = get_daily_price_curve(prices_df, day_index=0, cfg=cfg)
    schedules = generate_ev_schedules(cfg)

    results = run_all_heuristics(cfg, schedules=schedules, price_curve=price_curve)
