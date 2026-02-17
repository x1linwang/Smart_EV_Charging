"""
optimizer.py — LP/MILP Optimal Charging Schedule
==================================================

Solves the fleet charging problem to global optimality using linear
programming. This requires PERFECT FORESIGHT — it knows all future
prices, arrivals, departures, and SoC values in advance.

This is the gold-standard baseline. No learning-based method can beat
it because it has information no real-time controller would have.
The gap between LP optimal and RL tells students how much "the cost
of uncertainty" is.

Uses scipy.optimize.linprog for the LP formulation.

IEOR E4010: AI for Operations Research and Financial Engineering
Columbia University, Spring 2026
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from typing import List, Dict, Any, Optional, Tuple

from config import Config, DEFAULT_CONFIG
from data_utils import EVSchedule


def solve_optimal_schedule(
    schedules: List[EVSchedule],
    price_curve: np.ndarray,
    cfg: Config = DEFAULT_CONFIG,
    allow_v2g: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Solve the optimal charging schedule via Linear Programming.

    Decision Variables:
        p_charge[i, t]    : charging power for EV i at step t (kW), ≥ 0
        p_discharge[i, t] : discharging power for EV i at step t (kW), ≥ 0

    Objective: Minimize total cost
        sum_t { price[t] * dt * sum_i (p_charge[i,t] - p_discharge[i,t]) / 1000
                + degradation * dt * sum_i p_discharge[i,t] }

    Subject to:
        1. Power bounds:     0 ≤ p_charge[i,t] ≤ max_charge * connected[i,t]
                             0 ≤ p_discharge[i,t] ≤ max_discharge * connected[i,t]
        2. SoC dynamics:     soc[i,t+1] = soc[i,t] + η_c * p_charge[i,t] * dt / cap
                                                    - p_discharge[i,t] * dt / (η_d * cap)
        3. SoC bounds:       soc_min ≤ soc[i,t] ≤ soc_max
        4. Departure target: soc[i, departure_step] ≥ target_soc
        5. Transformer:      sum_i p_charge[i,t] ≤ transformer_limit  (for each t)
                             sum_i p_discharge[i,t] ≤ transformer_limit  (for each t)

    Args:
        schedules:   List of EVSchedule objects
        price_curve: Price curve in $/MWh, shape (total_steps,)
        cfg:         Configuration
        allow_v2g:   Whether to allow V2G discharging
        verbose:     Print progress and results

    Returns:
        Dict with:
        - schedule_charge:    np.ndarray (num_evs, total_steps) charge power kW
        - schedule_discharge: np.ndarray (num_evs, total_steps) discharge power kW
        - schedule_net:       np.ndarray (num_evs, total_steps) net power kW
        - soc_trajectory:     np.ndarray (num_evs, total_steps+1) SoC values
        - total_cost:         float, electricity cost ($)
        - v2g_revenue:        float, V2G revenue ($)
        - degradation_cost:   float, battery degradation ($)
        - net_cost:           float, total net cost ($)
        - evs_meeting_target: int
        - status:             str, solver status
    """
    N = len(schedules)      # Number of EVs
    T = cfg.time.total_steps  # Number of time steps
    dt = cfg.time.dt_hours    # Time step in hours

    if verbose:
        print(f"Solving LP: {N} EVs × {T} steps = {2*N*T} decision variables...")

    # ------------------------------------------------------------------
    # Variable layout:
    #   x = [p_charge(0,0), ..., p_charge(0,T-1),   # EV 0 charge
    #        p_charge(1,0), ..., p_charge(1,T-1),   # EV 1 charge
    #        ...
    #        p_charge(N-1,0), ..., p_charge(N-1,T-1),
    #        p_discharge(0,0), ..., p_discharge(N-1,T-1)]
    #
    #   Total: 2 * N * T variables
    # ------------------------------------------------------------------
    num_vars = 2 * N * T

    def idx_charge(i, t):
        return i * T + t

    def idx_discharge(i, t):
        return N * T + i * T + t

    # ------------------------------------------------------------------
    # Objective: minimize cost
    # ------------------------------------------------------------------
    c = np.zeros(num_vars)
    for i in range(N):
        for t in range(T):
            price_per_kwh = price_curve[t] / 1000.0  # $/MWh → $/kWh

            # Charging cost: price * power * dt
            c[idx_charge(i, t)] = price_per_kwh * dt

            # Discharging revenue (negative cost) + degradation
            if allow_v2g:
                c[idx_discharge(i, t)] = (
                    -price_per_kwh * dt  # Revenue from selling
                    + cfg.battery.degradation_cost_per_kwh * dt  # Degradation cost
                )

    # ------------------------------------------------------------------
    # Bounds: 0 ≤ p ≤ max_rate (only when connected)
    # ------------------------------------------------------------------
    bounds = []
    for i, ev in enumerate(schedules):
        for t in range(T):
            connected = ev.arrival_step <= t < ev.departure_step
            if connected:
                bounds.append((0, ev.max_charge_kw))
            else:
                bounds.append((0, 0))

    for i, ev in enumerate(schedules):
        for t in range(T):
            connected = ev.arrival_step <= t < ev.departure_step
            if connected and allow_v2g:
                bounds.append((0, ev.max_discharge_kw))
            else:
                bounds.append((0, 0))

    # ------------------------------------------------------------------
    # Inequality constraints: A_ub @ x ≤ b_ub
    # ------------------------------------------------------------------
    A_ub_rows = []
    b_ub_vals = []

    # Constraint: Transformer limit (charging) at each time step
    #   sum_i p_charge[i,t] ≤ transformer_limit
    for t in range(T):
        row = np.zeros(num_vars)
        for i in range(N):
            row[idx_charge(i, t)] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(cfg.grid.transformer_limit_kw)

    # Constraint: Transformer limit (discharging) at each time step
    if allow_v2g:
        for t in range(T):
            row = np.zeros(num_vars)
            for i in range(N):
                row[idx_discharge(i, t)] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(cfg.grid.transformer_limit_kw)

    # SoC constraints: soc_min ≤ soc[i,t] ≤ soc_max for all connected steps
    # and soc[i, departure] ≥ target_soc
    #
    # SoC at time t:
    #   soc[i,t] = arrival_soc[i] + (η_c/cap) * sum_{s<t} p_charge[i,s] * dt
    #                              - (1/(η_d*cap)) * sum_{s<t} p_discharge[i,s] * dt
    #
    # Upper bound: soc[i,t] ≤ soc_max
    #   → (η_c/cap)*dt * sum p_charge - (1/(η_d*cap))*dt * sum p_discharge ≤ soc_max - arrival_soc
    #
    # Lower bound: soc[i,t] ≥ soc_min
    #   → -(η_c/cap)*dt * sum p_charge + (1/(η_d*cap))*dt * sum p_discharge ≤ arrival_soc - soc_min

    eta_c = cfg.battery.charge_efficiency
    eta_d = cfg.battery.discharge_efficiency

    for i, ev in enumerate(schedules):
        cap = ev.battery_kwh
        charge_coeff = eta_c * dt / cap
        discharge_coeff = dt / (eta_d * cap)

        for t in range(ev.arrival_step + 1, min(ev.departure_step + 1, T + 1)):
            # SoC upper bound: soc[i,t] ≤ soc_max
            row = np.zeros(num_vars)
            for s in range(ev.arrival_step, t):
                row[idx_charge(i, s)] = charge_coeff
                row[idx_discharge(i, s)] = -discharge_coeff
            A_ub_rows.append(row)
            b_ub_vals.append(cfg.battery.soc_max - ev.arrival_soc)

            # SoC lower bound: soc[i,t] ≥ soc_min → -soc[i,t] ≤ -soc_min
            row2 = np.zeros(num_vars)
            for s in range(ev.arrival_step, t):
                row2[idx_charge(i, s)] = -charge_coeff
                row2[idx_discharge(i, s)] = discharge_coeff
            A_ub_rows.append(row2)
            b_ub_vals.append(ev.arrival_soc - cfg.battery.soc_min)

        # Departure target: soc[i, departure] ≥ target_soc
        dep = min(ev.departure_step, T)
        row_dep = np.zeros(num_vars)
        for s in range(ev.arrival_step, dep):
            row_dep[idx_charge(i, s)] = -charge_coeff
            row_dep[idx_discharge(i, s)] = discharge_coeff
        A_ub_rows.append(row_dep)
        b_ub_vals.append(ev.arrival_soc - ev.target_soc)

    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_vals) if b_ub_vals else None

    if verbose:
        n_constraints = len(A_ub_rows) if A_ub_rows else 0
        print(f"  Variables: {num_vars}, Constraints: {n_constraints}")

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
        method="highs",
        options={"disp": False, "time_limit": 60},
    )

    if result.status != 0:
        if verbose:
            print(f"  WARNING: Solver status {result.status}: {result.message}")
        # Try with relaxed constraints
        result_relaxed = linprog(
            c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
            method="highs",
            options={"disp": False, "time_limit": 120, "presolve": True},
        )
        if result_relaxed.status == 0:
            result = result_relaxed

    # ------------------------------------------------------------------
    # Extract solution
    # ------------------------------------------------------------------
    x = result.x if result.x is not None else np.zeros(num_vars)

    schedule_charge = np.zeros((N, T))
    schedule_discharge = np.zeros((N, T))
    for i in range(N):
        for t in range(T):
            schedule_charge[i, t] = max(0, x[idx_charge(i, t)])
            schedule_discharge[i, t] = max(0, x[idx_discharge(i, t)])

    schedule_net = schedule_charge - schedule_discharge

    # Reconstruct SoC trajectories
    soc_trajectory = np.zeros((N, T + 1))
    for i, ev in enumerate(schedules):
        soc_trajectory[i, 0] = ev.arrival_soc
        for t in range(T):
            soc_trajectory[i, t + 1] = soc_trajectory[i, t]
            if ev.arrival_step <= t < ev.departure_step:
                soc_trajectory[i, t + 1] += (
                    eta_c * schedule_charge[i, t] * dt / ev.battery_kwh
                    - schedule_discharge[i, t] * dt / (eta_d * ev.battery_kwh)
                )
                soc_trajectory[i, t + 1] = np.clip(
                    soc_trajectory[i, t + 1], cfg.battery.soc_min, cfg.battery.soc_max
                )

    # Compute costs
    total_cost = 0.0
    v2g_revenue = 0.0
    degradation_cost = 0.0
    for t in range(T):
        price_per_kwh = price_curve[t] / 1000.0
        for i in range(N):
            total_cost += schedule_charge[i, t] * dt * price_per_kwh
            v2g_revenue += schedule_discharge[i, t] * dt * price_per_kwh
            degradation_cost += (schedule_discharge[i, t] * dt
                                 * cfg.battery.degradation_cost_per_kwh)

    net_cost = total_cost - v2g_revenue + degradation_cost

    # Check how many EVs meet target
    evs_meeting = 0
    for i, ev in enumerate(schedules):
        dep = min(ev.departure_step, T)
        if soc_trajectory[i, dep] >= ev.target_soc - 0.01:
            evs_meeting += 1

    # Max power draw
    max_power = np.max(np.sum(schedule_charge, axis=0))

    if verbose:
        print(f"  Status:       {result.message}")
        print(f"  Net cost:     ${net_cost:.2f}")
        print(f"  Charge cost:  ${total_cost:.2f}")
        print(f"  V2G revenue:  ${v2g_revenue:.2f}")
        print(f"  Degradation:  ${degradation_cost:.2f}")
        print(f"  Max power:    {max_power:.1f} / {cfg.grid.transformer_limit_kw:.0f} kW")
        print(f"  EVs at target: {evs_meeting}/{N}")

    return {
        "schedule_charge": schedule_charge,
        "schedule_discharge": schedule_discharge,
        "schedule_net": schedule_net,
        "soc_trajectory": soc_trajectory,
        "total_cost": total_cost,
        "v2g_revenue": v2g_revenue,
        "degradation_cost": degradation_cost,
        "net_cost": net_cost,
        "max_power_kw": max_power,
        "evs_meeting_target": evs_meeting,
        "total_evs": N,
        "status": result.message,
        "strategy": "lp_optimal",
    }


def run_lp_on_env(
    env: EVChargingEnv,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Solve LP optimal and replay the schedule through the environment.

    This ensures the LP results are directly comparable with RL/heuristic
    results (same cost accounting, same constraint checks).

    Args:
        env:     EVChargingEnv instance (will be reset)
        verbose: Print progress

    Returns:
        Dict with LP results + environment episode metrics
    """
    obs, info = env.reset()
    cfg = env.cfg
    schedules = env.get_schedules()
    price_curve = env.get_price_curve()

    # Solve LP
    lp_result = solve_optimal_schedule(
        schedules, price_curve, cfg, allow_v2g=True, verbose=verbose
    )

    # Replay through environment
    obs, info = env.reset()
    total_reward = 0.0

    for t in range(cfg.time.total_steps):
        # Convert LP schedule to [-1, +1] actions
        action = np.zeros(cfg.fleet.num_evs, dtype=np.float32)
        for i, ev in enumerate(schedules):
            charge = lp_result["schedule_charge"][i, t]
            discharge = lp_result["schedule_discharge"][i, t]
            if charge > 0.01:
                action[i] = charge / ev.max_charge_kw
            elif discharge > 0.01:
                action[i] = -discharge / ev.max_discharge_kw

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    lp_result.update({
        "total_reward": total_reward,
        "env_net_cost": info["episode_net_cost"],
        "env_evs_meeting_target": info["evs_meeting_target"],
    })

    return lp_result


def plot_optimal_schedule(
    lp_result: Dict[str, Any],
    price_curve: np.ndarray,
    cfg: Config = DEFAULT_CONFIG,
):
    """Visualize the LP optimal solution."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    T = cfg.time.total_steps
    steps = np.arange(T)
    N = lp_result["schedule_net"].shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"LP Optimal Schedule — Net Cost: ${lp_result['net_cost']:.2f}", fontsize=14)

    # Panel 1: Power heatmap
    ax1 = axes[0, 0]
    cmap = mcolors.LinearSegmentedColormap.from_list("cv", ["#EF4444", "#FFFFFF", "#3B82F6"])
    im = ax1.imshow(lp_result["schedule_net"], aspect="auto", cmap=cmap,
                    vmin=-cfg.battery.max_discharge_rate_kw,
                    vmax=cfg.battery.max_charge_rate_kw, interpolation="nearest")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("EV ID")
    ax1.set_title("Optimal Power Allocation (kW)")
    plt.colorbar(im, ax=ax1, label="Power (kW)")

    # Panel 2: Aggregate power
    ax2 = axes[0, 1]
    total_charge = np.sum(np.maximum(lp_result["schedule_net"], 0), axis=0)
    total_discharge = np.abs(np.sum(np.minimum(lp_result["schedule_net"], 0), axis=0))
    ax2.fill_between(steps, 0, total_charge, alpha=0.6, color="#3B82F6", label="Charging")
    ax2.fill_between(steps, 0, -total_discharge, alpha=0.6, color="#EF4444", label="Discharging")
    ax2.axhline(y=cfg.grid.transformer_limit_kw, color="red", linestyle="--",
                linewidth=2, label=f"Limit ({cfg.grid.transformer_limit_kw} kW)")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Power (kW)")
    ax2.set_title("Aggregate Power vs. Transformer Limit")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Price with charging overlay
    ax3 = axes[1, 0]
    ax3.plot(steps, price_curve[:T], "k-", linewidth=1.5, label="Price")
    ax3.fill_between(steps, 0, price_curve[:T], alpha=0.1, color="orange")
    ax3b = ax3.twinx()
    ax3b.fill_between(steps, 0, total_charge, alpha=0.3, color="#3B82F6")
    ax3b.fill_between(steps, 0, -total_discharge, alpha=0.3, color="#EF4444")
    ax3b.set_ylabel("Power (kW)", color="blue")
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Price ($/MWh)")
    ax3.set_title("Optimal Charging Follows Price (charge low, sell high)")
    ax3.grid(True, alpha=0.3)

    # Panel 4: SoC trajectories
    ax4 = axes[1, 1]
    soc = lp_result["soc_trajectory"]
    for i in range(min(N, 20)):
        ax4.plot(np.arange(T + 1), soc[i, :], linewidth=0.8, alpha=0.7,
                label=f"EV{i}" if i < 5 else None)
    ax4.axhline(y=cfg.fleet.target_soc, color="green", linestyle="--",
                linewidth=1.5, label=f"Target ({cfg.fleet.target_soc:.0%})")
    ax4.set_xlabel("Time Step")
    ax4.set_ylabel("SoC")
    ax4.set_title("Optimal SoC Trajectories")
    ax4.set_ylim(0, 1.05)
    ax4.legend(fontsize=7, ncol=2)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("lp_optimal_results.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    from data_utils import generate_synthetic_prices, get_daily_price_curve, generate_ev_schedules

    cfg = Config()
    prices_df = generate_synthetic_prices(cfg, num_days=30)
    price_curve = get_daily_price_curve(prices_df, day_index=0, cfg=cfg)
    schedules = generate_ev_schedules(cfg)

    result = solve_optimal_schedule(schedules, price_curve, cfg, verbose=True)

    try:
        plot_optimal_schedule(result, price_curve, cfg)
    except ImportError:
        print("matplotlib not available — skipping plots")
