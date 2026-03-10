"""
environment.py — Gymnasium Environment for EV Fleet V2G Charging
=================================================================

A Gymnasium-compatible environment that simulates the depot overnight
charging problem. The agent controls charging/discharging power for
each EV at every 15-minute time step, subject to physical and operational
constraints.

ENVIRONMENT OVERVIEW
--------------------
The environment wraps a 24-hour charging scenario (96 × 15-minute steps)
into the standard gym.Env interface. Each episode represents one night
at the fleet depot. The RL agent interacts via:
    obs, info = env.reset()
    action = agent.predict(obs)
    obs, reward, done, _, info = env.step(action)

STATE SPACE (63-dimensional continuous vector)
----------------------------------------------
For each of the 20 EVs (60 values total):
  [3i+0] current_soc       — current battery level [0.0, 1.0]
  [3i+1] time_to_depart    — fraction of plug-in window remaining [0, 1]
                              (1 = just arrived, 0 = about to depart)
  [3i+2] is_connected      — 1 if EV is currently plugged in, else 0

Global state (3 values):
  [60]   norm_time          — fraction of the day elapsed [0, 1]
  [61]   norm_price         — current electricity price / max observed price
  [62]   load_fraction      — previous step's total power / transformer limit

IMPORTANT: The agent sees the CURRENT price but NOT future prices. This
is what distinguishes RL from the LP optimizer (which has perfect foresight).
The agent must learn to anticipate future high prices from the time-of-day
signal (norm_time) alone.

ACTION SPACE (20-dimensional continuous vector)
-----------------------------------------------
One action per EV, clipped to [-1, +1]:
  +1.0 → charge at full rate (max_charge_kw, e.g., 11 kW)
   0.0 → idle (no power exchange)
  -1.0 → discharge at full rate (max_discharge_kw = 11 kW, V2G mode)
  Values between -1 and +1 scale the power linearly.
  Actions for disconnected EVs are automatically zeroed out.

TRANSFORMER CONSTRAINT ENFORCEMENT
-----------------------------------
If the sum of all charging powers exceeds the 150 kW transformer limit,
all charging actions are *proportionally scaled down* to exactly hit the
limit. This is a soft projection, not a hard action clipping — meaning the
agent can request more power than available and the environment handles it.
The same applies to total V2G discharge.

REWARD FUNCTION (baseline)
---------------------------
  reward = -price_weight × step_cost - deadline_penalty - overload_penalty

  where step_cost = charging_cost - v2g_revenue + degradation_cost

  Penalty triggers:
  - deadline_penalty: fires at the moment an EV's departure_step is reached
    if its SoC < target_soc. Magnitude = penalty_weight × shortfall fraction.
  - overload_penalty: fires if total power exceeds transformer limit after
    the proportional scaling (can happen due to floating-point rounding).

  A custom reward function can be passed to make_env() / EVChargingEnv()
  to override this behavior. See the custom_reward_fn parameter.

EPISODE TERMINATION
-------------------
An episode ends (terminated=True) after exactly 96 steps (one full day).
No early termination. The final deadline check fires at step 96 for any
EVs whose scheduled departure is at or after step 96.

IEOR E4010: AI for Operations Research and Financial Engineering
Columbia University, Spring 2026
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, List, Tuple, Any

from config import Config, DEFAULT_CONFIG
from data_utils import EVSchedule, generate_ev_schedules, generate_synthetic_prices, get_daily_price_curve


class EVChargingEnv(gym.Env):
    """Gymnasium environment for fleet depot V2G charging optimization.

    This environment simulates one day (96 steps × 15 min = 24 hours)
    of charging operations at a delivery fleet depot. The agent must
    coordinate charging across all EVs to minimize total cost while
    meeting all constraints.

    The key challenge is the COUPLING between EVs: they all share
    the same transformer, so charging one EV affects what's available
    for others. This is what makes simple per-EV optimization insufficient.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        cfg: Config = DEFAULT_CONFIG,
        schedules: Optional[List[EVSchedule]] = None,
        price_curve: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None,
        custom_reward_fn=None,
    ):
        """Initialize the EV charging environment.

        Args:
            cfg:              Configuration object
            schedules:        Pre-generated EV schedules (or None to auto-generate
                              fresh random schedules each episode)
            price_curve:      Pre-generated price curve (or None to auto-generate
                              a random day from synthetic prices each episode)
            render_mode:      Rendering mode ("human" for text output per step)
            custom_reward_fn: Optional callable with signature:
                                fn(step_cost, deadline_penalty, overload_penalty,
                                   charging_energy, discharging_energy,
                                   soc_array, cfg, current_step, price) -> float
                              If provided, replaces the default reward formula.
                              Students implement this in main.ipynb to experiment
                              with reward shaping without modifying this file.
        """
        super().__init__()

        self.cfg = cfg
        self.render_mode = render_mode
        self._schedules_provided = schedules
        self._price_curve_provided = price_curve
        self._custom_reward_fn = custom_reward_fn

        # Shortcuts
        self.num_evs = cfg.fleet.num_evs
        self.total_steps = cfg.time.total_steps
        self.dt = cfg.time.dt_hours  # hours per step

        # ----------------------------------------------------------
        # Action space: one continuous value per EV in [-1, +1]
        # ----------------------------------------------------------
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_evs,),
            dtype=np.float32,
        )

        # ----------------------------------------------------------
        # Observation space
        # Per EV (3 values each):
        #   0: current SoC               [0, 1]
        #   1: normalized time to depart  [0, 1]  (1=just arrived, 0=departing now)
        #   2: is connected               {0, 1}
        # Global (3 values):
        #   0: normalized time step       [0, 1]
        #   1: normalized current price   [0, 1]
        #   2: current load fraction      [0, 1]  (total_power / transformer_limit)
        # ----------------------------------------------------------
        obs_dim = 3 * self.num_evs + 3
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Initialize state (will be set in reset())
        self.current_step = 0
        self.soc = np.zeros(self.num_evs, dtype=np.float64)
        self.schedules: List[EVSchedule] = []
        self.price_curve = np.zeros(self.total_steps, dtype=np.float64)
        self.price_max = 1.0  # For normalization

        # Episode tracking
        self.episode_cost = 0.0
        self.episode_v2g_revenue = 0.0
        self.episode_degradation = 0.0
        self.episode_penalties = 0.0
        self.episode_actions_log: List[np.ndarray] = []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment for a new episode.

        Each reset generates new random EV schedules and/or selects
        a random day's price curve, providing training variety.

        Returns:
            observation: Initial state vector
            info:        Dict with episode metadata
        """
        super().reset(seed=seed)

        # Generate or use provided schedules
        if self._schedules_provided is not None:
            self.schedules = self._schedules_provided
        else:
            ep_seed = seed if seed is not None else self.np_random.integers(0, 100_000)
            self.schedules = generate_ev_schedules(self.cfg, seed=int(ep_seed))

        # Generate or use provided price curve
        if self._price_curve_provided is not None:
            self.price_curve = self._price_curve_provided.copy()
        else:
            # Generate prices and pick a random day
            ep_seed = seed if seed is not None else self.np_random.integers(0, 100_000)
            prices_df = generate_synthetic_prices(self.cfg, num_days=30, seed=int(ep_seed))
            day = self.np_random.integers(0, 30)
            self.price_curve = get_daily_price_curve(prices_df, day_index=int(day), cfg=self.cfg)

        self.price_max = max(self.price_curve.max(), 1.0)

        # Initialize SoC for all EVs
        self.soc = np.zeros(self.num_evs, dtype=np.float64)
        for ev in self.schedules:
            self.soc[ev.ev_id] = ev.arrival_soc

        # Reset time
        self.current_step = 0

        # Reset episode tracking
        self.episode_cost = 0.0
        self.episode_v2g_revenue = 0.0
        self.episode_degradation = 0.0
        self.episode_penalties = 0.0
        self.episode_actions_log = []

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step of charging decisions.

        Args:
            action: Array of shape (num_evs,) with values in [-1, +1]
                   +1 = charge at max rate
                    0 = idle
                   -1 = discharge at max rate (V2G)

        Returns:
            observation: New state vector
            reward:      Scalar reward (higher is better)
            terminated:  True if episode is done (end of day)
            truncated:   Always False (no early termination)
            info:        Dict with step details
        """
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        # ----------------------------------------------------------
        # 1. Determine which EVs are connected
        # ----------------------------------------------------------
        connected = self._get_connected_mask()

        # Zero out actions for disconnected EVs
        action = action * connected

        # ----------------------------------------------------------
        # 2. Convert actions to power (kW)
        #    action > 0: charging, action < 0: discharging (V2G)
        # ----------------------------------------------------------
        power_kw = np.zeros(self.num_evs, dtype=np.float64)
        for i, ev in enumerate(self.schedules):
            if connected[i]:
                if action[i] >= 0:
                    power_kw[i] = action[i] * ev.max_charge_kw
                else:
                    power_kw[i] = action[i] * ev.max_discharge_kw

        # ----------------------------------------------------------
        # 3. Enforce transformer constraint
        #    Scale down proportionally if aggregate exceeds limit
        # ----------------------------------------------------------
        total_charging = np.sum(np.maximum(power_kw, 0))
        if total_charging > self.cfg.grid.transformer_limit_kw:
            scale = self.cfg.grid.transformer_limit_kw / total_charging
            power_kw = np.where(power_kw > 0, power_kw * scale, power_kw)

        total_discharging = np.abs(np.sum(np.minimum(power_kw, 0)))
        if total_discharging > self.cfg.grid.transformer_limit_kw:
            scale = self.cfg.grid.transformer_limit_kw / total_discharging
            power_kw = np.where(power_kw < 0, power_kw * scale, power_kw)

        # ----------------------------------------------------------
        # 4. Update SoC with energy balance
        # ----------------------------------------------------------
        energy_kwh = np.zeros(self.num_evs, dtype=np.float64)
        for i, ev in enumerate(self.schedules):
            if connected[i] and power_kw[i] != 0:
                if power_kw[i] > 0:  # Charging
                    energy = power_kw[i] * self.dt * self.cfg.battery.charge_efficiency
                else:  # Discharging
                    energy = power_kw[i] * self.dt / self.cfg.battery.discharge_efficiency

                # Apply SoC bounds
                new_soc = self.soc[i] + energy / ev.battery_kwh
                new_soc = np.clip(new_soc, self.cfg.battery.soc_min, self.cfg.battery.soc_max)

                # Actual energy transferred
                actual_delta = new_soc - self.soc[i]
                energy_kwh[i] = actual_delta * ev.battery_kwh

                self.soc[i] = new_soc

        # ----------------------------------------------------------
        # 5. Calculate costs and rewards
        # ----------------------------------------------------------
        price = self.price_curve[self.current_step]  # $/MWh
        price_per_kwh = price / 1000.0  # Convert to $/kWh

        # Energy cost for charging
        charging_energy = np.maximum(energy_kwh, 0)
        charging_cost = np.sum(charging_energy) * price_per_kwh

        # V2G revenue from discharging
        discharging_energy = np.abs(np.minimum(energy_kwh, 0))
        v2g_revenue = np.sum(discharging_energy) * price_per_kwh

        # Degradation cost for V2G
        degradation_cost = np.sum(discharging_energy) * self.cfg.battery.degradation_cost_per_kwh

        # Net cost this step
        step_cost = charging_cost - v2g_revenue + degradation_cost

        # Track cumulative
        self.episode_cost += charging_cost
        self.episode_v2g_revenue += v2g_revenue
        self.episode_degradation += degradation_cost

        # ----------------------------------------------------------
        # 6. Check for deadline violations (at departure)
        # ----------------------------------------------------------
        deadline_penalty = 0.0
        for ev in self.schedules:
            if self.current_step == ev.departure_step:
                if self.soc[ev.ev_id] < ev.target_soc - 0.01:  # Small tolerance
                    shortfall = ev.target_soc - self.soc[ev.ev_id]
                    deadline_penalty += self.cfg.rl.reward_deadline_penalty * shortfall

        # ----------------------------------------------------------
        # 7. Check for transformer overload
        # ----------------------------------------------------------
        total_power = np.sum(np.abs(power_kw))
        overload_penalty = 0.0
        if total_power > self.cfg.grid.transformer_limit_kw * 1.01:  # 1% tolerance
            overload_frac = (total_power / self.cfg.grid.transformer_limit_kw) - 1.0
            overload_penalty = self.cfg.rl.reward_overload_penalty * overload_frac

        self.episode_penalties += deadline_penalty + overload_penalty

        # ----------------------------------------------------------
        # 8. Compose reward
        #
        # DEFAULT: weighted sum of electricity cost + constraint penalties.
        #   reward = -(price_weight × step_cost) - deadline_penalty - overload_penalty
        #
        # The intuition:
        #   - step_cost = electricity bought - V2G sold + battery degradation
        #   - deadline_penalty fires when an EV departs below target SoC
        #   - overload_penalty fires when transformer limit is exceeded
        #
        # If a custom_reward_fn was provided at construction time (e.g., by
        # a student in the notebook), use that instead of the default.
        # ----------------------------------------------------------
        if self._custom_reward_fn is not None:
            reward = float(self._custom_reward_fn(
                step_cost=step_cost,
                deadline_penalty=deadline_penalty,
                overload_penalty=overload_penalty,
                charging_energy=charging_energy,
                discharging_energy=discharging_energy,
                soc_array=self.soc.copy(),
                cfg=self.cfg,
                current_step=self.current_step,
                price=price,
            ))
        else:
            reward = (
                - self.cfg.rl.reward_price_weight * step_cost
                - deadline_penalty
                - overload_penalty
            )

        # Log actions for analysis
        self.episode_actions_log.append(power_kw.copy())

        # ----------------------------------------------------------
        # 9. Advance time
        # ----------------------------------------------------------
        self.current_step += 1
        terminated = self.current_step >= self.total_steps
        truncated = False

        # Final deadline check at end of episode
        if terminated:
            for ev in self.schedules:
                if ev.departure_step >= self.total_steps:
                    if self.soc[ev.ev_id] < ev.target_soc - 0.01:
                        shortfall = ev.target_soc - self.soc[ev.ev_id]
                        penalty = self.cfg.rl.reward_deadline_penalty * shortfall
                        reward -= penalty
                        self.episode_penalties += penalty

        obs = self._get_observation()
        info = self._get_info()
        info.update({
            "step_cost": step_cost,
            "charging_cost": charging_cost,
            "v2g_revenue": v2g_revenue,
            "degradation_cost": degradation_cost,
            "deadline_penalty": deadline_penalty,
            "overload_penalty": overload_penalty,
            "total_power_kw": total_power,
            "power_kw": power_kw.copy(),
            "price_mwh": price,
        })

        if self.render_mode == "human":
            self._render_step(info)

        return obs, float(reward), terminated, truncated, info

    # ==============================================================
    # Observation Construction
    # ==============================================================
    def _get_observation(self) -> np.ndarray:
        """Build the observation vector.

        Layout:
          [ev0_soc, ev0_time_to_depart, ev0_connected,
           ev1_soc, ev1_time_to_depart, ev1_connected,
           ...
           norm_time, norm_price, load_fraction]
        """
        obs = np.zeros(3 * self.num_evs + 3, dtype=np.float32)

        for i, ev in enumerate(self.schedules):
            base = i * 3
            obs[base + 0] = float(self.soc[i])

            # Normalized time until departure (1 = just arrived, 0 = departing now)
            if self.current_step < ev.arrival_step:
                obs[base + 1] = 1.0  # Not yet arrived
            elif self.current_step >= ev.departure_step:
                obs[base + 1] = 0.0  # Already departed
            else:
                remaining = ev.departure_step - self.current_step
                total = ev.departure_step - ev.arrival_step
                obs[base + 1] = float(remaining / max(total, 1))

            # Connected flag
            obs[base + 2] = float(self._is_connected(i))

        # Global observations
        global_base = 3 * self.num_evs
        obs[global_base + 0] = float(self.current_step / max(self.total_steps - 1, 1))

        if self.current_step < self.total_steps:
            obs[global_base + 1] = float(self.price_curve[self.current_step] / self.price_max)
        else:
            obs[global_base + 1] = 0.0

        # Load fraction (from previous step's actions)
        if len(self.episode_actions_log) > 0:
            last_power = np.abs(self.episode_actions_log[-1])
            obs[global_base + 2] = float(
                np.sum(last_power) / max(self.cfg.grid.transformer_limit_kw, 1.0)
            )

        return obs

    # ==============================================================
    # Helper Methods
    # ==============================================================
    def _get_connected_mask(self) -> np.ndarray:
        """Return boolean mask: True if EV is plugged in at current step."""
        mask = np.zeros(self.num_evs, dtype=np.float64)
        for i, ev in enumerate(self.schedules):
            if ev.arrival_step <= self.current_step < ev.departure_step:
                mask[i] = 1.0
        return mask

    def _is_connected(self, ev_idx: int) -> bool:
        """Check if a specific EV is connected."""
        ev = self.schedules[ev_idx]
        return ev.arrival_step <= self.current_step < ev.departure_step

    def _get_info(self) -> Dict[str, Any]:
        """Compile episode information dict."""
        connected = self._get_connected_mask()
        evs_meeting_target = sum(
            1 for ev in self.schedules
            if self.soc[ev.ev_id] >= ev.target_soc - 0.01
        )
        return {
            "step": self.current_step,
            "num_connected": int(connected.sum()),
            "avg_soc": float(self.soc.mean()),
            "min_soc": float(self.soc.min()),
            "evs_meeting_target": evs_meeting_target,
            "total_evs": self.num_evs,
            "episode_cost": self.episode_cost,
            "episode_v2g_revenue": self.episode_v2g_revenue,
            "episode_degradation": self.episode_degradation,
            "episode_penalties": self.episode_penalties,
            "episode_net_cost": (self.episode_cost - self.episode_v2g_revenue
                                + self.episode_degradation),
        }

    def get_schedules(self) -> List[EVSchedule]:
        """Return current episode's EV schedules."""
        return self.schedules

    def get_price_curve(self) -> np.ndarray:
        """Return current episode's price curve."""
        return self.price_curve.copy()

    def get_soc_array(self) -> np.ndarray:
        """Return current SoC for all EVs."""
        return self.soc.copy()

    def get_actions_log(self) -> np.ndarray:
        """Return all actions taken this episode as array (steps, num_evs)."""
        if len(self.episode_actions_log) == 0:
            return np.array([])
        return np.array(self.episode_actions_log)

    # ==============================================================
    # Rendering
    # ==============================================================
    def _render_step(self, info: Dict):
        """Print a text summary of the current step."""
        t = info["step"]
        hour = (self.cfg.time.start_hour + t * self.cfg.time.dt_hours) % 24
        print(f"Step {t:3d} ({int(hour):02d}:{int((hour%1)*60):02d}) | "
              f"Price: ${info['price_mwh']:6.1f}/MWh | "
              f"Power: {info['total_power_kw']:6.1f}/{self.cfg.grid.transformer_limit_kw:.0f} kW | "
              f"Connected: {info['num_connected']:2d} | "
              f"Avg SoC: {info['avg_soc']:.1%} | "
              f"Cost: ${info['step_cost']:.3f}")


# ============================================================
# Environment Registration
# ============================================================
def make_env(cfg: Config = DEFAULT_CONFIG, custom_reward_fn=None, **kwargs) -> EVChargingEnv:
    """Factory function to create the environment.

    Args:
        cfg:              Configuration object
        custom_reward_fn: Optional custom reward function (see EVChargingEnv
                          docstring for the expected signature). Students pass
                          their reward function here when training a custom agent.
        **kwargs:         Additional arguments passed to EVChargingEnv
                          (schedules, price_curve, render_mode)

    Returns:
        EVChargingEnv instance
    """
    return EVChargingEnv(cfg=cfg, custom_reward_fn=custom_reward_fn, **kwargs)


# ============================================================
# Visualization
# ============================================================
def plot_episode_results(
    env: EVChargingEnv,
    title: str = "Charging Schedule",
):
    """Plot comprehensive results from a completed episode.

    Creates a 4-panel figure:
    1. Power allocation per EV over time (heatmap)
    2. Aggregate power vs transformer limit
    3. Price curve with charging overlay
    4. SoC trajectories for all EVs

    Args:
        env:   Environment after episode completion
        title: Plot title prefix
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    actions = env.get_actions_log()
    if len(actions) == 0:
        print("No actions logged — run an episode first.")
        return

    cfg = env.cfg
    steps = np.arange(len(actions))
    hours = (cfg.time.start_hour + steps * cfg.time.dt_hours) % 24

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"{title} — Net Cost: ${env.episode_cost - env.episode_v2g_revenue + env.episode_degradation:.2f}", fontsize=14)

    # --- Panel 1: Power heatmap ---
    ax1 = axes[0, 0]
    # Custom diverging colormap: red (discharge) → white (idle) → blue (charge)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "charge", ["#EF4444", "#FFFFFF", "#3B82F6"]
    )
    im = ax1.imshow(
        actions.T,
        aspect="auto",
        cmap=cmap,
        vmin=-cfg.battery.max_discharge_rate_kw,
        vmax=cfg.battery.max_charge_rate_kw,
        interpolation="nearest",
    )
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("EV ID")
    ax1.set_title("Power Allocation (kW): Blue=Charge, Red=Discharge")
    plt.colorbar(im, ax=ax1, label="Power (kW)")

    # --- Panel 2: Aggregate power ---
    ax2 = axes[0, 1]
    total_charge = np.sum(np.maximum(actions, 0), axis=1)
    total_discharge = np.sum(np.minimum(actions, 0), axis=1)
    ax2.fill_between(steps, 0, total_charge, alpha=0.6, color="#3B82F6", label="Charging")
    ax2.fill_between(steps, total_discharge, 0, alpha=0.6, color="#EF4444", label="Discharging")
    ax2.axhline(y=cfg.grid.transformer_limit_kw, color="red", linestyle="--",
                linewidth=2, label=f"Transformer limit ({cfg.grid.transformer_limit_kw} kW)")
    ax2.axhline(y=-cfg.grid.transformer_limit_kw, color="red", linestyle="--", linewidth=2)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Power (kW)")
    ax2.set_title("Aggregate Power Profile")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Price curve ---
    ax3 = axes[1, 0]
    prices = env.get_price_curve()[:len(actions)]
    ax3.plot(steps, prices, "k-", linewidth=1.5, label="Price")
    ax3.fill_between(steps, 0, prices, alpha=0.1, color="orange")

    # Overlay charging power on secondary axis
    ax3b = ax3.twinx()
    ax3b.fill_between(steps, 0, total_charge, alpha=0.3, color="#3B82F6", label="Charging")
    ax3b.fill_between(steps, total_discharge, 0, alpha=0.3, color="#EF4444", label="Discharging")
    ax3b.set_ylabel("Power (kW)", color="blue")

    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Price ($/MWh)")
    ax3.set_title("Price vs. Charging Profile")
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: SoC trajectories ---
    ax4 = axes[1, 1]
    # Reconstruct SoC trajectories from actions
    soc_history = _reconstruct_soc_history(env)
    for i in range(min(env.num_evs, 20)):
        ev = env.schedules[i]
        ax4.plot(steps, soc_history[i, :len(steps)], linewidth=0.8, alpha=0.7,
                label=f"EV{i}" if i < 5 else None)
        # Mark departure
        if ev.departure_step < len(steps):
            ax4.plot(ev.departure_step, soc_history[i, ev.departure_step],
                    "rv" if soc_history[i, ev.departure_step] < ev.target_soc else "g^",
                    markersize=4)

    ax4.axhline(y=cfg.fleet.target_soc, color="green", linestyle="--",
                linewidth=1.5, label=f"Target SoC ({cfg.fleet.target_soc:.0%})")
    ax4.axhline(y=cfg.battery.soc_min, color="red", linestyle=":",
                linewidth=1, label=f"Min SoC ({cfg.battery.soc_min:.0%})")
    ax4.set_xlabel("Time Step")
    ax4.set_ylabel("State of Charge")
    ax4.set_title("Battery SoC Trajectories (▲=target met, ▼=missed)")
    ax4.set_ylim(0, 1.05)
    ax4.legend(fontsize=7, ncol=2)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}_results.png", dpi=150, bbox_inches="tight")
    plt.show()


def _reconstruct_soc_history(env: EVChargingEnv) -> np.ndarray:
    """Reconstruct SoC trajectories from action log.

    Returns array of shape (num_evs, total_steps+1).
    """
    actions = env.get_actions_log()
    num_steps = len(actions)
    cfg = env.cfg

    soc = np.zeros((env.num_evs, num_steps + 1), dtype=np.float64)
    for ev in env.schedules:
        soc[ev.ev_id, 0] = ev.arrival_soc

    for t in range(num_steps):
        soc[:, t + 1] = soc[:, t]
        for i, ev in enumerate(env.schedules):
            if ev.arrival_step <= t < ev.departure_step and actions[t, i] != 0:
                power = actions[t, i]
                if power > 0:
                    energy = power * cfg.time.dt_hours * cfg.battery.charge_efficiency
                else:
                    energy = power * cfg.time.dt_hours / cfg.battery.discharge_efficiency
                new_soc = soc[i, t] + energy / ev.battery_kwh
                soc[i, t + 1] = np.clip(new_soc, cfg.battery.soc_min, cfg.battery.soc_max)

    return soc


# ============================================================
# Quick Test
# ============================================================
if __name__ == "__main__":
    from data_utils import generate_synthetic_prices, get_daily_price_curve

    cfg = Config()
    print(cfg.summary())

    # Generate data
    prices_df = generate_synthetic_prices(cfg, num_days=30)
    price_curve = get_daily_price_curve(prices_df, day_index=0, cfg=cfg)
    schedules = generate_ev_schedules(cfg)

    # Create environment
    env = make_env(cfg, schedules=schedules, price_curve=price_curve, render_mode="human")
    obs, info = env.reset()

    print(f"\nObservation shape: {obs.shape}")
    print(f"Action shape: {env.action_space.shape}")
    print(f"Initial info: {info}")

    # Run one episode with random actions
    print("\n--- Running episode with random actions ---")
    total_reward = 0.0
    done = False
    step_count = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step_count += 1

    print(f"\n--- Episode Complete ---")
    print(f"Steps: {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Electricity cost: ${info['episode_cost']:.2f}")
    print(f"V2G revenue:      ${info['episode_v2g_revenue']:.2f}")
    print(f"Degradation cost: ${info['episode_degradation']:.2f}")
    print(f"Net cost:         ${info['episode_net_cost']:.2f}")
    print(f"Penalties:        ${info['episode_penalties']:.2f}")
    print(f"EVs meeting target: {info['evs_meeting_target']}/{info['total_evs']}")

    # Visualize
    try:
        plot_episode_results(env, title="Random Actions")
    except ImportError:
        print("matplotlib not available — skipping plots")
