"""
rl_agent.py — Reinforcement Learning Agent (Pillar 3)
======================================================

Trains a PPO (Proximal Policy Optimization) agent using Stable-Baselines3
to learn a charging policy through trial and error in the EVChargingEnv.

WHY REINFORCEMENT LEARNING?
----------------------------
Unlike the LP optimizer (which requires knowing all future prices) and the
ML/DL forecasters (which only predict prices, not directly make charging
decisions), the RL agent learns a *direct policy*: given the current state
(SoC levels, time remaining, current price), output charging decisions.

The agent learns by running thousands of simulated charging episodes and
receiving a scalar reward signal. Over time, it discovers patterns like:
"charge aggressively at night (low price), reduce charging at 6-8 PM
(peak price), use V2G to discharge when price spikes."

KEY INSIGHT: THE COST OF UNCERTAINTY
--------------------------------------
The LP optimizer solves a mathematical program knowing all 96 future price
values — it achieves the theoretical minimum cost. The RL agent only sees the
*current* price and must learn to anticipate future prices from the time-of-day
signal. The gap between their costs = "the value of perfect information."
This gap has real economic significance: it represents how much a depot would
pay for a reliable 24-hour price forecast.

  LP cost (perfect foresight) ≤ RL cost (learned heuristic) ≤ ASAP cost (no optimization)

PPO ALGORITHM OVERVIEW
-----------------------
PPO is an actor-critic algorithm. Two neural networks are trained jointly:
  - Policy (actor) π(a|s):   maps state → action distribution (Gaussian here,
                              since actions are continuous)
  - Value function (critic) V(s): estimates expected future reward from state s

PPO uses a "clipped surrogate objective" to limit how much the policy can
change in a single update, preventing large destabilizing updates that cause
training divergence. This makes it significantly more stable than vanilla
policy gradient methods.

NETWORK ARCHITECTURE
---------------------
Both policy and value networks are 2-layer MLPs:
  State (63-dim) → Linear(256) → ReLU → Linear(128) → ReLU → output

STUDENT WORK
------------
Students design a custom reward function in main.ipynb and train an improved
agent. The .py file is provided infrastructure; do not modify it directly.

IEOR E4010: AI for Operations Research and Financial Engineering
Columbia University, Spring 2026
"""

import os
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import warnings

from config import Config, DEFAULT_CONFIG
from data_utils import generate_ev_schedules, generate_synthetic_prices, get_daily_price_curve
from environment import EVChargingEnv, make_env

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    warnings.warn(
        "stable-baselines3 not installed. Install with:\n"
        "  pip install stable-baselines3\n"
        "RL training will not be available, but you can still run other modules."
    )


# ============================================================
# Custom Callback for Logging
# ============================================================
if HAS_SB3:
    class ChargingCallback(BaseCallback):
        """Custom callback to log episode metrics during training."""

        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.episode_costs = []
            self.episode_targets_met = []

        def _on_step(self) -> bool:
            # Check for episode end in infos
            for info in self.locals.get("infos", []):
                if "episode" in info:
                    self.episode_costs.append(info.get("episode_net_cost", 0))
                    self.episode_targets_met.append(
                        info.get("evs_meeting_target", 0) / max(info.get("total_evs", 1), 1)
                    )
            return True


# ============================================================
# Training
# ============================================================
def train_rl_agent(
    cfg: Config = DEFAULT_CONFIG,
    total_timesteps: Optional[int] = None,
    save_path: str = "ppo_ev_charging",
    verbose: bool = True,
    custom_reward_fn=None,
) -> Dict[str, Any]:
    """Train a PPO agent on the EV charging environment.

    TRAINING PROCEDURE
    ------------------
    1. Create a vectorized training environment (DummyVecEnv wrapping
       EVChargingEnv). Each episode randomizes EV schedules and picks a
       random day's price curve — this provides diverse training scenarios
       and prevents overfitting to a single charging pattern.

    2. Create a fixed evaluation environment (same EV schedules and price
       curve every evaluation call) for consistent performance tracking.

    3. Train PPO with the provided hyperparameters. PPO collects n_steps
       of experience, then performs n_epochs mini-batch gradient updates.

    4. EvalCallback evaluates the policy every eval_freq steps on the fixed
       evaluation scenario and saves the best checkpoint.

    5. ChargingCallback logs episode costs and target achievement rates
       for plotting training progress.

    TRAINING TIME
    -------------
    200k timesteps ≈ ~2000 episodes × 96 steps. On CPU this takes roughly
    5-10 minutes. On GPU it is marginally faster (the environment is the
    bottleneck, not the neural network).

    Args:
        cfg:              Configuration (hyperparameters in cfg.rl)
        total_timesteps:  Override cfg.rl.total_timesteps
        save_path:        Where to save the trained model (.zip extension added)
        verbose:          Print training progress
        custom_reward_fn: Optional custom reward function passed to the env.
                          See EVChargingEnv for the expected signature.

    Returns:
        Dict with:
            model           — trained PPO model
            save_path       — where model was saved
            training_stats  — {episode_costs, episode_targets_met} lists
            total_timesteps — number of timesteps trained
    """
    if not HAS_SB3:
        raise ImportError("stable-baselines3 is required for RL training")

    if total_timesteps is None:
        total_timesteps = cfg.rl.total_timesteps

    if verbose:
        print(f"Training PPO for {total_timesteps:,} timesteps...")
        print(f"  Learning rate: {cfg.rl.learning_rate}")
        print(f"  Batch size: {cfg.rl.batch_size}")
        print(f"  Reward weights: price={cfg.rl.reward_price_weight}, "
              f"deadline={cfg.rl.reward_deadline_penalty}, "
              f"overload={cfg.rl.reward_overload_penalty}")

    # Create training environment (random scenarios each episode for diversity)
    def make_train_env():
        env = make_env(cfg, custom_reward_fn=custom_reward_fn)
        env = Monitor(env)
        return env

    vec_env = DummyVecEnv([make_train_env])

    # Create evaluation environment (fixed scenario for consistent eval)
    eval_schedules = generate_ev_schedules(cfg, seed=999)
    eval_prices_df = generate_synthetic_prices(cfg, num_days=30, seed=999)
    eval_price_curve = get_daily_price_curve(eval_prices_df, day_index=0, cfg=cfg)

    eval_env = Monitor(make_env(cfg, schedules=eval_schedules, price_curve=eval_price_curve))

    # PPO agent
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=cfg.rl.learning_rate,
        n_steps=cfg.rl.n_steps,
        batch_size=cfg.rl.batch_size,
        n_epochs=cfg.rl.n_epochs,
        gamma=cfg.rl.gamma,
        gae_lambda=cfg.rl.gae_lambda,
        clip_range=cfg.rl.clip_range,
        ent_coef=cfg.rl.ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=1 if verbose else 0,
        seed=cfg.rl.seed,
        device="auto",
    )

    # Callbacks
    charging_callback = ChargingCallback()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./{save_path}_best/",
        log_path=f"./{save_path}_logs/",
        eval_freq=cfg.rl.eval_freq,
        n_eval_episodes=cfg.rl.n_eval_episodes,
        deterministic=True,
        verbose=1 if verbose else 0,
    )

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, charging_callback],
        progress_bar=verbose,
    )

    # Save final model
    model.save(save_path)
    if verbose:
        print(f"\nModel saved to {save_path}")

    return {
        "model": model,
        "save_path": save_path,
        "training_stats": {
            "episode_costs": charging_callback.episode_costs,
            "episode_targets_met": charging_callback.episode_targets_met,
        },
        "total_timesteps": total_timesteps,
    }


# ============================================================
# Evaluation
# ============================================================
def evaluate_rl_agent(
    model_or_path,
    cfg: Config = DEFAULT_CONFIG,
    schedules=None,
    price_curve=None,
    n_episodes: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate a trained RL agent.

    Args:
        model_or_path: Trained PPO model or path to saved model
        cfg:           Configuration
        schedules:     Fixed schedules for evaluation (or None for random)
        price_curve:   Fixed price curve (or None for random)
        n_episodes:    Number of evaluation episodes
        verbose:       Print results

    Returns:
        Dict with evaluation metrics
    """
    if not HAS_SB3:
        raise ImportError("stable-baselines3 is required")

    if isinstance(model_or_path, str):
        model = PPO.load(model_or_path)
    else:
        model = model_or_path

    all_costs = []
    all_v2g = []
    all_targets = []
    all_rewards = []
    all_penalties = []
    last_env = None

    for ep in range(n_episodes):
        env = make_env(cfg, schedules=schedules, price_curve=price_curve)
        obs, info = env.reset()

        total_reward = 0.0
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        all_costs.append(info["episode_net_cost"])
        all_v2g.append(info["episode_v2g_revenue"])
        all_targets.append(info["evs_meeting_target"])
        all_rewards.append(total_reward)
        all_penalties.append(info["episode_penalties"])
        last_env = env

    results = {
        "strategy": "ppo_rl",
        "net_cost_mean": np.mean(all_costs),
        "net_cost_std": np.std(all_costs),
        "v2g_revenue_mean": np.mean(all_v2g),
        "targets_met_mean": np.mean(all_targets),
        "targets_met_min": np.min(all_targets),
        "total_reward_mean": np.mean(all_rewards),
        "penalties_mean": np.mean(all_penalties),
        "total_evs": cfg.fleet.num_evs,
        "n_episodes": n_episodes,
        "last_env": last_env,
    }

    if verbose:
        print(f"\nRL Agent Evaluation ({n_episodes} episodes):")
        print(f"  Net cost:     ${results['net_cost_mean']:.2f} ± ${results['net_cost_std']:.2f}")
        print(f"  V2G revenue:  ${results['v2g_revenue_mean']:.2f}")
        print(f"  Targets met:  {results['targets_met_mean']:.1f}/{cfg.fleet.num_evs} "
              f"(min: {results['targets_met_min']})")
        print(f"  Penalties:    ${results['penalties_mean']:.2f}")
        print(f"  Avg reward:   {results['total_reward_mean']:.2f}")

    return results


def run_single_episode(
    model_or_path,
    cfg: Config = DEFAULT_CONFIG,
    schedules=None,
    price_curve=None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a single episode and return detailed results for comparison.

    Returns dict compatible with heuristic/LP result format.
    """
    if not HAS_SB3:
        raise ImportError("stable-baselines3 is required")

    if isinstance(model_or_path, str):
        model = PPO.load(model_or_path)
    else:
        model = model_or_path

    env = make_env(cfg, schedules=schedules, price_curve=price_curve)
    obs, info = env.reset()

    total_reward = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    return {
        "strategy": "ppo_rl",
        "total_reward": total_reward,
        "total_cost": info["episode_cost"],
        "v2g_revenue": info["episode_v2g_revenue"],
        "degradation_cost": info["episode_degradation"],
        "net_cost": info["episode_net_cost"],
        "penalties": info["episode_penalties"],
        "evs_meeting_target": info["evs_meeting_target"],
        "total_evs": info["total_evs"],
        "env": env,
    }


# ============================================================
# Visualization
# ============================================================
def plot_training_curves(training_stats: Dict, window: int = 20):
    """Plot training progress."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("RL Training Progress", fontsize=14)

    costs = training_stats.get("episode_costs", [])
    targets = training_stats.get("episode_targets_met", [])

    if costs:
        ax1 = axes[0]
        ax1.plot(costs, alpha=0.2, color="blue")
        if len(costs) >= window:
            smoothed = np.convolve(costs, np.ones(window) / window, mode="valid")
            ax1.plot(range(window - 1, len(costs)), smoothed, "b-", linewidth=2,
                    label=f"Moving avg (w={window})")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Net Cost ($)")
        ax1.set_title("Episode Net Cost (lower is better)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    if targets:
        ax2 = axes[1]
        ax2.plot(targets, alpha=0.2, color="green")
        if len(targets) >= window:
            smoothed = np.convolve(targets, np.ones(window) / window, mode="valid")
            ax2.plot(range(window - 1, len(targets)), smoothed, "g-", linewidth=2,
                    label=f"Moving avg (w={window})")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Fraction Meeting Target")
        ax2.set_title("Target SoC Achievement (higher is better)")
        ax2.set_ylim(0, 1.05)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("rl_training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    if not HAS_SB3:
        print("stable-baselines3 not installed. Install with:")
        print("  pip install stable-baselines3")
        exit(1)

    cfg = Config()
    # Shorter training for quick test
    cfg.rl.total_timesteps = 50_000

    # Train
    result = train_rl_agent(cfg, verbose=True)

    # Evaluate
    eval_result = evaluate_rl_agent(result["model"], cfg, n_episodes=5)

    # Plot training curves
    try:
        plot_training_curves(result["training_stats"])
    except ImportError:
        print("matplotlib not available — skipping plots")
