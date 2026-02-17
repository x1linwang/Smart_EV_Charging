"""
data_utils.py — Data Generation for EV Smart Charging Project
===============================================================

Provides two data sources:
1. SYNTHETIC (default): Generates realistic electricity prices and
   EV schedules based on statistical patterns from real markets.
   No internet or API keys required.

2. REALISTIC (optional): Downloads actual wholesale electricity prices
   from US ISOs via the `gridstatus` library. Requires internet access
   and `pip install gridstatus`.

IEOR E4010: AI for Operations Research and Financial Engineering
Columbia University, Spring 2026
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from config import Config, DEFAULT_CONFIG


# ============================================================
# EV Schedule Data Structure
# ============================================================
@dataclass
class EVSchedule:
    """Schedule for a single EV at the depot.

    Attributes:
        ev_id:          Unique identifier (0-indexed)
        arrival_step:   Time step when EV arrives and plugs in
        departure_step: Time step when EV must depart (hard deadline)
        arrival_soc:    Battery state of charge at arrival [0, 1]
        target_soc:     Required SoC at departure [0, 1]
        battery_kwh:    Battery capacity (kWh)
        max_charge_kw:  Maximum charging power (kW)
        max_discharge_kw: Maximum V2G discharge power (kW)
    """
    ev_id: int
    arrival_step: int
    departure_step: int
    arrival_soc: float
    target_soc: float
    battery_kwh: float
    max_charge_kw: float
    max_discharge_kw: float

    @property
    def energy_needed_kwh(self) -> float:
        """Energy required to reach target from arrival SoC."""
        return (self.target_soc - self.arrival_soc) * self.battery_kwh

    @property
    def hours_available(self) -> float:
        """Hours between arrival and departure."""
        return (self.departure_step - self.arrival_step) * 0.25  # 15-min steps

    def __repr__(self) -> str:
        return (f"EV{self.ev_id:02d}(arrive=step{self.arrival_step}, "
                f"depart=step{self.departure_step}, "
                f"SoC={self.arrival_soc:.0%}→{self.target_soc:.0%}, "
                f"need={self.energy_needed_kwh:.1f}kWh, "
                f"avail={self.hours_available:.1f}h)")


# ============================================================
# Synthetic Price Generation
# ============================================================
def generate_synthetic_prices(
    cfg: Config = DEFAULT_CONFIG,
    num_days: Optional[int] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate synthetic electricity prices mimicking CAISO patterns.

    The price model captures three key real-world patterns:
    1. Overnight valley: Low demand → low prices (midnight–6 AM)
    2. Solar duck curve: Midday dip from solar over-generation (10 AM–3 PM)
    3. Evening ramp:     Sharp peak as solar drops and demand rises (5–9 PM)

    Additionally, weekday/weekend effects and random noise create
    realistic day-to-day variation for ML/DL training.

    Args:
        cfg:      Configuration object with price parameters
        num_days: Number of days to generate (default: cfg.price.num_days_history)
        seed:     Random seed (default: cfg.seed)

    Returns:
        DataFrame with columns:
        - datetime:       Timestamp (hourly resolution)
        - price_mwh:      Price in $/MWh
        - hour:           Hour of day (0–23)
        - day_of_week:    Day of week (0=Mon, 6=Sun)
        - month:          Month (1–12)
        - is_weekend:     Boolean
        - day_index:      Day number (0-indexed)
    """
    if num_days is None:
        num_days = cfg.price.num_days_history
    if seed is None:
        seed = cfg.seed
    rng = np.random.default_rng(seed)

    pcfg = cfg.price
    hours = np.arange(24)
    records = []

    # Start date for the synthetic series
    start_date = pd.Timestamp("2023-01-01")

    for day in range(num_days):
        date = start_date + pd.Timedelta(days=day)
        dow = date.dayofweek  # 0=Mon, 6=Sun
        month = date.month
        is_weekend = dow >= 5

        # Seasonal adjustment: higher prices in summer (cooling load)
        seasonal = 5.0 * np.sin(2 * np.pi * (month - 1) / 12 - np.pi / 3)

        # Weekend discount: lower demand on weekends
        weekend_adj = -8.0 if is_weekend else 0.0

        for hour in hours:
            price = pcfg.base_price + seasonal + weekend_adj

            # Overnight valley (midnight to 6 AM)
            if 0 <= hour < 6:
                price -= pcfg.overnight_discount * (1.0 - hour / 6.0)

            # Morning ramp (6 AM to 10 AM)
            elif 6 <= hour < 10:
                price += 5.0 * (hour - 6) / 4.0

            # Solar duck curve dip (10 AM to 3 PM)
            elif pcfg.solar_peak_start <= hour < pcfg.solar_peak_end:
                mid = (pcfg.solar_peak_start + pcfg.solar_peak_end) / 2
                depth = 1.0 - ((hour - mid) / (pcfg.solar_peak_end - pcfg.solar_peak_start) * 2) ** 2
                price -= pcfg.solar_dip * max(0, depth)

            # Evening peak ramp (5 PM to 9 PM)
            elif pcfg.peak_start <= hour < pcfg.peak_end:
                # Bell-shaped peak centered at 7 PM
                center = (pcfg.peak_start + pcfg.peak_end) / 2
                width = (pcfg.peak_end - pcfg.peak_start) / 2
                peak_shape = 1.0 - ((hour - center) / width) ** 2
                price += pcfg.evening_peak * max(0, peak_shape)

            # Late night wind-down (9 PM to midnight)
            elif hour >= pcfg.peak_end:
                price -= pcfg.overnight_discount * (hour - pcfg.peak_end) / (24 - pcfg.peak_end)

            # Add noise
            noise = rng.normal(0, pcfg.noise_std)
            # Occasional price spikes (1% chance)
            if rng.random() < 0.01:
                noise += rng.exponential(30)
            # Occasional negative prices from solar/wind surplus (0.5% chance, midday only)
            if 10 <= hour <= 15 and rng.random() < 0.005:
                noise -= rng.exponential(20)

            price = max(0, price + noise)  # Floor at $0/MWh

            records.append({
                "datetime": date + pd.Timedelta(hours=hour),
                "price_mwh": round(price, 2),
                "hour": hour,
                "day_of_week": dow,
                "month": month,
                "is_weekend": is_weekend,
                "day_index": day,
            })

    df = pd.DataFrame(records)
    return df


def get_daily_price_curve(
    prices_df: pd.DataFrame,
    day_index: int = 0,
    cfg: Config = DEFAULT_CONFIG,
) -> np.ndarray:
    """Extract a single day's price curve at simulation resolution.

    Interpolates hourly prices to 15-minute resolution for use
    in the environment and optimizer.

    Args:
        prices_df:  DataFrame from generate_synthetic_prices()
        day_index:  Which day to extract
        cfg:        Config for time resolution

    Returns:
        np.ndarray of shape (total_steps,) with prices in $/MWh
    """
    day_data = prices_df[prices_df["day_index"] == day_index].sort_values("hour")
    hourly_prices = day_data["price_mwh"].values

    if len(hourly_prices) < 24:
        raise ValueError(f"Day {day_index} has only {len(hourly_prices)} hours of data")

    # Reorder to start from simulation start hour
    start_hour = int(cfg.time.start_hour)
    reordered = np.concatenate([hourly_prices[start_hour:], hourly_prices[:start_hour]])

    # Interpolate to 15-minute resolution
    hourly_indices = np.arange(24)
    step_indices = np.linspace(0, 23, cfg.time.total_steps, endpoint=False)
    prices_15min = np.interp(step_indices, hourly_indices, reordered)

    return prices_15min


# ============================================================
# EV Schedule Generation
# ============================================================
def generate_ev_schedules(
    cfg: Config = DEFAULT_CONFIG,
    seed: Optional[int] = None,
) -> List[EVSchedule]:
    """Generate stochastic EV arrival/departure schedules.

    Models realistic depot patterns:
    - Arrivals are spread across the afternoon/evening window
    - Departures are spread across the early morning window
    - Arrival SoC follows a truncated normal distribution
    - Each EV has enough time to charge from arrival SoC to target

    Args:
        cfg:  Configuration object
        seed: Random seed (default: cfg.seed)

    Returns:
        List of EVSchedule objects, one per EV
    """
    if seed is None:
        seed = cfg.seed
    rng = np.random.default_rng(seed)

    fcfg = cfg.fleet
    tcfg = cfg.time
    bcfg = cfg.battery

    schedules = []

    for i in range(fcfg.num_evs):
        # Generate arrival time (uniform within window)
        arrival_hour = rng.uniform(fcfg.arrival_start_hour, fcfg.arrival_end_hour)
        # Offset from simulation start
        arrival_offset = arrival_hour - tcfg.start_hour
        if arrival_offset < 0:
            arrival_offset += 24  # Wrap around midnight
        arrival_step = int(arrival_offset * tcfg.steps_per_hour)
        arrival_step = np.clip(arrival_step, 0, tcfg.total_steps - 1)

        # Generate departure time (uniform within window, next day)
        departure_hour = rng.uniform(fcfg.departure_start_hour, fcfg.departure_end_hour)
        departure_offset = departure_hour - tcfg.start_hour
        if departure_offset < 0:
            departure_offset += 24  # Next day
        if departure_offset <= arrival_offset:
            departure_offset += 24  # Ensure departure after arrival
        departure_step = int(departure_offset * tcfg.steps_per_hour)
        departure_step = min(departure_step, tcfg.total_steps - 1)

        # Ensure minimum connection time (at least 2 hours)
        min_steps = int(2 * tcfg.steps_per_hour)
        if departure_step - arrival_step < min_steps:
            departure_step = min(arrival_step + min_steps, tcfg.total_steps - 1)

        # Generate arrival SoC (truncated normal)
        arrival_soc = rng.normal(fcfg.arrival_soc_mean, fcfg.arrival_soc_std)
        arrival_soc = np.clip(arrival_soc, fcfg.arrival_soc_min, fcfg.arrival_soc_max)

        # Verify feasibility: can the EV reach target SoC in available time?
        hours_available = (departure_step - arrival_step) * tcfg.dt_hours
        max_energy = bcfg.max_charge_rate_kw * hours_available * bcfg.charge_efficiency
        energy_needed = (fcfg.target_soc - arrival_soc) * bcfg.capacity_kwh

        # If infeasible, boost arrival SoC to make it feasible
        if energy_needed > max_energy:
            min_arrival_soc = fcfg.target_soc - max_energy / bcfg.capacity_kwh
            arrival_soc = max(arrival_soc, min_arrival_soc + 0.05)  # Small margin

        schedules.append(EVSchedule(
            ev_id=i,
            arrival_step=arrival_step,
            departure_step=departure_step,
            arrival_soc=round(arrival_soc, 3),
            target_soc=fcfg.target_soc,
            battery_kwh=bcfg.capacity_kwh,
            max_charge_kw=bcfg.max_charge_rate_kw,
            max_discharge_kw=bcfg.max_discharge_rate_kw,
        ))

    return schedules


# ============================================================
# Realistic Data Loader (Optional)
# ============================================================
def load_realistic_prices(
    iso: str = "CAISO",
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    node: str = "TH_SP15_GEN-APND",
) -> pd.DataFrame:
    """Load real wholesale electricity prices via gridstatus.

    Requires: pip install gridstatus

    Supports US ISOs: CAISO, ERCOT, PJM, MISO, NYISO, ISONE, SPP.

    Args:
        iso:        ISO name (e.g., "CAISO", "ERCOT", "PJM")
        start_date: Start date string (YYYY-MM-DD)
        end_date:   End date string (YYYY-MM-DD)
        node:       Pricing node (ISO-specific; default is CAISO SP15)

    Returns:
        DataFrame with columns: datetime, price_mwh, hour, day_of_week,
        month, is_weekend, day_index

    Raises:
        ImportError: If gridstatus is not installed
        ValueError:  If ISO is not supported

    Example:
        >>> df = load_realistic_prices("CAISO", "2024-06-01", "2024-06-30")
        >>> print(df.head())
    """
    try:
        import gridstatus
    except ImportError:
        raise ImportError(
            "gridstatus is not installed. Install with:\n"
            "  pip install gridstatus\n\n"
            "Or use generate_synthetic_prices() for offline operation."
        )

    iso_map = {
        "CAISO": gridstatus.CAISO,
        "ERCOT": gridstatus.Ercot,
        "PJM": gridstatus.PJM,
        "MISO": gridstatus.MISO,
        "NYISO": gridstatus.NYISO,
        "ISONE": gridstatus.ISONE,
        "SPP": gridstatus.SPP,
    }

    iso_upper = iso.upper()
    if iso_upper not in iso_map:
        raise ValueError(
            f"ISO '{iso}' not supported. Choose from: {list(iso_map.keys())}"
        )

    print(f"Fetching {iso_upper} day-ahead prices from {start_date} to {end_date}...")
    iso_obj = iso_map[iso_upper]()

    # Fetch day-ahead LMP data
    df = iso_obj.get_lmp(
        start=start_date,
        end=end_date,
        market="DAY_AHEAD_HOURLY",
        locations=[node] if iso_upper == "CAISO" else None,
    )

    # Standardize column names
    # gridstatus returns different column names per ISO
    price_col = None
    for col in ["LMP", "lmp", "Price", "price", "SPP"]:
        if col in df.columns:
            price_col = col
            break

    if price_col is None:
        # Use the first numeric column that looks like a price
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        price_col = numeric_cols[0] if len(numeric_cols) > 0 else None

    if price_col is None:
        raise ValueError(f"Could not identify price column in data: {df.columns.tolist()}")

    # Find the datetime column
    time_col = None
    for col in ["Time", "Interval Start", "Interval End", "datetime", "timestamp"]:
        if col in df.columns:
            time_col = col
            break

    if time_col is None and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        time_col = df.columns[0]

    if time_col is None:
        raise ValueError(f"Could not identify time column in data: {df.columns.tolist()}")

    # Build standardized DataFrame
    result = pd.DataFrame()
    result["datetime"] = pd.to_datetime(df[time_col])
    result["price_mwh"] = df[price_col].values
    result["hour"] = result["datetime"].dt.hour
    result["day_of_week"] = result["datetime"].dt.dayofweek
    result["month"] = result["datetime"].dt.month
    result["is_weekend"] = result["day_of_week"] >= 5

    # Add day_index
    result["date"] = result["datetime"].dt.date
    unique_dates = sorted(result["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    result["day_index"] = result["date"].map(date_to_idx)
    result.drop(columns=["date"], inplace=True)

    # Sort and reset index
    result.sort_values("datetime", inplace=True)
    result.reset_index(drop=True, inplace=True)

    print(f"Loaded {len(result)} rows, {len(unique_dates)} days, "
          f"price range: ${result['price_mwh'].min():.1f}–${result['price_mwh'].max():.1f}/MWh")

    return result


# ============================================================
# Train/Test Splitting
# ============================================================
def train_test_split_prices(
    prices_df: pd.DataFrame,
    cfg: Config = DEFAULT_CONFIG,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split price data chronologically for ML/DL training.

    Uses chronological split (not random) because price data is
    a time series — future data should not leak into training.

    Args:
        prices_df: DataFrame from generate_synthetic_prices() or load_realistic_prices()
        cfg:       Config with train_ratio

    Returns:
        (train_df, test_df) tuple
    """
    n_days = prices_df["day_index"].max() + 1
    split_day = int(n_days * cfg.price.train_ratio)

    train_df = prices_df[prices_df["day_index"] < split_day].copy()
    test_df = prices_df[prices_df["day_index"] >= split_day].copy()

    print(f"Train: {len(train_df)} rows ({split_day} days), "
          f"Test: {len(test_df)} rows ({n_days - split_day} days)")

    return train_df, test_df


# ============================================================
# Visualization Helpers
# ============================================================
def plot_price_profile(prices_df: pd.DataFrame, num_days: int = 7):
    """Plot price profiles for visual inspection.

    Args:
        prices_df: DataFrame from generate_synthetic_prices()
        num_days:  Number of days to overlay
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overlay several daily profiles
    ax1 = axes[0]
    for day in range(min(num_days, prices_df["day_index"].max() + 1)):
        day_data = prices_df[prices_df["day_index"] == day]
        label = f"Day {day}" if day < 3 else None
        alpha = 0.7 if day < 3 else 0.2
        ax1.plot(day_data["hour"].values, day_data["price_mwh"].values,
                alpha=alpha, label=label, linewidth=1.0)

    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Price ($/MWh)")
    ax1.set_title("Daily Price Profiles (Overlaid)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 23)

    # Right: average profile with std band
    ax2 = axes[1]
    hourly = prices_df.groupby("hour")["price_mwh"]
    mean_prices = hourly.mean()
    std_prices = hourly.std()

    ax2.plot(mean_prices.index, mean_prices.values, "b-", linewidth=2, label="Mean")
    ax2.fill_between(mean_prices.index,
                     mean_prices.values - std_prices.values,
                     mean_prices.values + std_prices.values,
                     alpha=0.2, color="blue", label="±1 Std Dev")
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Price ($/MWh)")
    ax2.set_title("Average Daily Price Profile")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 23)

    # Annotate key periods
    ax2.axvspan(0, 6, alpha=0.05, color="blue", label="_Off-peak")
    ax2.axvspan(10, 15, alpha=0.05, color="green", label="_Solar dip")
    ax2.axvspan(17, 21, alpha=0.05, color="red", label="_Peak")
    ax2.text(3, mean_prices.max() * 0.9, "Off-peak", ha="center", fontsize=8, color="blue")
    ax2.text(12.5, mean_prices.min() * 1.1, "Solar dip", ha="center", fontsize=8, color="green")
    ax2.text(19, mean_prices.max() * 0.95, "Peak", ha="center", fontsize=8, color="red")

    plt.tight_layout()
    plt.savefig("price_profiles.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved price_profiles.png")


def plot_ev_schedules(schedules: List[EVSchedule], cfg: Config = DEFAULT_CONFIG):
    """Visualize EV arrival/departure schedule as a Gantt chart.

    Args:
        schedules: List of EVSchedule from generate_ev_schedules()
        cfg:       Config for time resolution
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(14, max(6, len(schedules) * 0.35)))

    for ev in schedules:
        # Connection bar
        y = ev.ev_id
        x_start = ev.arrival_step * cfg.time.dt_hours + cfg.time.start_hour
        width = (ev.departure_step - ev.arrival_step) * cfg.time.dt_hours

        # Color by urgency: energy needed / time available
        if ev.hours_available > 0:
            urgency = ev.energy_needed_kwh / (ev.max_charge_kw * ev.hours_available)
        else:
            urgency = 1.0
        color = plt.cm.RdYlGn(1.0 - min(urgency, 1.0))

        ax.barh(y, width, left=x_start, height=0.7, color=color,
                edgecolor="gray", linewidth=0.5, alpha=0.8)

        # Label with SoC info
        ax.text(x_start + width / 2, y,
                f"{ev.arrival_soc:.0%}→{ev.target_soc:.0%}",
                ha="center", va="center", fontsize=7, fontweight="bold")

    # Format axes
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("EV ID")
    ax.set_yticks(range(len(schedules)))
    ax.set_yticklabels([f"EV{s.ev_id:02d}" for s in schedules], fontsize=8)

    # Set x-axis to show hours of day properly
    ax.set_xlim(cfg.time.start_hour, cfg.time.start_hour + cfg.time.simulation_hours)
    hours = np.arange(cfg.time.start_hour,
                      cfg.time.start_hour + cfg.time.simulation_hours + 1, 2)
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{int(h % 24):02d}:00" for h in hours], fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)

    ax.set_title("EV Fleet Schedule (color = charging urgency: green=relaxed, red=tight)")
    ax.invert_yaxis()

    # Legend
    green_patch = mpatches.Patch(color=plt.cm.RdYlGn(0.8), label="Relaxed (plenty of time)")
    yellow_patch = mpatches.Patch(color=plt.cm.RdYlGn(0.5), label="Moderate")
    red_patch = mpatches.Patch(color=plt.cm.RdYlGn(0.1), label="Tight (barely feasible)")
    ax.legend(handles=[green_patch, yellow_patch, red_patch],
             loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig("ev_schedules.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved ev_schedules.png")


# ============================================================
# Main: Demo data generation
# ============================================================
if __name__ == "__main__":
    cfg = Config()
    print(cfg.summary())
    print()

    # Generate synthetic prices
    print("Generating synthetic electricity prices...")
    prices = generate_synthetic_prices(cfg)
    print(f"Generated {len(prices)} hourly prices over {cfg.price.num_days_history} days")
    print(f"Price stats: mean=${prices['price_mwh'].mean():.1f}, "
          f"min=${prices['price_mwh'].min():.1f}, "
          f"max=${prices['price_mwh'].max():.1f}/MWh")
    print()

    # Generate EV schedules
    print("Generating EV fleet schedules...")
    schedules = generate_ev_schedules(cfg)
    for ev in schedules[:5]:
        print(f"  {ev}")
    print(f"  ... ({len(schedules)} total)")
    print()

    # Extract one day's price curve for simulation
    print("Extracting Day 0 price curve at 15-min resolution...")
    day0_prices = get_daily_price_curve(prices, day_index=0, cfg=cfg)
    print(f"Shape: {day0_prices.shape}, "
          f"range: ${day0_prices.min():.1f}–${day0_prices.max():.1f}/MWh")
    print()

    # Train/test split
    train_df, test_df = train_test_split_prices(prices, cfg)

    # Visualize (if matplotlib is available)
    try:
        print("\nPlotting...")
        plot_price_profile(prices, num_days=10)
        plot_ev_schedules(schedules, cfg)
    except ImportError:
        print("matplotlib not available — skipping plots")
