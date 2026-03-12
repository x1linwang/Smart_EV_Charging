"""
dl_forecaster.py — Deep Learning Price Forecasting (Pillar 2)
==============================================================

Uses LSTM (Long Short-Term Memory) networks to forecast electricity
prices from raw sequential price data. Unlike the ML approach, the
LSTM learns its own feature representations from the sequence.

WHAT THIS MODULE DOES
---------------------
Given a long time series of hourly electricity prices, it trains an LSTM
neural network to predict the next hour's price from the *most recent
seq_len hours* (default: 168 hours = 1 week). The model learns temporal
dependencies automatically — no hand-crafted features required.

HOW THE LSTM WORKS HERE
-----------------------
The input to the LSTM is a sliding window of 168 normalized price values,
shaped as (batch_size, 168, 1). The LSTM processes this sequence step-by-step,
maintaining a hidden state that accumulates information about past prices.
The final hidden state is passed through two fully connected layers to produce
a single predicted price (the next hour).

  Window: [p_{t-167}, p_{t-166}, ..., p_{t-1}, p_t]  →  prediction: p_{t+1}

For multi-step forecasting (predict the next 24 hours), the model is run
autoregressively: each prediction is fed back as the newest input.

IMPORTANT NORMALIZATION NOTE
-----------------------------
Prices are z-score normalized using the *training set* mean and standard
deviation. At inference time, the same statistics must be used to normalize
inputs and denormalize outputs. These stats are saved in the lstm_dict.

ML vs DL COMPARISON
--------------------
On structured price data with known seasonal patterns, XGBoost with good
feature engineering often *matches or beats* LSTM. LSTM shines when:
  - Patterns are too complex to manually engineer features for
  - The sequence length matters (LSTM captures long-range dependencies)
  - Multi-variate inputs are available (price + weather + grid signals)

STUDENT WORK
------------
Students improve the LSTM in main.ipynb by subclassing PriceLSTM or
modifying hyperparameters. The .py file itself is provided infrastructure.

IEOR E4010: AI for Operations Research and Financial Engineering
Columbia University, Spring 2026
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List
import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import Config, DEFAULT_CONFIG


# ============================================================
# Dataset
# ============================================================
class PriceSequenceDataset(Dataset):
    """PyTorch Dataset for price sequence windows.

    Each sample is a window of `seq_len` consecutive prices,
    and the target is the next price after the window.

    Args:
        prices:  1D numpy array of hourly prices
        seq_len: Number of time steps in each input sequence
    """

    def __init__(self, prices: np.ndarray, seq_len: int = 168):
        self.prices = prices.astype(np.float32)
        self.seq_len = seq_len

        # Normalize prices
        self.mean = float(np.mean(self.prices))
        self.std = float(np.std(self.prices)) + 1e-8
        self.prices_norm = (self.prices - self.mean) / self.std

    def __len__(self) -> int:
        return len(self.prices) - self.seq_len - 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.prices_norm[idx: idx + self.seq_len]
        y = self.prices_norm[idx + self.seq_len]
        return torch.tensor(x).unsqueeze(-1), torch.tensor(y)

    def denormalize(self, y_norm: np.ndarray) -> np.ndarray:
        """Convert normalized predictions back to original scale."""
        return y_norm * self.std + self.mean


# ============================================================
# LSTM Model
# ============================================================
class PriceLSTM(nn.Module):
    """Baseline LSTM model for electricity price forecasting.

    Architecture:
        Input (batch, seq_len, 1) → LSTM (num_layers) → Last time step
        → Linear(hidden → hidden//2) → ReLU → Dropout → Linear(hidden//2 → 1)
        → Output scalar (next-hour price in normalized units)

    Why this architecture?
    ----------------------
    - Two LSTM layers allow the network to learn both low-level patterns
      (hour-to-hour changes) and high-level structures (daily/weekly cycles).
    - Dropout between layers provides regularization to prevent overfitting
      on the relatively small price dataset (~8,760 samples for 1 year).
    - The two-layer FC head transforms the LSTM's rich hidden representation
      down to a single scalar prediction.

    Hyperparameter sensitivity:
    - hidden_size: Larger = more capacity to model complex patterns, but
      slower training and more data needed. 64 is a good default.
    - num_layers: More layers = deeper temporal representation. 2 is
      usually sufficient; 3+ rarely helps without more data.
    - dropout: Higher = more regularization. Should match dataset size.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Tensor of shape (batch_size,) with predicted prices
        """
        # LSTM output: (batch, seq_len, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last time step's output
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)

        # Fully connected layers
        output = self.fc(last_hidden).squeeze(-1)  # (batch,)

        return output


# ============================================================
# Training
# ============================================================
def train_lstm(
    train_prices: np.ndarray,
    val_prices: Optional[np.ndarray] = None,
    cfg: Config = DEFAULT_CONFIG,
    verbose: bool = True,
    model_class=None,
) -> Dict[str, Any]:
    """Train an LSTM model for price forecasting.

    TRAINING PROCEDURE
    ------------------
    1. Normalize prices using training set statistics (z-score).
    2. Create sliding window datasets: each sample is
       (prices[i:i+seq_len], prices[i+seq_len]).
    3. Train with Adam optimizer + MSE loss.
    4. Use ReduceLROnPlateau to halve LR when val loss plateaus.
    5. Apply gradient clipping (max_norm=1.0) to prevent exploding gradients.
    6. Apply early stopping (patience=10 epochs) on validation loss.
    7. Keep the best model checkpoint (lowest val loss).

    NORMALIZATION NOTE
    ------------------
    The returned lstm_dict stores norm_mean and norm_std. These MUST be used
    when calling evaluate_lstm() or predict_next_24h_lstm() — the model
    was trained on normalized data and outputs normalized predictions.

    Args:
        train_prices: 1D array of training hourly prices (e.g., ~7000 samples)
        val_prices:   1D array of validation prices (uses train normalization)
        cfg:          Configuration (controls hidden_size, num_layers, etc.)
        verbose:      Print training progress every 5 epochs
        model_class:  Optional custom model class (must be a subclass of
                      nn.Module with the same input/output signature as
                      PriceLSTM). If None, uses the default PriceLSTM.
                      Students pass their custom architecture here.

    Returns:
        Dict with keys:
            model      — trained PyTorch model (on CPU/GPU, eval mode)
            device     — torch.device used for training
            history    — {train_loss, val_loss, val_mae} lists
            norm_mean  — float, training price mean
            norm_std   — float, training price std
            seq_len    — int, input sequence length
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Training on device: {device}")

    seq_len = cfg.forecast.lstm_sequence_length

    # Create datasets
    train_dataset = PriceSequenceDataset(train_prices, seq_len=seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.forecast.lstm_batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_loader = None
    val_dataset = None
    if val_prices is not None and len(val_prices) > seq_len + 1:
        val_dataset = PriceSequenceDataset(val_prices, seq_len=seq_len)
        # Use training normalization stats for validation
        val_dataset.mean = train_dataset.mean
        val_dataset.std = train_dataset.std
        val_dataset.prices_norm = (val_dataset.prices - train_dataset.mean) / train_dataset.std
        val_loader = DataLoader(val_dataset, batch_size=cfg.forecast.lstm_batch_size, shuffle=False)

    # Create model — use custom class if provided, otherwise default PriceLSTM
    ModelClass = model_class if model_class is not None else PriceLSTM
    model = ModelClass(
        input_size=1,
        hidden_size=cfg.forecast.lstm_hidden_size,
        num_layers=cfg.forecast.lstm_num_layers,
        dropout=cfg.forecast.lstm_dropout,
    ).to(device)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.forecast.lstm_learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=False
    )
    criterion = nn.MSELoss()

    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_mae": []}
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    patience_limit = 10

    for epoch in range(cfg.forecast.lstm_epochs):
        # --- Train ---
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)

        # --- Validate ---
        val_loss = None
        val_mae = None
        if val_loader is not None:
            model.eval()
            val_losses = []
            val_preds = []
            val_actuals = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    val_losses.append(loss.item())
                    val_preds.extend(y_pred.cpu().numpy())
                    val_actuals.extend(y_batch.cpu().numpy())

            val_loss = np.mean(val_losses)
            # Denormalize for MAE
            val_preds_denorm = np.array(val_preds) * train_dataset.std + train_dataset.mean
            val_actuals_denorm = np.array(val_actuals) * train_dataset.std + train_dataset.mean
            val_mae = np.mean(np.abs(val_preds_denorm - val_actuals_denorm))

            history["val_loss"].append(val_loss)
            history["val_mae"].append(val_mae)

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break

        if verbose and (epoch + 1) % 5 == 0:
            msg = f"  Epoch {epoch+1:3d}/{cfg.forecast.lstm_epochs} | Train Loss: {avg_train_loss:.6f}"
            if val_loss is not None:
                msg += f" | Val Loss: {val_loss:.6f} | Val MAE: ${val_mae:.2f}/MWh"
            print(msg)

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()

    return {
        "model": model,
        "device": device,
        "history": history,
        "norm_mean": train_dataset.mean,
        "norm_std": train_dataset.std,
        "seq_len": seq_len,
    }


# ============================================================
# Evaluation
# ============================================================
def evaluate_lstm(
    lstm_dict: Dict[str, Any],
    test_prices: np.ndarray,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate LSTM model on test data.

    Args:
        lstm_dict:    Output from train_lstm()
        test_prices:  1D array of test prices
        verbose:      Print metrics

    Returns:
        Dict with mae, rmse, r2, predictions, actuals
    """
    model = lstm_dict["model"]
    device = lstm_dict["device"]
    seq_len = lstm_dict["seq_len"]
    mean = lstm_dict["norm_mean"]
    std = lstm_dict["norm_std"]

    # Create test sequences
    prices_norm = (test_prices.astype(np.float32) - mean) / std

    predictions = []
    actuals = []

    model.eval()
    with torch.no_grad():
        for i in range(len(prices_norm) - seq_len - 1):
            x = torch.tensor(prices_norm[i: i + seq_len]).unsqueeze(0).unsqueeze(-1).to(device)
            y_pred = model(x).cpu().numpy()[0]
            predictions.append(y_pred * std + mean)
            actuals.append(test_prices[i + seq_len])

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    r2 = 1 - np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2)

    if verbose:
        print(f"\nLSTM Test Results:")
        print(f"  MAE:  ${mae:.2f}/MWh")
        print(f"  RMSE: ${rmse:.2f}/MWh")
        print(f"  R²:   {r2:.4f}")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "predictions": predictions,
        "actuals": actuals,
    }


def predict_next_24h_lstm(
    lstm_dict: Dict[str, Any],
    recent_prices: np.ndarray,
) -> np.ndarray:
    """Forecast next 24 hours using the LSTM model.

    Args:
        lstm_dict:     Output from train_lstm()
        recent_prices: Last seq_len+ hours of prices

    Returns:
        np.ndarray of shape (24,) with predicted prices
    """
    model = lstm_dict["model"]
    device = lstm_dict["device"]
    seq_len = lstm_dict["seq_len"]
    mean = lstm_dict["norm_mean"]
    std = lstm_dict["norm_std"]

    # Normalize
    prices_norm = (recent_prices.astype(np.float32) - mean) / std
    buffer = list(prices_norm[-seq_len:])

    predictions = []
    model.eval()
    with torch.no_grad():
        for _ in range(24):
            x = torch.tensor(buffer[-seq_len:]).unsqueeze(0).unsqueeze(-1).to(device)
            y_pred = model(x).cpu().numpy()[0]
            predictions.append(float(y_pred * std + mean))
            buffer.append(y_pred)

    return np.maximum(0, np.array(predictions))


# ============================================================
# Model Persistence (Save / Load)
# ============================================================
def save_lstm_model(lstm_dict: Dict[str, Any], path: str = "submission/lstm_model.pth") -> None:
    """Save a trained LSTM model to disk.

    Saves the model's state_dict (weights) along with normalization statistics
    and architecture hyperparameters needed to reconstruct the model at
    inference time. Uses torch.save() which stores a Python dict.

    IMPORTANT: This saves weights only, not the model class definition.
    To load the model, you must use the same model class (PriceLSTM or your
    custom subclass).

    Args:
        lstm_dict: Output from train_lstm()
        path:      Save path (e.g., "submission/lstm_model.pth")
    """
    import os

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    model = lstm_dict["model"]
    save_data = {
        "model_state_dict": model.state_dict(),
        "norm_mean": lstm_dict["norm_mean"],
        "norm_std": lstm_dict["norm_std"],
        "seq_len": lstm_dict["seq_len"],
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "history": lstm_dict.get("history", {}),
    }
    torch.save(save_data, path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"LSTM model saved to '{path}' ({size_mb:.2f} MB)")
    print(f"  Architecture: hidden={model.hidden_size}, layers={model.num_layers}")
    print(f"  Sequence length: {lstm_dict['seq_len']} hours")
    print(f"  Normalization: mean={lstm_dict['norm_mean']:.2f}, std={lstm_dict['norm_std']:.2f}")


def load_lstm_model(
    path: str = "submission/lstm_model.pth",
    model_class=None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a saved LSTM model from disk.

    Args:
        path:        Path to the .pth file saved by save_lstm_model()
        model_class: Model class to use for reconstruction (defaults to
                     PriceLSTM). Must match the class used when saving.
        device:      Device string ('cpu', 'cuda') or None for auto-detect.

    Returns:
        lstm_dict compatible with evaluate_lstm() and predict_next_24h_lstm()
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    save_data = torch.load(path, map_location=device)

    ModelClass = model_class if model_class is not None else PriceLSTM
    model = ModelClass(
        input_size=1,
        hidden_size=save_data["hidden_size"],
        num_layers=save_data["num_layers"],
        dropout=0.0,  # Dropout disabled at inference
    ).to(device)
    model.load_state_dict(save_data["model_state_dict"])
    model.eval()

    print(f"LSTM model loaded from '{path}'")
    print(f"  Architecture: hidden={save_data['hidden_size']}, layers={save_data['num_layers']}")

    return {
        "model": model,
        "device": device,
        "history": save_data.get("history", {}),
        "norm_mean": save_data["norm_mean"],
        "norm_std": save_data["norm_std"],
        "seq_len": save_data["seq_len"],
    }


# ============================================================
# Visualization
# ============================================================
def plot_dl_results(
    lstm_results: Dict[str, Any],
    ml_results: Optional[Dict[str, Dict[str, float]]] = None,
    history: Optional[Dict] = None,
    num_days: int = 7,
):
    """Plot DL forecasting results, optionally comparing with ML.

    Args:
        lstm_results: Output from evaluate_lstm()
        ml_results:   Output from ml_forecaster.evaluate_ml_models() (optional)
        history:      Training history dict (optional)
        num_days:     Days to plot
    """
    import matplotlib.pyplot as plt

    n_panels = 3 if history else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    fig.suptitle("DL (LSTM) Price Forecasting Results", fontsize=14)

    hours = num_days * 24

    # Panel 1: Time series comparison
    ax1 = axes[0]
    ax1.plot(lstm_results["actuals"][:hours], "k-", linewidth=0.8, alpha=0.7, label="Actual")
    ax1.plot(lstm_results["predictions"][:hours], linewidth=0.8,
            color="#7C3AED", label=f"LSTM (MAE=${lstm_results['mae']:.1f})")
    if ml_results:
        for name, res in ml_results.items():
            ax1.plot(res["predictions"][:hours], linewidth=0.8, alpha=0.6,
                    label=f"{name} (MAE=${res['mae']:.1f})")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Price ($/MWh)")
    ax1.set_title("LSTM vs. ML Predictions")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Error comparison
    ax2 = axes[1]
    lstm_errors = lstm_results["predictions"] - lstm_results["actuals"]
    ax2.hist(lstm_errors, bins=50, alpha=0.6, color="#7C3AED",
            label=f"LSTM (σ={np.std(lstm_errors):.1f})")
    if ml_results:
        for name, res in ml_results.items():
            errors = res["predictions"][:len(lstm_errors)] - res["actuals"][:len(lstm_errors)]
            ax2.hist(errors, bins=50, alpha=0.4, label=f"{name} (σ={np.std(errors):.1f})")
    ax2.set_xlabel("Error ($/MWh)")
    ax2.set_title("Error Distribution: DL vs. ML")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Training curves
    if history:
        ax3 = axes[2]
        ax3.plot(history["train_loss"], label="Train Loss")
        if history.get("val_loss"):
            ax3.plot(history["val_loss"], label="Val Loss")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss (MSE)")
        ax3.set_title("Training Curves")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale("log")

    plt.tight_layout()
    plt.savefig("dl_forecasting_results.png", dpi=150, bbox_inches="tight")
    plt.show()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    from data_utils import generate_synthetic_prices, train_test_split_prices

    cfg = Config()
    # Use fewer epochs for quick test
    cfg.forecast.lstm_epochs = 20

    # Generate data
    print("Generating synthetic price data...")
    prices_df = generate_synthetic_prices(cfg)
    train_df, test_df = train_test_split_prices(prices_df, cfg)

    train_prices = train_df["price_mwh"].values
    test_prices = test_df["price_mwh"].values

    # Train
    print(f"\nTraining LSTM (seq_len={cfg.forecast.lstm_sequence_length}, "
          f"hidden={cfg.forecast.lstm_hidden_size}, layers={cfg.forecast.lstm_num_layers})...")
    lstm_dict = train_lstm(train_prices, val_prices=test_prices[:1000], cfg=cfg)

    # Evaluate
    lstm_results = evaluate_lstm(lstm_dict, test_prices)

    # Plot
    try:
        plot_dl_results(lstm_results, history=lstm_dict["history"])
    except ImportError:
        print("matplotlib not available — skipping plots")
