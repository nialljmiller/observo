#!/usr/bin/env python3
"""
Weather Forecaster — simple LSTM, sklearn scalers, same public API as before.
"""

import os
import csv
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)          # (batch, seq, hidden)
        return self.fc(out[:, -1, :])  # (batch, output_dim)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class _SeqDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_length: int, target_col: int):
        self.data = data
        self.seq_length = seq_length
        self.target_col = target_col

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + self.seq_length, self.target_col]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------

class WeatherForecaster:
    """
    Simple LSTM-based weather forecaster.

    Feature columns (passed to __init__ as feature_cols):
        default → ["DHT_Humidity_percent", "BMP_Temperature_C", "BMP_Pressure_hPa"]

    Target column (passed as target_col):
        default → "BMP_Temperature_C"

    Public API is backward-compatible with the previous version:
        train_model / save_model / load_model
        load_master_data
        predict_future / infer_timestamps / save_predictions_to_csv
        plot_training_loss / plot_final_losses
    """

    def __init__(
        self,
        master_file: str = None,
        data: pd.DataFrame = None,
        # Model hyper-parameters
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        # Training
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        # Sequence
        seq_length: int = 500,
        # Features
        feature_cols: list = None,
        target_col: str = "BMP_Temperature_C",
        # Misc
        device=None,
        # Legacy positional args (ignored but accepted for compatibility)
        input_dim=None,
        output_dim=None,
        target_seq_length=None,
    ):
        self.master_file = master_file
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        
        self.feature_cols = feature_cols or [
            "DHT_Humidity_percent",
            "BMP_Temperature_C",
            "BMP_Pressure_hPa",
        ]
        self.target_col = target_col

        # Load & pre-process data
        raw = data if data is not None else self.load_master_data()
        self._fit_scalers(raw)
        self._scaled = self._scale(raw)

        # seq_length must be < len(data) so at least one training sample exists
        self.seq_length = min(seq_length, len(self._scaled) - 1)
        self._target_idx = self.feature_cols.index(self.target_col)

        self._build_model()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self):
        self.model = _LSTMNet(
            input_dim=len(self.feature_cols),
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=1,
            dropout=self.dropout,
        ).to(self.device)
        self.criterion = nn.HuberLoss(delta=1.0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

    def _fit_scalers(self, df: pd.DataFrame):
        """Fit one StandardScaler per feature column."""
        self.scalers = {}
        for col in self.feature_cols:
            s = StandardScaler()
            valid = df[col].dropna().values.reshape(-1, 1)
            s.fit(valid)
            self.scalers[col] = s

    def _scale(self, df: pd.DataFrame) -> np.ndarray:
        """Return scaled numpy array, shape (N, n_features)."""
        cols = []
        for col in self.feature_cols:
            vals = df[col].values.reshape(-1, 1).astype(np.float32)
            vals = np.nan_to_num(vals, nan=0.0)
            cols.append(self.scalers[col].transform(vals))
        arr = np.hstack(cols).astype(np.float32)
        return arr

    def _scale_raw(self, arr: np.ndarray) -> np.ndarray:
        """Scale a raw (N, n_features) numpy array using stored scalers."""
        out = np.empty_like(arr, dtype=np.float32)
        for i, col in enumerate(self.feature_cols):
            col_vals = np.nan_to_num(arr[:, i].reshape(-1, 1), nan=0.0)
            out[:, i] = self.scalers[col].transform(col_vals).ravel()
        return out

    def _inv_scale_target(self, scaled_vals: np.ndarray) -> np.ndarray:
        """Inverse-scale predicted target values back to original units."""
        return self.scalers[self.target_col].inverse_transform(
            scaled_vals.reshape(-1, 1)
        ).ravel()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_master_data(self) -> pd.DataFrame:
        """
        Load, resample to 1-minute, interpolate gaps ≤ 2 h, return DataFrame.
        """
        df = pd.read_csv(self.master_file, on_bad_lines="skip")
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
        df = df.set_index("Timestamp").resample("1min").mean()
        max_gap = pd.Timedelta("2H")
        limit = int(max_gap / pd.Timedelta("1min"))
        df = df.interpolate(method="time", limit=limit, limit_direction="both").dropna()
        return df.reset_index()

    def train_model(
        self,
        epochs: int = 10,
        loss_csv_path: str = "training_loss.csv",
        final_loss_csv_path: str = "final_losses.csv",
    ):
        dataset = _SeqDataset(self._scaled, self.seq_length, self._target_idx)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        epoch_loss = 0.0

        with open(loss_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Loss"])

            for epoch in range(epochs):
                self.model.train()
                epoch_loss = 0.0
                for bx, by in loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    self.optimizer.zero_grad()
                    pred = self.model(bx).squeeze()
                    loss = self.criterion(pred, by)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()

                epoch_loss /= len(loader)
                logging.info(f"Epoch [{epoch+1}/{epochs}]  loss={epoch_loss:.8f}")
                writer.writerow([epoch + 1, epoch_loss])
                self.scheduler.step(epoch_loss)

        with open(final_loss_csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch_loss])

    def predict_future(self, recent_sequence: np.ndarray, steps_ahead: int = 6) -> np.ndarray:
        """
        Predict `steps_ahead` future temperature values.

        `recent_sequence` — raw (unscaled) array of shape (seq_length, n_features),
        columns in the same order as self.feature_cols.
        """
        self.model.eval()
        raw = recent_sequence.copy().astype(np.float32)

        predictions = []
        for _ in range(steps_ahead):
            scaled = self._scale_raw(raw)
            tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                scaled_pred = self.model(tensor).item()

            # Back to original scale
            temp_pred = self._inv_scale_target(np.array([scaled_pred]))[0]
            predictions.append(temp_pred)

            # Advance window: copy last row, update target column, drop oldest
            next_row = raw[-1].copy()
            next_row[self._target_idx] = temp_pred
            raw = np.vstack((raw[1:], next_row))

        return np.array(predictions)

    def save_model(self, model_path: str):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "seq_length": self.seq_length,
                "feature_cols": self.feature_cols,
                "target_col": self.target_col,
                "scalers": self.scalers,
            },
            model_path,
        )
        logging.info(f"Model saved to {model_path}")

    def load_model(self, model_path: str):
        ckpt = torch.load(model_path, map_location=self.device)
        self.hidden_dim = ckpt.get("hidden_dim", self.hidden_dim)
        self.num_layers = ckpt.get("num_layers", self.num_layers)
        self.dropout = ckpt.get("dropout", self.dropout)
        self.learning_rate = ckpt.get("learning_rate", self.learning_rate)
        self.seq_length = ckpt.get("seq_length", self.seq_length)
        self.feature_cols = ckpt.get("feature_cols", self.feature_cols)
        self.target_col = ckpt.get("target_col", self.target_col)
        self.scalers = ckpt["scalers"]
        self._target_idx = self.feature_cols.index(self.target_col)
        self._build_model()
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logging.info(f"Model loaded from {model_path}")

    # ------------------------------------------------------------------
    # Static helpers (unchanged interface)
    # ------------------------------------------------------------------

    @staticmethod
    def infer_timestamps(last_timestamp, steps_ahead: int, interval_seconds: float):
        return [
            last_timestamp + timedelta(seconds=interval_seconds * i)
            for i in range(1, steps_ahead + 1)
        ]

    @staticmethod
    def save_predictions_to_csv(predictions, future_timestamps, output_file: str):
        pd.DataFrame(
            {"Timestamp": future_timestamps, "Predicted_Temperature": predictions}
        ).to_csv(output_file, index=False)
        logging.info(f"Predictions saved to {output_file}")

    @staticmethod
    def plot_training_loss(file_path="training_loss.csv", output_path="training_loss_plot.png"):
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                return
            creation_date = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d")
            plt.figure(figsize=(8, 6))
            plt.plot(df["Epoch"], df["Loss"], marker="o")
            plt.title(f"Training Loss Per Epoch (file: {creation_date})")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.yscale("log")
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            logging.info(f"Training loss plot saved to {output_path}")
        except FileNotFoundError:
            logging.warning(f"{file_path} not found.")
        except Exception as e:
            logging.error(f"plot_training_loss: {e}")

    @staticmethod
    def plot_final_losses(file_path="final_losses.csv", output_path="final_losses_plot.png"):
        try:
            with open(file_path) as f:
                losses = [float(row[0]) for row in csv.reader(f) if row]
            if not losses:
                return
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, len(losses) + 1), losses, marker="o", color="blue")
            plt.yscale("log")
            plt.title("Final Losses Across Runs")
            plt.xlabel("Run")
            plt.ylabel("Loss (log scale)")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            logging.info(f"Final losses plot saved to {output_path}")
        except FileNotFoundError:
            logging.warning(f"{file_path} not found.")
        except Exception as e:
            logging.error(f"plot_final_losses: {e}")
