#!/usr/bin/env python3
"""Temperature forecaster for the current Tempestas/Observo schema.

The model predicts ``Ambient_Temperature_C`` from recent ambient temperature
and relative humidity.  Pressure is deliberately not a required feature so the
model remains usable while the pressure sensor is absent and does not silently
substitute fabricated pressure values.
"""

import csv
import logging
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

DEFAULT_FEATURE_COLS = ["Ambient_Temperature_C", "DHT22_Humidity_percent"]
DEFAULT_TARGET_COL = "Ambient_Temperature_C"


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
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class _SeqDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_length: int, target_col: int):
        self.data = data
        self.seq_length = seq_length
        self.target_col = target_col

    def __len__(self):
        return max(0, len(self.data) - self.seq_length)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + self.seq_length, self.target_col]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


class WeatherForecaster:
    """LSTM forecaster using the new canonical station measurements."""

    def __init__(
        self,
        master_file: str = None,
        data: pd.DataFrame = None,
        hidden_dim: int = 96,
        num_layers: int = 2,
        dropout: float = 0.15,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        seq_length: int = 120,
        feature_cols: list = None,
        target_col: str = DEFAULT_TARGET_COL,
        device=None,
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
        self.feature_cols = list(feature_cols or DEFAULT_FEATURE_COLS)
        self.target_col = target_col

        if self.target_col not in self.feature_cols:
            raise ValueError(
                f"target_col {self.target_col!r} must be present in feature_cols"
            )

        raw = data.copy() if data is not None else self.load_master_data()
        raw = self._prepare_feature_frame(raw)
        if len(raw) < 3:
            raise ValueError("Not enough valid weather rows to initialize forecaster")

        self._fit_scalers(raw)
        self._scaled = self._scale(raw)
        self.seq_length = min(int(seq_length), len(self._scaled) - 1)
        if self.seq_length < 2:
            raise ValueError("Not enough rows for a forecast sequence")
        self._target_idx = self.feature_cols.index(self.target_col)
        self._build_model()

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
            self.optimizer, mode="min", factor=0.5, patience=4
        )

    def _prepare_feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing forecast feature column(s): {missing}")

        out = df.copy()
        for col in self.feature_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=self.feature_cols).reset_index(drop=True)
        return out

    def _fit_scalers(self, df: pd.DataFrame):
        self.scalers = {}
        for col in self.feature_cols:
            scaler = StandardScaler()
            scaler.fit(df[[col]].to_numpy(dtype=np.float64))
            self.scalers[col] = scaler

    def _scale(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for col in self.feature_cols:
            vals = df[[col]].to_numpy(dtype=np.float32)
            cols.append(self.scalers[col].transform(vals))
        return np.hstack(cols).astype(np.float32)

    def _scale_raw(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim != 2 or arr.shape[1] != len(self.feature_cols):
            raise ValueError(
                f"Expected array of shape (N, {len(self.feature_cols)}), got {arr.shape}"
            )
        if not np.isfinite(arr).all():
            raise ValueError("Forecast input contains NaN or infinite values")

        out = np.empty_like(arr, dtype=np.float32)
        for i, col in enumerate(self.feature_cols):
            out[:, i] = self.scalers[col].transform(arr[:, [i]]).ravel()
        return out

    def _inv_scale_target(self, scaled_vals: np.ndarray) -> np.ndarray:
        return self.scalers[self.target_col].inverse_transform(
            scaled_vals.reshape(-1, 1)
        ).ravel()

    def load_master_data(self) -> pd.DataFrame:
        """Load canonical data, resample to one minute, and fill only short gaps.

        Timestamps are parsed as UTC.  Interpolation is limited to ten minutes;
        longer outages are left missing and then excluded rather than being
        turned into invented weather.
        """
        if not self.master_file or not os.path.exists(self.master_file):
            raise FileNotFoundError(self.master_file or "master_file not specified")

        df = pd.read_csv(self.master_file, on_bad_lines="warn")
        if "Timestamp" not in df.columns:
            raise ValueError("Master weather data has no Timestamp column")

        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")

        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Master weather data missing forecast columns: {missing}")

        numeric = df[self.feature_cols].apply(pd.to_numeric, errors="coerce")
        numeric.index = pd.DatetimeIndex(df["Timestamp"])
        numeric = numeric.resample("1min").mean()
        numeric = numeric.interpolate(
            method="time", limit=10, limit_direction="both", limit_area="inside"
        )
        numeric = numeric.dropna(subset=self.feature_cols)
        return numeric.reset_index().rename(columns={"index": "Timestamp"})

    def train_model(
        self,
        epochs: int = 25,
        loss_csv_path: str = "training_loss.csv",
        final_loss_csv_path: str = "final_losses.csv",
    ):
        dataset = _SeqDataset(self._scaled, self.seq_length, self._target_idx)
        if len(dataset) < 1:
            raise ValueError("Not enough sequence samples to train forecast model")

        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
        )
        final_epoch_loss = np.nan

        with open(loss_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Loss"])
            for epoch in range(int(epochs)):
                self.model.train()
                total_loss = 0.0
                batches = 0
                for bx, by in loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    self.optimizer.zero_grad()
                    pred = self.model(bx).squeeze(-1)
                    loss = self.criterion(pred, by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    self.optimizer.step()
                    total_loss += loss.item()
                    batches += 1

                final_epoch_loss = total_loss / max(1, batches)
                logging.info(
                    "Forecast epoch %d/%d loss=%.8f",
                    epoch + 1,
                    epochs,
                    final_epoch_loss,
                )
                writer.writerow([epoch + 1, final_epoch_loss])
                self.scheduler.step(final_epoch_loss)

        with open(final_loss_csv_path, "a", newline="") as f:
            csv.writer(f).writerow([final_epoch_loss])

    def predict_future(self, recent_sequence: np.ndarray, steps_ahead: int = 60) -> np.ndarray:
        """Autoregressively predict future ambient temperature values."""
        raw = np.asarray(recent_sequence, dtype=np.float32).copy()
        if len(raw) < self.seq_length:
            raise ValueError(
                f"Need at least {self.seq_length} recent rows; received {len(raw)}"
            )
        raw = raw[-self.seq_length :]
        if not np.isfinite(raw).all():
            raise ValueError("Recent forecast sequence contains missing values")

        self.model.eval()
        predictions = []
        for _ in range(int(steps_ahead)):
            scaled = self._scale_raw(raw)
            tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                scaled_pred = self.model(tensor).item()
            temp_pred = float(self._inv_scale_target(np.array([scaled_pred]))[0])
            predictions.append(temp_pred)

            # For exogenous variables (currently humidity), persistence is used.
            # Only the predicted target temperature is advanced.
            next_row = raw[-1].copy()
            next_row[self._target_idx] = temp_pred
            raw = np.vstack((raw[1:], next_row))

        return np.asarray(predictions)

    def save_model(self, model_path: str):
        torch.save(
            {
                "schema_version": 2,
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
        logging.info("Forecast model saved to %s", model_path)

    def load_model(self, model_path: str):
        ckpt = torch.load(model_path, map_location=self.device)
        ckpt_features = ckpt.get("feature_cols")
        ckpt_target = ckpt.get("target_col")
        if ckpt.get("schema_version") != 2:
            raise ValueError("Forecast checkpoint is from the old weather schema")
        if ckpt_features != self.feature_cols or ckpt_target != self.target_col:
            raise ValueError(
                "Forecast checkpoint feature schema does not match current model: "
                f"checkpoint={ckpt_features}/{ckpt_target}, "
                f"current={self.feature_cols}/{self.target_col}"
            )

        self.hidden_dim = ckpt["hidden_dim"]
        self.num_layers = ckpt["num_layers"]
        self.dropout = ckpt["dropout"]
        self.learning_rate = ckpt["learning_rate"]
        self.seq_length = ckpt["seq_length"]
        self.scalers = ckpt["scalers"]
        self._target_idx = self.feature_cols.index(self.target_col)
        self._build_model()
        self.model.load_state_dict(ckpt["model_state_dict"])
        try:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as exc:
            logging.warning("Could not restore optimizer state: %s", exc)
        logging.info("Forecast model loaded from %s", model_path)

    @staticmethod
    def infer_timestamps(last_timestamp, steps_ahead: int, interval_seconds: float):
        return [
            last_timestamp + timedelta(seconds=interval_seconds * i)
            for i in range(1, int(steps_ahead) + 1)
        ]

    @staticmethod
    def save_predictions_to_csv(predictions, future_timestamps, output_file: str):
        pd.DataFrame(
            {"Timestamp": future_timestamps, "Predicted_Temperature": predictions}
        ).to_csv(output_file, index=False)
        logging.info("Predictions saved to %s", output_file)

    @staticmethod
    def plot_training_loss(file_path="training_loss.csv", output_path="training_loss_plot.png"):
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                return
            fig, ax = __import__("matplotlib.pyplot", fromlist=["plt"]).subplots(figsize=(8, 6))
            ax.plot(df["Epoch"], df["Loss"], marker="o")
            ax.set_title("Forecast training loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Huber loss")
            ax.set_yscale("log")
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            fig.tight_layout()
            fig.savefig(output_path)
            __import__("matplotlib.pyplot", fromlist=["plt"]).close(fig)
        except Exception as exc:
            logging.warning("plot_training_loss failed: %s", exc)

    @staticmethod
    def plot_final_losses(file_path="final_losses.csv", output_path="final_losses_plot.png"):
        try:
            with open(file_path) as f:
                losses = [float(row[0]) for row in csv.reader(f) if row]
            if not losses:
                return
            plt = __import__("matplotlib.pyplot", fromlist=["plt"])
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(range(1, len(losses) + 1), losses, marker="o")
            ax.set_yscale("log")
            ax.set_title("Final forecast loss across runs")
            ax.set_xlabel("Run")
            ax.set_ylabel("Huber loss")
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            fig.tight_layout()
            fig.savefig(output_path)
            plt.close(fig)
        except Exception as exc:
            logging.warning("plot_final_losses failed: %s", exc)
