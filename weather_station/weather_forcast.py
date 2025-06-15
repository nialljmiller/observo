import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from datetime import timedelta
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class WeatherForecaster:
    def __init__(self, master_file=None, data=None, input_dim=3, hidden_dim=64, output_dim=1,
                 num_layers=2, learning_rate=0.001, batch_size=256, device=None, target_seq_length=1000):
        self.master_file = master_file
        self.data = data
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_seq_length = target_seq_length

        # Replaced the simple LSTM with a CNN+LSTM in WeatherLSTM
        self.model = self.WeatherLSTM(input_dim, hidden_dim, output_dim, num_layers).to(self.device)

        # HuberLoss is fine; it balances outliers vs. over-smoothing
        self.criterion = nn.HuberLoss(delta=1.0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3, verbose=True)

        # Preprocessing logic
        master_data = self.data if self.data is not None else self.load_master_data()
        self.data = self.process_input_data(
            master_data[["DHT_Humidity_percent", "BMP_Temperature_C", "BMP_Pressure_hPa"]].values
        )

    def debug_data(self, data, step_name):
        """
        Debug the data for NaN and extreme values after each preprocessing step.
        """
        print(f"Debugging data after {step_name}:")
        print(f"Shape: {data.shape}")
        print(f"Min: {np.nanmin(data)}")
        print(f"Max: {np.nanmax(data)}")
        print(f"NaN count: {np.isnan(data).sum()}")
        print(f"First row: {data[0]}")
        print(f"Last row: {data[-1]}")

    def process_input_data(self, data):
        data = self.validate_input_data(data)
        # data = self.debug_data(data, "raw input validation")

        data = self.smooth_data(data, window_size=10)
        # data = self.debug_data(data, "smoothing")

        # Hard scale temperature
        self.temp_min = np.min(data[:, 1])
        self.temp_max = np.max(data[:, 1])
        data[:, 1] = (
            (data[:, 1] - self.temp_min) / (self.temp_max - self.temp_min)
            if self.temp_max != self.temp_min
            else 0
        )

        # Hard scale humidity
        self.hum_min = np.min(data[:, 0])
        self.hum_max = np.max(data[:, 0])
        data[:, 0] = (
            (data[:, 0] - self.hum_min) / (self.hum_max - self.hum_min)
            if self.hum_max != self.hum_min
            else 0
        )

        # Hard scale pressure
        self.pres_min = np.min(data[:, 2])
        self.pres_max = np.max(data[:, 2])
        data[:, 2] = (
            (data[:, 2] - self.pres_min) / (self.pres_max - self.pres_min)
            if self.pres_max != self.pres_min
            else 0
        )

        # Add interaction terms
        humidity, temperature, pressure = data[:, 0], data[:, 1], data[:, 2]
        data = np.column_stack(
            (data, humidity * temperature, humidity * pressure, temperature * pressure)
        )

        # Add lag features
        lags = [5, 30, 300]
        df = pd.DataFrame(data)
        for lag in lags:
            lagged_data = df.shift(lag).bfill().values
            data = np.column_stack((data, lagged_data))

        # Add rate of change
        rate_of_change = np.diff(data, axis=0, prepend=data[0:1, :])
        data = np.column_stack((data, rate_of_change))

        # Final NaN cleaning
        if np.isnan(data).any():
            print("Warning: NaN detected in final processed data. Replacing with zeros...")
            data = np.nan_to_num(data, nan=0.0)

        # Update input_dim and seq_length
        self.input_dim = data.shape[1]
        # seq_length determines the length of the sequences used for training
        # and inference.  The previous behaviour divided the length of the
        # dataset by ``target_seq_length`` which often resulted in extremely
        # short (e.g. length 1) sequences when ``target_seq_length`` was set to
        # a large value.  This in turn caused the model to train on almost no
        # temporal context, leading to very poor predictions.  Instead we want
        # ``seq_length`` to be at most ``target_seq_length`` but never larger
        # than ``len(data) - 1`` so that at least one training sample exists.
        self.seq_length = min(len(data) - 1, self.target_seq_length)
        return data

    def validate_input_data(self, data):
        """
        Validate input data to ensure there are no NaN or extreme values.
        """
        if np.isnan(data).any():
            print("Warning: Input data contains NaN values. Attempting to fill...")
            data = pd.DataFrame(data).fillna(method='ffill').fillna(method='bfill').values
        if np.isinf(data).any():
            print("Warning: Input data contains infinite values. Clipping...")
            data = np.clip(
                data,
                a_min=np.finfo(np.float32).min,
                a_max=np.finfo(np.float32).max
            )
        return data

    def inverse_transform_predictions(self, predictions):
        """
        Reverse scaling for temperature predictions.
        """
        return predictions * (self.temp_max - self.temp_min) + self.temp_min


    def process_inference_data(self, data):
        """
        Process raw input data for inference, using stored scaling parameters and
        replicating the feature engineering performed during training.
        Assumes `data` is a NumPy array of shape (seq_length, 3).
        
        **IMPORTANT:** For consistent results, ensure that `data` has been interpolated.
        The easiest way is to pass in data obtained via load_master_data.
        """
        data = self.validate_input_data(data)
        data = self.smooth_data(data, window_size=10)
        
        # Scale using stored training parameters (do NOT recompute these!)
        data[:, 1] = (data[:, 1] - self.temp_min) / (self.temp_max - self.temp_min) if self.temp_max != self.temp_min else 0
        data[:, 0] = (data[:, 0] - self.hum_min) / (self.hum_max - self.hum_min) if self.hum_max != self.hum_min else 0
        data[:, 2] = (data[:, 2] - self.pres_min) / (self.pres_max - self.pres_min) if self.pres_max != self.pres_min else 0

        # Add interaction terms as in training
        humidity, temperature, pressure = data[:, 0], data[:, 1], data[:, 2]
        data = np.column_stack((data, humidity * temperature, humidity * pressure, temperature * pressure))

        # Add lag features (for lags [5, 30, 300])
        lags = [5, 30, 300]
        df = pd.DataFrame(data)
        for lag in lags:
            lagged_data = df.shift(lag).bfill().values
            data = np.column_stack((data, lagged_data))
        
        # Add rate of change (first difference)
        rate_of_change = np.diff(data, axis=0, prepend=data[0:1, :])
        data = np.column_stack((data, rate_of_change))
        return data


    def predict_future(self, recent_sequence, steps_ahead=6):
        """
        Predict future temperature values using inference preprocessing that matches training.
        """
        self.model.eval()
        future_predictions = []

        # If recent_sequence is a tensor, squeeze and convert to numpy
        if isinstance(recent_sequence, torch.Tensor):
            recent_sequence = recent_sequence.squeeze(0).cpu().numpy()

        # Use the training-consistent preprocessing for inference
        processed_sequence = self.process_inference_data(recent_sequence)
        # Convert to 3D tensor: shape (1, seq_length, feature_dim)
        processed_sequence = torch.tensor(processed_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        if torch.isnan(processed_sequence).any():
            print("Warning: Processed sequence contains NaN values. Attempting to fix...")
            processed_sequence = torch.nan_to_num(processed_sequence, nan=0.0)

        for _ in range(steps_ahead):
            with torch.no_grad():
                next_prediction = self.model(processed_sequence).item()
                future_predictions.append(next_prediction)

                # Update the sequence: use model's prediction to update the temperature feature (assumed index 1)
                last_timestep = processed_sequence[:, -1, :].clone()
                next_timestep = last_timestep.clone()
                next_timestep[0, 1] = next_prediction

                # Shift sequence and append the new timestep
                processed_sequence = torch.cat(
                    (processed_sequence[:, 1:, :], next_timestep.unsqueeze(1)), dim=1
                )

        return self.inverse_transform_predictions(np.array(future_predictions))



    @staticmethod
    def smooth_data(data, window_size=10):
        """
        Apply a moving average to smooth the data.
        """
        # Use a rolling mean to smooth the data (you can swap to a median if you want)
        smoothed_data = pd.DataFrame(data).rolling(window=window_size, min_periods=1, center=True).mean()
        return smoothed_data.values

    def load_master_data(self):
        """Load master data robustly from CSV, resample, and interpolate.
        
        This version:
          - Converts Timestamp to datetime and sorts data.
          - Resamples to a fixed frequency (default "1T"; change if needed).
          - Interpolates gaps up to a maximum (default: 2 hours).
          - Optionally applies a rolling median filter to reduce noise.
        """
        # Read and clean the CSV
        data = pd.read_csv(self.master_file, on_bad_lines='skip')
        data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
        data = data.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
        
        # Set Timestamp as index for resampling
        data.set_index("Timestamp", inplace=True)
        
        # Resample to a fixed frequency (adjust "1T" as needed)
        freq = "1T"
        data = data.resample(freq).mean()
        
        # Define maximum gap for interpolation (default: 2 hours)
        max_gap = pd.Timedelta("2H")
        limit = int(max_gap / pd.Timedelta(freq))
        
        # Interpolate only small gaps
        data = data.interpolate(method='time', limit=limit, limit_direction='both')
        
        # Drop remaining NaNs (from gaps too large to guess)
        data = data.dropna()
        
        # Optionally, apply a rolling median filter (uncomment if needed)
        # data = data.rolling(window=5, min_periods=1, center=True).median()
        
        # Reset index so Timestamp becomes a column again
        data = data.reset_index()
        return data


    ########################################################################
    # -- Drop-in replacement: CNN+LSTM model hidden behind same class name --
    ########################################################################
    class WeatherLSTM(nn.Module):
        """
        Hybrid CNN+LSTM to capture both short-term local patterns (CNN)
        and long-term dependencies (LSTM).
        """
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
            super().__init__()
            # Convolution layers to extract local/time-neighbor features
            self.cnn_out_channels = 32
            self.kernel_size = 3

            # 1D Convolution + BatchNorm
            self.conv1 = nn.Conv1d(
                in_channels=input_dim,
                out_channels=self.cnn_out_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2
            )
            self.bn1 = nn.BatchNorm1d(self.cnn_out_channels)

            self.conv2 = nn.Conv1d(
                in_channels=self.cnn_out_channels,
                out_channels=self.cnn_out_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2
            )
            self.bn2 = nn.BatchNorm1d(self.cnn_out_channels)

            self.relu = nn.ReLU()
            self.dropout_cnn = nn.Dropout(0.2)

            # LSTM to model temporal dependencies
            # We apply dropout between LSTM layers if num_layers > 1
            self.lstm = nn.LSTM(
                input_size=self.cnn_out_channels,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0.0
            )

            # Fully connected layer to map LSTM output to final prediction
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            """
            x shape: (batch, seq_len, input_dim)
            returns shape: (batch, output_dim)
            """
            # Permute to (batch, input_dim, seq_len) for Conv1d
            x = x.permute(0, 2, 1)

            # First conv
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            # Second conv
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)

            # Dropout to help regularize
            x = self.dropout_cnn(x)

            # Permute back for LSTM: (batch, seq_len, cnn_out_channels)
            x = x.permute(0, 2, 1)

            # LSTM
            lstm_out, _ = self.lstm(x)   # (batch, seq_len, hidden_dim)
            # Take last timestep
            last_out = lstm_out[:, -1, :]  # (batch, hidden_dim)

            # Map to final output
            out = self.fc(last_out)       # (batch, output_dim)
            return out

    ########################################################################

    def save_model(self, model_path):
        """Save the model and optimizer state to a file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'learning_rate': self.learning_rate
        }, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path):
        """
        Load the model and optimizer state from a file.
        """
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.input_dim = checkpoint['input_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        self.output_dim = checkpoint['output_dim']
        self.num_layers = checkpoint['num_layers']
        self.learning_rate = checkpoint['learning_rate']
        print(f"Model loaded from {model_path}")

    @staticmethod
    def create_sequences(data, seq_length):
        x, y = [], []
        for i in range(len(data) - seq_length):
            # Input sequence (seq_length rows, all columns)
            x.append(data[i:i+seq_length, :])
            # Target is temperature at the next step (column index 1)
            y.append(data[i+seq_length, 1])
        return np.array(x), np.array(y)

    def train_model(self, epochs=10, loss_csv_path="training_loss.csv", final_loss_csv_path="final_losses.csv"):
        x, y = self.create_sequences(self.data, self.seq_length)
        x_train = torch.tensor(x, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        with open(loss_csv_path, mode='w', newline='') as loss_file:
            loss_writer = csv.writer(loss_file)
            loss_writer.writerow(["Epoch", "Loss"])

            for epoch in range(epochs):
                self.model.train()
                epoch_loss = 0

                for batch_x, batch_y in loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs.squeeze(), batch_y)
                    epoch_loss += loss.item()
                    loss.backward()
                    self.optimizer.step()

                epoch_loss /= len(loader)
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.10f}")
                loss_writer.writerow([epoch + 1, epoch_loss])
                self.scheduler.step(epoch_loss)

        with open(final_loss_csv_path, mode='a', newline='') as final_loss_file:
            final_loss_writer = csv.writer(final_loss_file)
            final_loss_writer.writerow([epoch_loss])

    @staticmethod
    def infer_timestamps(last_timestamp, steps_ahead, interval_seconds):
        future_timestamps = [
            last_timestamp + timedelta(seconds=interval_seconds * i)
            for i in range(1, steps_ahead + 1)
        ]
        return future_timestamps

    @staticmethod
    def save_predictions_to_csv(predictions, future_timestamps, output_file):
        df = pd.DataFrame({
            "Timestamp": future_timestamps,
            "Predicted_Temperature": predictions
        })
        df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")

    def plot_training_loss(file_path="training_loss.csv", output_path="training_loss_plot.png"):
        """
        Reads training_loss.csv and saves a line plot of the loss per epoch.
        The plot includes the file creation date in the title and uses an exponential scale for the loss axis.
        """
        try:
            creation_date = None
            if os.path.exists(file_path):
                creation_timestamp = os.path.getmtime(file_path)
                creation_date = datetime.fromtimestamp(creation_timestamp).strftime('%Y-%m-%d')

            with open(file_path, mode='r') as file:
                reader = csv.reader(file)
                data = list(reader)

                if len(data) < 2:
                    print("No training data found in the file to plot.")
                    return

                epochs = []
                losses = []
                for row in data[1:]:  # Skip header
                    epochs.append(int(row[0]))
                    losses.append(float(row[1]))

                plt.figure(figsize=(8, 6))
                plt.plot(epochs, losses, marker='o', linestyle='-', label='Loss')
                title = "Training Loss Per Epoch"
                if creation_date:
                    title += f" (File Created: {creation_date})"
                plt.title(title)
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.yscale('log')
                plt.grid(True, which="both", linestyle='--', linewidth=0.5)
                plt.legend()
                plt.savefig(output_path)
                plt.close()

                print(f"Training loss plot saved to {output_path}.")
        except FileNotFoundError:
            print(f"File {file_path} not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def plot_final_losses(file_path="final_losses.csv", output_path="final_losses_plot.png"):
        """
        Reads final_losses.csv and saves a line plot of the final losses across runs with an exponential y-axis.
        """
        try:
            with open(file_path, mode='r') as file:
                reader = csv.reader(file)
                data = list(reader)

                if not data:
                    print("No final losses data found in the file to plot.")
                    return

                runs = list(range(1, len(data) + 1))
                losses = [float(row[0]) for row in data]

                plt.figure(figsize=(8, 6))
                plt.plot(runs, losses, marker='o', color='blue', label='Final Loss')
                plt.yscale('log')
                plt.title("Final Losses Across Runs")
                plt.xlabel("Run")
                plt.ylabel("Loss (log scale)")
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.xticks(runs)
                plt.legend()
                plt.savefig(output_path)
                plt.close()

                print(f"Final losses plot saved to {output_path}.")
        except FileNotFoundError:
            print(f"File {file_path} not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
