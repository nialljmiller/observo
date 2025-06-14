#!/usr/bin/env python3
"""
Weather Station Data Ingestion and System Stats Collection Script

This script:
    - Loads and appends incoming weather station data into a master CSV file.
    - Generates a JSON version of the master CSV file.
    - Collects system stats (CPU, memory, GPU, disk, etc.) and appends them to a CSV file.
"""

import os
import csv
import json
import mimetypes
import subprocess
import warnings
from datetime import datetime, timedelta
from io import BytesIO
import glob
import fcntl

import pandas as pd
import pytz
import psutil
import logging

from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetTemperature,
    nvmlShutdown,
    NVML_TEMPERATURE_GPU,
)
from pynvml import *

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# File paths
INCOMING_FILE = "/media/bigdata/weather_station/weather_data.csv"
MASTER_FILE = "/media/bigdata/weather_station/all_data.csv"
MASTER_FILE_JSON = "/media/bigdata/weather_station/all_data.json"
PLOT_ALL_TIME = "/media/bigdata/weather_station/weather_plot_all.png"
PLOT_1_DAY = "/media/bigdata/weather_station/weather_plot_1_day.png"
PLOT_1_HOUR = "/media/bigdata/weather_station/weather_plot_1_hour.png"
ROLLING_AVERAGES_FILE = "/media/bigdata/weather_station/rolling_averages.csv"
PREDICT_FILE = "/media/bigdata/weather_station/predictions.csv"

# Suppress specific warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Ensure the master file exists; if not, create it with the appropriate header.
if not os.path.exists(MASTER_FILE):
    pd.DataFrame(columns=[
        "Timestamp", "BMP_Temperature_C", "BMP_Pressure_hPa",
        "BMP_Altitude_m", "DHT_Temperature_C", "DHT_Humidity_percent",
        "BH1750_Light_lx"
    ]).to_csv(MASTER_FILE, index=False)

def safe_write_csv(df: pd.DataFrame, filename: str) -> None:
    """Safely write a DataFrame to ``filename`` using a temporary file.

    The file is written while holding an exclusive lock and then atomically
    replaced so readers never see a partially written file.
    """
    temp_file = f"{filename}.tmp"
    with open(temp_file, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        df.to_csv(f, index=False)
        fcntl.flock(f, fcntl.LOCK_UN)
    os.replace(temp_file, filename)



def load_master_data(fp):
    """
    Load master data from a CSV file.
    If timestamps are missing, infer them based on the file's modification time
    or by interpolating between valid timestamps.
    """
    try:
        # Use warn so you know if any rows are messed up
        data = pd.read_csv(fp, on_bad_lines="warn")
    except Exception as e:
        logging.error(f"Error reading master file: {e}")
        return pd.DataFrame()

    if "Timestamp" not in data.columns:
        logging.warning("No 'Timestamp' column found. Inferring timestamps using file modification time.")
        # Use file modification time as a base timestamp
        mod_time = os.path.getmtime(fp)
        base_time = pd.to_datetime(mod_time, unit="s")
        # Create timestamps at a fixed interval (e.g., 1 second apart)
        data["Timestamp"] = [base_time + pd.Timedelta(seconds=i) for i in range(len(data))]
    else:
        # Convert timestamps; unparseable ones become NaT
        data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
        if data["Timestamp"].isnull().any():
            logging.warning("Missing timestamps found. Attempting to infer them by interpolation.")
            valid_count = data["Timestamp"].notnull().sum()
            if valid_count >= 2:
                # Convert to numeric (nanoseconds since epoch) for interpolation
                numeric_ts = data["Timestamp"].apply(lambda x: x.value if pd.notnull(x) else None)
                numeric_ts = pd.Series(numeric_ts).interpolate(method="linear")
                data["Timestamp"] = pd.to_datetime(numeric_ts)
            else:
                # If not enough valid timestamps, use current time for missing ones
                logging.warning("Not enough valid timestamps to interpolate. Filling missing ones with current time.")
                data["Timestamp"] = data["Timestamp"].fillna(pd.Timestamp.now())

    # Sort the DataFrame with valid timestamps first
    data = data.sort_values("Timestamp", na_position="last").reset_index(drop=True)
    return data



def generate_json_from_csv(csv_path, json_path):
    """
    Convert CSV data to JSON format.

    Args:
        csv_path (str): Path to the source CSV file.
        json_path (str): Path to save the generated JSON file.
    """
    try:
        with open(csv_path, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            data = list(reader)

        with open(json_path, "w") as jsonfile:
            json.dump(data, jsonfile, indent=4)

        logging.info(f"JSON data successfully written to {json_path}")
    except Exception as e:
        logging.error(f"Error generating JSON file: {e}")


def append_new_data(master_data):
    """
    Append new weather data from the incoming file to the master DataFrame.

    Args:
        master_data (pd.DataFrame): Existing master data.

    Returns:
        pd.DataFrame: Updated master data.
    """
    if not os.path.exists(INCOMING_FILE):
        logging.warning(f"Incoming file {INCOMING_FILE} does not exist. Skipping this iteration.")
        return master_data

    file_type, _ = mimetypes.guess_type(INCOMING_FILE)
    if file_type != "text/csv":
        logging.warning(f"{INCOMING_FILE} is not a text/csv file. Detected type: {file_type}")
        return master_data

    try:
        # Read and clear the incoming file while holding a lock so the
        # Raspberry Pi does not modify it concurrently.
        with open(INCOMING_FILE, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            incoming_data = pd.read_csv(f, encoding="utf-8", on_bad_lines="skip")
            f.seek(0)
            f.truncate()  # Clear the file after reading
            fcntl.flock(f, fcntl.LOCK_UN)

        incoming_data["Timestamp"] = pd.to_datetime(incoming_data["Timestamp"], errors="coerce")
        incoming_data = incoming_data.dropna(subset=["Timestamp"])
    except Exception as e:
        logging.error(f"Error loading incoming data: {e}")
        return master_data

    # Ensure timestamps are timezone-aware and converted to UTC
    if incoming_data["Timestamp"].dt.tz is None:
        incoming_data["Timestamp"] = incoming_data["Timestamp"].dt.tz_localize("UTC")
    else:
        incoming_data["Timestamp"] = incoming_data["Timestamp"].dt.tz_convert("UTC")

    if master_data["Timestamp"].dt.tz is None:
        master_data["Timestamp"] = master_data["Timestamp"].dt.tz_localize("UTC")
    else:
        master_data["Timestamp"] = master_data["Timestamp"].dt.tz_convert("UTC")

    # Concatenate and remove duplicates
    combined_data = pd.concat([master_data, incoming_data], ignore_index=True)
    combined_data = combined_data.drop_duplicates(subset="Timestamp").sort_values("Timestamp").reset_index(drop=True)

    # Save updated master data and generate corresponding JSON
    #combined_data.to_csv(MASTER_FILE, index=False)
    safe_write_csv(combined_data, MASTER_FILE)
    generate_json_from_csv(MASTER_FILE, MASTER_FILE_JSON)
    logging.info(f"Appended new data and saved to {MASTER_FILE}. Total rows: {len(combined_data)}.")

    return combined_data


def initialize_csv(output_file="system_stats.csv"):
    """
    Initialize a CSV file with headers for system statistics.

    Args:
        output_file (str): Path to the CSV file to be initialized.
    """
    with open(output_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Timestamp",
            "CPU Usage (%)",
            "CPU Temp (°C)",
            "Memory Usage (%)",
            "GPU Usage (%)",
            "GPU Memory Usage (%)",
            "GPU Temp (°C)",
            "Disk Usage",
            "Net Disk I/O (MB)",
            "Thermals"
        ])


def gather_system_stats(output_file="system_stats.csv"):
    """
    Gather and append system statistics to a CSV file.

    Args:
        output_file (str): Path to the CSV file where stats will be appended.
    """
    # Initialize NVIDIA Management Library (for GPU stats)
    nvmlInit()
    gpu_handle = nvmlDeviceGetHandleByIndex(0)  # Assuming a single NVIDIA GPU

    # Gather CPU usage and temperature
    cpu_usage = psutil.cpu_percent(interval=1)
    try:
        cpu_temp = psutil.sensors_temperatures()["coretemp"][0].current
    except (KeyError, IndexError):
        cpu_temp = "N/A"

    # Gather memory usage stats
    memory_usage = psutil.virtual_memory().percent

    # Gather GPU usage and temperature
    gpu_utilization = nvmlDeviceGetUtilizationRates(gpu_handle)
    gpu_memory = nvmlDeviceGetMemoryInfo(gpu_handle)
    gpu_temp = nvmlDeviceGetTemperature(gpu_handle, NVML_TEMPERATURE_GPU)

    # Gather disk usage stats
    disk_usage = []
    for partition in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            disk_usage.append({
                "device": partition.device,
                "mountpoint": partition.mountpoint,
                "used": usage.used,
                "total": usage.total,
                "percent": usage.percent
            })
        except PermissionError:
            continue

    # Gather disk I/O stats
    disk_io = psutil.disk_io_counters()
    total_read = disk_io.read_bytes / (1024 ** 2)   # MB
    total_write = disk_io.write_bytes / (1024 ** 2)   # MB

    # Gather additional temperatures using `sensors`
    formatted_temps = {}
    try:
        sensors_output = subprocess.check_output(["sensors"], text=True).splitlines()
        for line in sensors_output:
            if ":" in line:
                parts = line.split(":")
                label = parts[0].strip()
                temp_data = parts[1].strip().split()
                # Allow one decimal point; adjust if negative values are possible
                if temp_data and temp_data[0].replace(".", "", 1).lstrip("-").isdigit():
                    formatted_temps[label] = float(temp_data[0].replace("°C", ""))
    except Exception as e:
        logging.error(f"Error fetching additional temperatures: {e}")

    # Format disk usage and temperature information for CSV
    disk_usage_str = "; ".join(
        f"{entry['device']}({entry['mountpoint']}): {entry['used'] / 1024**3:.2f}GB/"
        f"{entry['total'] / 1024**3:.2f}GB ({entry['percent']}%)"
        for entry in disk_usage
    )
    temp_str = "; ".join(f"{key}: {value}°C" for key, value in formatted_temps.items())

    # Append system stats to the CSV file
    with open(output_file, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            datetime.now().isoformat(),
            f"{cpu_usage}%",
            f"{cpu_temp}°C" if cpu_temp != "N/A" else "N/A",
            f"{memory_usage}%",
            f"{gpu_utilization.gpu}%",
            f"{gpu_memory.used / gpu_memory.total * 100:.2f}%",
            f"{gpu_temp}°C",
            disk_usage_str,
            f"Read: {total_read:.2f}MB, Write: {total_write:.2f}MB",
            temp_str
        ])

    nvmlShutdown()


def main():
    #logging.info("Starting server weather ingest script...")

    #local_stats_file = "my_pc_stats.csv"
    # Uncomment the next line if you wish to initialize the stats CSV with headers.
    #initialize_csv(local_stats_file)
    #gather_system_stats(local_stats_file)
    #logging.info("Appending new weather data...")
    master_data = load_master_data(MASTER_FILE)
    master_data = append_new_data(master_data)


if __name__ == "__main__":
    main()
