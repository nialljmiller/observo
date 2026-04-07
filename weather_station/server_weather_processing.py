# Standard Library Imports
import re
import os
import shutil
import csv
import json
import time
import warnings
warnings.filterwarnings("ignore")
import glob
import subprocess
import mimetypes
import logging
from datetime import datetime, timedelta
from io import BytesIO
import logging
from PIL import ImageFont
import os
import glob
import logging
import json
import torch
import pandas as pd
from torchvision.ops import nms
from PIL import Image, ImageDraw, ImageFont, ImageSequence
from datetime import datetime
from tqdm import tqdm

import os
import glob
import logging
from PIL import Image, ImageDraw, ImageFont
import torch
import pandas as pd
from torchvision.ops import nms

import os
import glob
import logging
from PIL import Image, ImageDraw, ImageFont
import torch
import pandas as pd
from torchvision.ops import nms
import torch
import pandas as pd
import numpy as np
import pytz
from pytz import UTC
from PIL import Image, ImageDraw, ImageFont
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.collections
import matplotlib.dates  # Additional import if specific functions are needed
import matplotlib.ticker as mticker
import matplotlib.ticker as ticker
from matplotlib.dates import DateFormatter
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler
from numpy.polynomial import Polynomial
#import GPy
import os
import glob
import logging
from PIL import Image, ImageDraw, ImageFont
import torch
import pandas as pd
from torchvision.ops import nms
import psutil

import warnings
import bird_detection



# Suppress FutureWarnings related to torch.cuda.amp.autocast
warnings.filterwarnings(
    "ignore", 
    message=".*torch\\.cuda\\.amp\\.autocast.*", 
    category=FutureWarning
)

# Custom Module Imports
from weather_forcast import WeatherForecaster  # Assuming it's a custom module

# Suppress specific warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

#from instabot import Bot
#try:
#    os.remove("config/birdbot9000_uuid_and_cookie.json")
#except:
#    pass

#try:
    #bot = Bot()
    #bot.login(username="birdbot9000", password="BirdBot9K!")
#except:
#    pass

# File paths
INCOMING_FILE = "/media/bigdata/weather_station/weather_data.csv"
MASTER_FILE = "/media/bigdata/weather_station/all_data.csv"
MASTER_FILE_json = "/media/bigdata/weather_station/all_data.json"
PLOT_ALL_TIME = "/media/bigdata/weather_station/weather_plot_all.png"
PLOT_1_DAY = "/media/bigdata/weather_station/weather_plot_1_day.png"
PLOT_1_HOUR = "/media/bigdata/weather_station/weather_plot_1_hour.png"
ROLLING_AVERAGES_FILE = "/media/bigdata/weather_station/rolling_averages.csv"
PREDICT_FILE = "/media/bigdata/weather_station/predictions.csv"


# Ensure master file exists
if not os.path.exists(MASTER_FILE):
    pd.DataFrame(columns=[
        "Timestamp", "BMP_Temperature_C", "BMP_Pressure_hPa",
        "BMP_Altitude_m", "DHT_Temperature_C", "DHT_Humidity_percent",
        "BH1750_Light_lx"
    ]).to_csv(MASTER_FILE, index=False)

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


def initialize_csv(output_file="system_stats.csv"):
    """Initializes the CSV file with the header."""
    with open(output_file, mode='w', newline='') as csvfile:
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
        
        



def check_sensor_validity(df, min_frac=0.10):
    """
    Run once per processing cycle. Returns a dict of booleans for every sensor/column.
    A sensor is 'valid' if at least min_frac of its recent values are non-NaN.
    """
    def valid(col):
        return col in df.columns and df[col].notna().mean() >= min_frac

    flags = {
        "bmp_temp":     valid("BMP_Temperature_C"),
        "bmp_pressure": valid("BMP_Pressure_hPa"),
        "dht_temp":     valid("DHT_Temperature_C"),
        "dht_humidity": valid("DHT_Humidity_percent"),
        "bh1750":       valid("BH1750_Light_lx"),
    }
    flags["dht"] = flags["dht_temp"] and flags["dht_humidity"]

    # Log anything offline
    for name, ok in flags.items():
        if not ok:
            logging.warning(f"Sensor offline or missing: {name}")
    return flags

def save_latest_copy(image_dir, output_name="latest.jpg"):
    """
    Creates a copy of the most recent weather plot as 'latest.jpg'
    
    This function should be called right after generating a new weather plot.
    """

    # Get all image files
    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
    
    if not image_files:
        print("No image files found.")
        return False
    
    # Find the most recent file based on modification time
    latest_file = max(image_files, key=os.path.getmtime)
    
    # Create the output path
    output_path = os.path.join(image_dir, output_name)
    
    # Copy the file
    shutil.copy2(latest_file, output_path)
    print(f"Copied {latest_file} to {output_path}")
    return True


STATS_FILE = "my_pc_stats.csv"

def gather_system_stats(output_file=STATS_FILE):
    """Append CPU/memory/temp stats to a CSV, creating it with a header if needed."""
    cpu_usage  = psutil.cpu_percent(interval=1)
    memory_pct = psutil.virtual_memory().percent
    try:
        cpu_temp = psutil.sensors_temperatures()["coretemp"][0].current
    except (KeyError, IndexError, AttributeError):
        cpu_temp = None

    write_header = not os.path.exists(output_file) or os.path.getsize(output_file) == 0
    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Timestamp", "CPU_Usage_pct", "Memory_Usage_pct", "CPU_Temp_C"])
        writer.writerow([
            datetime.now().isoformat(),
            round(cpu_usage, 1),
            round(memory_pct, 1),
            round(cpu_temp, 1) if cpu_temp is not None else "N/A",
        ])





def generate_json_from_csv(csv_path, json_path):
    """
    Convert CSV data to JSON format.

    Args:
        csv_path (str): Path to the source CSV file.
        json_path (str): Path to save the generated JSON file.
    """
    try:
        with open(csv_path, "r") as csvfile:
            reader = csv.DictReader(csvfile)  # Reads CSV as dictionaries
            data = list(reader)  # Convert the entire CSV into a list of dictionaries

        # Save the data to a JSON file
        with open(json_path, "w") as jsonfile:
            json.dump(data, jsonfile, indent=4)

        print(f"JSON data successfully written to {json_path}")

    except Exception as e:
        print(f"Error generating JSON file: {e}")


def generate_summary_plot(data, output_path, sensor_flags=None):
    if sensor_flags is None:
        sensor_flags = {"dht_humidity": True}

    fig, ax_temp_c = plt.subplots(figsize=(10, 6))
    ax_temp_c.plot(data["Timestamp"], data["Median_Temperature_C"], color="purple", alpha=0.7, label="Temperature (°C)")
    ax_temp_c.tick_params(axis="y", labelcolor="blue")

    ax_temp_f = ax_temp_c.twinx()
    ax_temp_f.spines["left"].set_position(("axes", 0))
    ax_temp_f.plot(data["Timestamp"], data["Median_Temperature_C"] * 9/5 + 32, color="purple", alpha=0.7)
    ax_temp_f.set_ylabel("Temperature (°C/°F)", color="purple")
    ax_temp_f.tick_params(axis="y", labelcolor="red")
    ax_temp_f.yaxis.set_label_position("left")
    ax_temp_f.yaxis.tick_left()

    ax_hum = ax_temp_c.twinx()
    ax_hum.spines["right"].set_visible(True)
    if sensor_flags["dht_humidity"]:
        ax_hum.plot(data["Timestamp"], data["DHT_Humidity_percent_Smoothed"], label="Humidity (%)", color="green", alpha=0.7)
        ax_hum.set_ylabel("Humidity (%)", color="green")
        ax_hum.tick_params(axis="y", labelcolor="green")
    else:
        ax_hum.set_visible(False)
        ax_temp_c.set_title("Summary (DHT offline — humidity unavailable)")

    ax_temp_c.legend(loc="upper left")
    ax_hum.legend(loc="upper right")
    ax_temp_c.xaxis.set_tick_params(rotation=45)
    ax_temp_c.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()





def generate_hourly_gif_with_plot(image_dir, output_gif, data, sensor_flags=None):
    """
    Generates an animated GIF from the images in `image_dir` (from the past hour).
    For each image, a plot (appended below the image) shows the cumulative temperature
    and humidity data from the start of the hour (now - 1 hour) up to the image’s timestamp.
    
    Args:
        image_dir (str): Directory where the .jpg images are stored.
        output_gif (str): Path for saving the final GIF.
        data (pd.DataFrame): DataFrame with at least these columns:
            - "Timestamp": datetime values (assumed to be timezone-aware in UTC)
            - "Median_Temperature_C": temperature in °C
            - "DHT_Humidity_percent_Smoothed": humidity (%)
    """
    # Temporary filename (to avoid access issues)
    temp_gif = os.path.join(os.path.dirname(output_gif), "temp.gif")

    # Get current UTC time as a timezone-aware datetime
    now = datetime.now(pytz.UTC)

    # For the cumulative plot, define the starting time as one hour ago.
    plot_start = now - timedelta(hours=1)

    # Get all JPEG images (assumes naming convention: YYYYMMDD_HHMMSS.jpg)
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    
    # Filter images from the last hour (using filename timestamps)
    recent_images = []
    for img in image_files:
        try:
            # Extract timestamp from filename (expected format: YYYYMMDD_HHMMSS)
            img_filename = os.path.basename(img).replace(".jpg", "")
            img_timestamp = datetime.strptime(img_filename, "%Y%m%d_%H%M%S")
            # Localize the naive timestamp to UTC
            img_timestamp = pytz.UTC.localize(img_timestamp)
            if now - img_timestamp <= timedelta(hours=1):
                recent_images.append((img, img_timestamp))
        except ValueError:
            continue  # Skip files with incorrect naming format

    if len(recent_images) < 2:
        print("[GIF] Not enough images to generate GIF (need at least 2). Skipping.")
        return

    print(f"[GIF] Generating GIF from {len(recent_images)} images...")

    frames = []
    # Define the Mountain Time zone (using America/Denver to account for DST)
    mountain_tz = pytz.timezone("America/Denver")

    # Loop over each image and create a composite frame
    for img_path, timestamp in recent_images:
        # Convert the image timestamp to Mountain Time for display
        mountain_time = timestamp.astimezone(mountain_tz)

        # Open image and perform any transforms (here, rotate 180°)
        img = Image.open(img_path).transpose(Image.ROTATE_180)

        # Add the timestamp text onto the image
        draw = ImageDraw.Draw(img)
        try:
            # Adjust the font size as needed; if arial.ttf is not found, use the default font.
            font = ImageFont.truetype("arial.ttf", 32)
        except IOError:
            font = ImageFont.load_default()
        text = mountain_time.strftime("%Y-%m-%d %H:%M:%S")
        # Position the text near the bottom-right (adjust these values as needed)
        text_position = (img.width - 250, img.height - 50)
        draw.text(text_position, text, font=font, fill=(255, 0, 0))

        # -----
        # Instead of plotting only a small window around the frame's timestamp,
        # we now accumulate data from the start of the hour (plot_start) up to the frame's timestamp.
        subset = data[(data["Timestamp"] >= plot_start) & (data["Timestamp"] <= timestamp)]

        # Create the plot
        fig, ax_temp_c = plt.subplots(figsize=(8, 4))
        if not subset.empty:

            # Plot temperature in Celsius on the primary y-axis
            ax_temp_c.plot(subset["Timestamp"], subset["Median_Temperature_C"],
                   color="purple", alpha=0.7, label="Temp (°C)")
            ax_temp_c.tick_params(axis="y", labelcolor="purple")

            # Add Fahrenheit scale on a secondary y-axis
            ax_temp_f = ax_temp_c.twinx()
            ax_temp_f.plot(subset["Timestamp"], subset["Median_Temperature_C"] * 9/5 + 32,
                   color="orange", alpha=0.7, label="Temp (°F)")
            ax_temp_f.set_ylabel("Temperature (°C/°F)", color="red")
            ax_temp_f.tick_params(axis="y", labelcolor="orange")
            ax_temp_f.yaxis.set_label_position("left")
            ax_temp_f.yaxis.tick_left()

            ax_hum = ax_temp_c.twinx()
            if sensor_flags and sensor_flags.get("dht_humidity"):
                ax_hum.plot(subset["Timestamp"], subset["DHT_Humidity_percent_Smoothed"],
                    color="green", alpha=0.7, label="Humidity (%)")
                ax_hum.set_ylabel("Humidity (%)", color="green")
                ax_hum.tick_params(axis="y", labelcolor="green")
            else:
                ax_hum.set_visible(False)

            plot_title = (f"Data from {plot_start.astimezone(mountain_tz).strftime('%H:%M:%S')} "
                          f"to {mountain_time.strftime('%H:%M:%S')}")
        else:
            # If no data is found, indicate that on the plot.
            ax_temp_c.text(0.5, 0.5, "No data available", ha="center", va="center",
                           transform=ax_temp_c.transAxes, fontsize=16)
            plot_title = f"No Data around {mountain_time.strftime('%H:%M:%S')}"


        # Create a formatter that converts to Mountain Time and formats as HH:MM:SS
        formatter = mdates.DateFormatter('%H:%M:%S', tz=mountain_tz)

        # Apply this formatter to your primary axis
        ax_temp_c.xaxis.set_major_formatter(formatter)

        ax_temp_c.set_title(plot_title)
        ax_temp_c.set_xlabel("Time")
        fig.autofmt_xdate()
        plt.tight_layout()

        # Save the plot to a BytesIO buffer and then open it with PIL
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        plot_img = Image.open(buf)

        # Resize the plot so its width matches the image width
        if plot_img.width != img.width:
            new_height = int(plot_img.height * img.width / plot_img.width)
            try:
                resample_filter = Image.Resampling.LANCZOS
            except AttributeError:
                resample_filter = Image.ANTIALIAS  # For older versions of Pillow

            plot_img = plot_img.resize((img.width, new_height), resample_filter)

        # Combine the original image (with timestamp) and the plot (stacked vertically)
        combined_height = img.height + plot_img.height
        combined_img = Image.new("RGB", (img.width, combined_height))
        combined_img.paste(img, (0, 0))
        combined_img.paste(plot_img, (0, img.height))

        frames.append(combined_img)

    # Save the frames as an animated GIF
    frames[0].save(
        temp_gif,
        save_all=True,
        append_images=frames[1:],
        duration=100,  # Duration in milliseconds per frame; adjust as desired
        loop=0
    )

    # Rename the temporary GIF to the final output path (atomic operation)
    os.replace(temp_gif, output_gif)
    print(f"[GIF] GIF saved to {output_gif}.")


def generate_daily_gif_with_plot(image_dir, output_gif, data, sensor_flags=None):
    """
    Generates an animated GIF from the images in `image_dir` (from the past 24 hours).
    For each image, a plot (appended below the image) shows the cumulative temperature
    and humidity data from the start of the day (now - 24 hours) up to the image's timestamp.
    
    Optimized for performance with 24 hours of data.
    
    Args:
        image_dir (str): Directory where the .jpg images are stored.
        output_gif (str): Path for saving the final GIF.
        data (pd.DataFrame): DataFrame with at least these columns:
            - "Timestamp": datetime values (assumed to be timezone-aware in UTC)
            - "Median_Temperature_C": temperature in °C
            - "DHT_Humidity_percent_Smoothed": humidity (%)
    """
    # Temporary filename (to avoid access issues)
    temp_gif = os.path.join(os.path.dirname(output_gif), "temp.gif")

    # Get current UTC time as a timezone-aware datetime
    now = datetime.now(pytz.UTC)

    # For the cumulative plot, define the starting time as 24 hours ago
    plot_start = now - timedelta(hours=24)

    # Get all JPEG images (assumes naming convention: YYYYMMDD_HHMMSS.jpg)
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    
    # Filter images from the last 24 hours (using filename timestamps)
    recent_images = []
    for img in image_files:
        try:
            # Extract timestamp from filename (expected format: YYYYMMDD_HHMMSS)
            img_filename = os.path.basename(img).replace(".jpg", "")
            img_timestamp = datetime.strptime(img_filename, "%Y%m%d_%H%M%S")
            # Localize the naive timestamp to UTC
            img_timestamp = pytz.UTC.localize(img_timestamp)
            if now - img_timestamp <= timedelta(hours=24):
                recent_images.append((img, img_timestamp))
        except ValueError:
            continue  # Skip files with incorrect naming format

    if len(recent_images) < 2:
        print("[DAILY GIF] Not enough images to generate GIF (need at least 2). Skipping.")
        return

    print(f"[DAILY GIF] Generating GIF from {len(recent_images)} images...")

    # Performance optimization: Pre-filter and downsample data for plotting
    data_subset = data[(data["Timestamp"] >= plot_start) & (data["Timestamp"] <= now)]
    # Downsample data for plotting if we have too many points (keep every Nth point)
    if len(data_subset) > 500:  # Adjust threshold as needed
        step = len(data_subset) // 500
        data_subset = data_subset.iloc[::step]

    frames = []
    # Define the Mountain Time zone (using America/Denver to account for DST)
    mountain_tz = pytz.timezone("America/Denver")

    # Process ALL images - no skipping

    # Loop over each image and create a composite frame
    for i, (img_path, timestamp) in enumerate(recent_images):
        if i % 10 == 0:  # Progress indicator
            print(f"[DAILY GIF] Processing frame {i+1}/{len(recent_images)}")
            
        # Convert the image timestamp to Mountain Time for display
        mountain_time = timestamp.astimezone(mountain_tz)

        # Open image and perform any transforms (here, rotate 180°)
        img = Image.open(img_path).transpose(Image.ROTATE_180)

        # Add the timestamp text onto the image
        draw = ImageDraw.Draw(img)
        try:
            # Smaller font for daily GIF to save space
            font = ImageFont.truetype("arial.ttf", 28)
        except IOError:
            font = ImageFont.load_default()
        text = mountain_time.strftime("%Y-%m-%d %H:%M:%S")
        # Position the text near the bottom-right
        text_position = (img.width - 250, img.height - 40)
        draw.text(text_position, text, font=font, fill=(255, 0, 0))

        # Get data subset up to current timestamp
        current_subset = data_subset[data_subset["Timestamp"] <= timestamp]

        # Create the plot with reduced size for performance
        fig, ax_temp_c = plt.subplots(figsize=(6, 3))  # Smaller plot for performance
        if not current_subset.empty:

            # Simplified plotting - only temperature and humidity
            ax_temp_c.plot(current_subset["Timestamp"], current_subset["Median_Temperature_C"],
                   color="purple", alpha=0.8, linewidth=1, label="Temp (°C)")
            ax_temp_c.set_ylabel("Temperature (°C)", color="purple")
            ax_temp_c.tick_params(axis="y", labelcolor="purple")

            # Add humidity on secondary y-axis
            ax_hum = ax_temp_c.twinx()
            
            if sensor_flags and sensor_flags.get("dht_humidity"):
                ax_hum.plot(current_subset["Timestamp"], subset["DHT_Humidity_percent_Smoothed"],
                    color="green", alpha=0.7, label="Humidity (%)")
                ax_hum.set_ylabel("Humidity (%)", color="green")
                ax_hum.tick_params(axis="y", labelcolor="green")
            else:
                ax_hum.set_visible(False)

            # Simplified title
            plot_title = f"24h Data to {mountain_time.strftime('%H:%M')}"
        else:
            # If no data is found, indicate that on the plot
            ax_temp_c.text(0.5, 0.5, "No data available", ha="center", va="center",
                           transform=ax_temp_c.transAxes, fontsize=12)
            plot_title = f"No Data around {mountain_time.strftime('%H:%M:%S')}"

        # Simplified time formatting for 24h view
        formatter = mdates.DateFormatter('%H:%M', tz=mountain_tz)
        ax_temp_c.xaxis.set_major_formatter(formatter)
        
        # Reduce number of x-axis ticks for cleaner look
        ax_temp_c.xaxis.set_major_locator(mdates.HourLocator(interval=4))

        ax_temp_c.set_title(plot_title, fontsize=10)
        ax_temp_c.set_xlabel("Time", fontsize=8)
        
        # Smaller tick labels
        ax_temp_c.tick_params(labelsize=8)
        if 'ax_hum' in locals():
            ax_hum.tick_params(labelsize=8)
        
        fig.autofmt_xdate()
        plt.tight_layout()

        # Save the plot to a BytesIO buffer with lower DPI for performance
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=72)  # Lower DPI for performance
        plt.close(fig)
        buf.seek(0)
        plot_img = Image.open(buf)

        # Resize the plot so its width matches the image width
        if plot_img.width != img.width:
            new_height = int(plot_img.height * img.width / plot_img.width)
            try:
                resample_filter = Image.Resampling.LANCZOS
            except AttributeError:
                resample_filter = Image.ANTIALIAS

            plot_img = plot_img.resize((img.width, new_height), resample_filter)

        # Combine the original image (with timestamp) and the plot
        combined_height = img.height + plot_img.height
        combined_img = Image.new("RGB", (img.width, combined_height))
        combined_img.paste(img, (0, 0))
        combined_img.paste(plot_img, (0, img.height))

        frames.append(combined_img)

    print("[DAILY GIF] Saving GIF...")
    
    # Save the frames as an animated GIF with longer duration per frame
    frames[0].save(
        temp_gif,
        save_all=True,
        append_images=frames[1:],
        duration=200,  # Longer duration per frame (vs 100 for hourly)
        loop=0,
        optimize=True  # Optimize GIF size
    )

    # Rename the temporary GIF to the final output path (atomic operation)
    os.replace(temp_gif, output_gif)
    print(f"[DAILY GIF] GIF saved to {output_gif}.")


def compute_ECI(data):
    """Compute Environmental Comfort Index (ECI) from a weather DataFrame.

    The function is tolerant of differing humidity column names used by
    various data sources.  It will look for a column named ``"Humidity"`` and
    fall back to ``"DHT_Humidity_percent"`` if necessary.  The remaining
    required columns are:

    - ``"Median_Temperature_C"`` (°C)
    - ``"BMP_Pressure_hPa"`` (hPa)
    - ``"BH1750_Light_lx"`` (lux)

    Returns:
        pd.Series: Smoothed ECI values in the range 0–1.
    """

    def comfort_score(x, center, width, skew=0.0, kurtosis=1.0):
        x_shifted = x - (skew * width)
        score = np.exp(-((x_shifted - center) / width) ** (2 * kurtosis))
        return np.clip(score, 0, 1)

    # Determine which humidity column is available
    humidity_col = "Humidity" if "Humidity" in data.columns else "DHT_Humidity_percent"
    if humidity_col not in data.columns:
        raise KeyError("Data must contain 'Humidity' or 'DHT_Humidity_percent' column")

    # Compute normalized comfort scores
    H_norm = comfort_score(data[humidity_col], center=50, width=20, skew=0.2, kurtosis=1.2)
    T_norm = comfort_score(data["Median_Temperature_C"], center=21, width=5, skew=0.5, kurtosis=1.0)
    P_norm = comfort_score(data["BMP_Pressure_hPa"], center=1013, width=10, skew=0.0, kurtosis=1.5)
    L_norm = comfort_score(data["BH1750_Light_lx"], center=8000, width=7000, skew=-0.2, kurtosis=0.8)

    # Weighted sum of normalized scores
    ECI_raw = (
        0.4 * T_norm +
        0.35 * H_norm +
        0.1 * P_norm +
        0.15 * L_norm
    )

    # Smooth result with centered rolling mean
    return pd.Series(ECI_raw).rolling(window=20, center=True, min_periods=1).mean()


def generate_plots(data, predict_data, output_path, title, out_of_date_flag):
    """Generate a 4x4 subplot for temperature, humidity, pressure, and light with additional calculated metrics."""
    # Convert timestamps to Mountain Time
    mountain_tz = pytz.timezone("America/Denver")
    data["Timestamp"] = data["Timestamp"].dt.tz_convert(mountain_tz)
    predict_data["Timestamp"] = predict_data["Timestamp"].dt.tz_convert(mountain_tz)
    altitude_m = data["BMP_Altitude_m"]
    altitude_m = altitude_m.where(altitude_m > 3000, 2134.164856161599)

    # Calculate median temperature

    # Replace NaNs and Infs in all numeric columns
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Only drop rows where core (BMP / timestamp) columns are missing —
    # DHT columns may be entirely NaN if that sensor is dead, and that is OK.
    core_cols = ["Timestamp", "BMP_Temperature_C", "BMP_Pressure_hPa"]
    data.dropna(subset=core_cols, inplace=True)

    # ── Sensor validity checks ──────────────────────────────────────────────
    # A column is considered "valid" if at least 10 % of its values are real.
    def _sensor_valid(series, min_frac=0.10):
        if series is None or len(series) == 0:
            return False
        return series.notna().mean() >= min_frac

    dht_temp_valid     = _sensor_valid(data.get("DHT_Temperature_C"))
    dht_humidity_valid = _sensor_valid(data.get("DHT_Humidity_percent"))
    dht_valid          = dht_temp_valid and dht_humidity_valid

    def _dead_sensor_notice(ax, name="DHT Sensor"):
        """Draw a centred notice on an axes when its sensor is offline."""
        ax.text(
            0.5, 0.5, f"{name}\nNo Data / Sensor Offline",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, color="grey",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
        )

    # Convert temperatures to Fahrenheit
    data["BMP_Temperature_F"] = data["BMP_Temperature_C"] * 9 / 5 + 32
    data["DHT_Temperature_F"] = data["DHT_Temperature_C"] * 9 / 5 + 32


    # Ensure the Timestamp column is clean and valid
    if "Timestamp" in data.columns:
        data = data.dropna(subset=["Timestamp"])  # Drop rows with missing timestamps
        if not pd.api.types.is_datetime64_any_dtype(data["Timestamp"]):
            # Convert to datetime if not already in the correct format
            data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
        data = data.dropna(subset=["Timestamp"])  # Drop rows with invalid timestamps


    timestamps = np.arange(len(data["Timestamp"]))
    smooth_humidity = data["DHT_Humidity_percent_Smoothed"].values if dht_humidity_valid else None

    # Heat Index Calculation
    T = data["DHT_Temperature_F"] if dht_temp_valid else data["BMP_Temperature_F"]
    H = smooth_humidity if dht_humidity_valid else np.full(len(data), 50.0)  # neutral fallback

    h = altitude_m
  
    data["Heat_Index"] = np.where(
        data["Median_Temperature_C"] > 27,  # Use heat index for temps >27°C
        -42.379 + 2.04901523 * data["DHT_Temperature_F"] + 10.14333127 * H
        - 0.22475541 * data["DHT_Temperature_F"] * H
        - 0.00683783 * data["DHT_Temperature_F"]**2
        - 0.05481717 * H**2
        + 0.00122874 * data["DHT_Temperature_F"]**2 * H
        + 0.00085282 * data["DHT_Temperature_F"] * H**2
        - 0.00000199 * data["DHT_Temperature_F"]**2 * H**2,
        data["Median_Temperature_C"],  # Below threshold, return actual temperature
    )
              


    # Use the Magnus-Tetens formula for vapor pressure
    e = H / 100 * 6.112 * np.exp((17.62 * data["Median_Temperature_C"]) / (data["Median_Temperature_C"] + 243.12))

    # Calculate specific humidity
    data["Specific_Humidity_gkg"] = 0.622 * e / (data["Sea_Level_Pressure_hPa"] - e) * 1000

    data["ECI"] = compute_ECI(data)
    
    fig, axs = plt.subplots(4, 2, figsize=(15, 15))

    # Get the timestamp of the last real datapoint
    last_real_timestamp = data["Timestamp"].iloc[-1]

    # Calculate the real data time span
    real_start = data["Timestamp"].iloc[0]
    real_end = last_real_timestamp
    real_time_span = real_end - real_start

    # Calculate the maximum predicted time span (25% of the real time span)
    max_predicted_span = real_time_span / 4
    max_predicted_end_time = last_real_timestamp + max_predicted_span

    # Filter predict_data so that predicted timestamps are within the allowed span
    predict_data_subset = predict_data[predict_data["Timestamp"] <= max_predicted_end_time].copy()

    # Prepend the last real datapoint to the predicted subset so the lines join
    last_real_row = data.iloc[[-1]].copy()
    last_real_row["Timestamp"] = last_real_timestamp  # Ensure timestamp is correct
    # Use a temperature column from the real data (e.g., BMP_Temperature_C)
    last_real_row["Predicted_Temperature"] = last_real_row["BMP_Temperature_C"]
    predict_data_subset = pd.concat([last_real_row, predict_data_subset], ignore_index=True)


    # Prepend the last real datapoint to the predicted subset.
    # Here, we assume that the actual temperature to join is, say, BMP_Temperature_C.
    # You can change that to whichever temperature column you want.
    last_real_row = data.iloc[[-1]].copy()
    last_real_row["Timestamp"] = last_real_timestamp  # Ensure the timestamp is set
    last_real_row["Predicted_Temperature"] = last_real_row["BMP_Temperature_C"]
    predict_data_subset = pd.concat([last_real_row, predict_data_subset], ignore_index=True)

    # Temperature Plot
    ax1 = axs[0, 0]
    ax1.plot(data["Timestamp"], data["BMP_Temperature_C"], color="blue", alpha=0.1)
    ax1.plot(data["Timestamp"], data["BMP_Temperature_Smoothed"], label="BMP Temp", color="blue", alpha=0.7)
    if dht_temp_valid:
        ax1.plot(data["Timestamp"], data["DHT_Temperature_C"], color="cyan", alpha=0.1)
        ax1.plot(data["Timestamp"], data["DHT_Temperature_Smoothed"], label="DHT Temp", color="cyan", alpha=0.7)
    # Only show predictions when DHT was healthy (model input included DHT humidity)
    if dht_valid:
        ax1.plot(predict_data_subset["Timestamp"], predict_data_subset["Predicted_Temperature"], label="Predicted Temp", color="royalblue", alpha=0.7)
    ax1.plot(data["Timestamp"], data["Median_Temperature_C"], color="magenta", linestyle="--")
    ax1.set_title("Temperature" + ("" if dht_temp_valid else " (DHT offline)"))
    ax1.set_ylabel("Temperature (°C)")
    ax1.legend(loc="upper left")
    ax1.grid()


    # Compute x-axis limits using only the real data timestamps
    x_min = data["Timestamp"].min()
    x_max = data["Timestamp"].max()
    x_range = x_max - x_min
    x_padding = x_range * 0.05  # 5% padding

    # Apply the limits to the x-axis
    ax1.set_xlim(x_min - x_padding, x_max + x_padding)

    # Define the real temperature columns (ignore the predicted temperature)
    real_temp_cols = ["BMP_Temperature_C", "BMP_Temperature_Smoothed", "Median_Temperature_C"]
    if dht_temp_valid:
        real_temp_cols += ["DHT_Temperature_C", "DHT_Temperature_Smoothed"]

    # Calculate the overall min and max from the real temperature data
    y_min = data[real_temp_cols].min().min()
    y_max = data[real_temp_cols].max().max()
    y_range = y_max - y_min
    y_padding = y_range * 0.05  # 5% padding

    # Apply the limits to the y-axis
    ax1.set_ylim(y_min - y_padding, y_max + y_padding)



    
    
    # Add Fahrenheit Scale
    ax_f = ax1.twinx()
    ax_f.plot(data["Timestamp"], data["BMP_Temperature_C"] * 9/5 + 32, alpha=0)  # Invisible line for correct scaling
    ax_f.set_ylabel("Temperature (°F)", color="red")
    ax_f.tick_params(axis="y", labelcolor="red")





    # Humidity Plot with Cubic Spline
    ax2 = axs[0, 1]
    if dht_humidity_valid:
        ax2.plot(data["Timestamp"], data["DHT_Humidity_percent"], color="green", alpha=0.01)
        ax2.plot(data["Timestamp"], data["DHT_Humidity_percent_Smoothed"], color="green", label="Humidity (%)", alpha=0.7)
        ax2.set_ylabel("Humidity (%)")
        ax2.legend()
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
    else:
        _dead_sensor_notice(ax2, "DHT Sensor (Humidity)")
    ax2.set_title("Humidity" + ("" if dht_humidity_valid else " (DHT offline)"))
    ax2.grid()
    


    # Combined Pressure and Sea-Level Pressure Plot
    ax3 = axs[1, 0]  # Use the left plot in the second row
    ax3_secondary = ax3.twinx()

    # Adjust the rendering order of the twin axes
    ax3_secondary.set_zorder(ax3.get_zorder() - 1)
    ax3.set_facecolor((0, 0, 0, 0))  # Make ax3 background transparent

    # Plot Sea-Level Pressure first (green lines)
    ax3_secondary.plot(data["Timestamp"], data["Sea_Level_Pressure_hPa"]/10, label="Sea-Level Pressure (kPa)", color="green", alpha=0.3, zorder=1)
    ax3_secondary.plot(data["Timestamp"], data["Sea_Level_Pressure_hPa_Smoothed"]/10, color="green", zorder=1)
    ax3_secondary.set_ylabel("Sea-Level Pressure (kPa)", color="green")
    ax3_secondary.tick_params(axis="y", labelcolor="green")
    ax3_secondary.legend(loc="upper right")
    ax3_secondary.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.3f}"))  # Formats numbers to 3 decimal places

    # Plot Pressure on top (red lines)
    ax3.plot(data["Timestamp"], data["BMP_Pressure_hPa"]/10, label="Pressure (kPa)", color="red", alpha=0.3, zorder=2)
    ax3.plot(data["Timestamp"], data["BMP_Pressure_hPa_Smoothed"]/10, color="red", zorder=3)

    ax3.set_title("Pressure and Sea-Level Pressure")
    ax3.set_ylabel("Pressure (kPa)", color="red")
    ax3.tick_params(axis="y", labelcolor="red")
    ax3.legend(loc="upper left")
    ax3.grid()

    # Light Plot
    axs[1, 1].plot(data["Timestamp"], data["BH1750_Light_lx"], label="Light (lx)", color="orange")
    axs[1, 1].plot(data["Timestamp"], data["BH1750_Light_lx_Smoothed"], color="orange", alpha=0.3)
    
    axs[1, 1].set_title("Light")
    axs[1, 1].set_ylabel("Light (lx)")
    axs[1, 1].legend()
    axs[1, 1].grid()

   
    # Heat Index Plot
    ax2 = axs[2, 0]
    if dht_valid:
        ax2.plot(data["Timestamp"], data["Heat_Index"], label="Heat Index", color="orange")
        ax2.set_ylabel("Heat Index")
        ax2.legend()
    else:
        _dead_sensor_notice(ax2, "DHT Sensor (Heat Index)")
    ax2.set_title("Heat Index" + ("" if dht_valid else " (DHT offline)"))
    ax2.grid()
    
    
    
    
    # Dew Point Plot
    ax3 = axs[2, 1]

    if dht_valid:
        # Function to apply a buffer to transitions
        def apply_buffer(data, column_name, threshold=0):
            values = data[column_name].values
            buffered_values = values.copy()
            
            for i in range(1, len(values) - 1):
                # Check for crossing into negative (or positive)
                if (values[i] >= threshold and values[i - 1] < threshold) or \
                   (values[i] < threshold and values[i - 1] >= threshold):
                    # Apply buffer by extending the previous value
                    buffered_values[i + 1] = values[i]
                    buffered_values[i + 2] = values[i+1]
            
            return buffered_values

        # Create separate series with NaNs to avoid connecting lines
        dew_point_above = data["Dew_Point_C_smoothed"].where(data["Dew_Point_C_smoothed"] >= 0)
        dew_point_below = data["Dew_Point_C_smoothed"].where(data["Dew_Point_C_smoothed"] < 0)

        data["Dew_Point_C_Buffered"] = apply_buffer(data, "Dew_Point_C", threshold=0)

        ax3.plot(data["Timestamp"], dew_point_above, label="Dew Point (°C)", color="blue")
        ax3.plot(data["Timestamp"], dew_point_below, label="Frost Point (°C)", color="lightblue", alpha=0.7)

        ax3.set_ylabel("Point Temperature (°C)")
        ax3.legend()

        combined_min = data["Dew_Point_C"].min(skipna=True)
        combined_max = data["Dew_Point_C"].max(skipna=True)
        if pd.isna(combined_min) or pd.isna(combined_max):
            combined_min, combined_max = -10, 40
        if combined_min < 0:
            combined_min *= 1.1
        else:
            combined_min *= 0.9
        if combined_max < 0:
            combined_max *= 0.8
        else:
            combined_max *= 1.1
        if combined_min >= combined_max:
            combined_min, combined_max = -10, 40
        ax3.set_ylim(combined_min, combined_max)
    else:
        _dead_sensor_notice(ax3, "DHT Sensor (Dew/Frost Point)")

    ax3.set_title("Dew and Frost Point" + ("" if dht_valid else " (DHT offline)"))
    ax3.grid()

    # Specific Humidity Plot
    ax5 = axs[3, 0]
    if dht_valid:
        ax5.plot(data["Timestamp"], data["Specific_Humidity_gkg"], label="Specific Humidity (g/kg)", color="brown")
        ax5.set_ylabel("Specific Humidity (g/kg)")
        ax5.legend()
    else:
        _dead_sensor_notice(ax5, "DHT Sensor (Specific Humidity)")
    ax5.set_title("Specific Humidity" + ("" if dht_valid else " (DHT offline)"))
    ax5.grid()

    # Environmental Comfort Index Plot
    ax6 = axs[3, 1]
    if dht_valid:
        ax6.plot(data["Timestamp"], data["ECI"], color="k", linewidth=1.5)
        norm = plt.Normalize(0, 1)
        cmap = plt.cm.get_cmap("RdYlGn")
        timestamps_num = matplotlib.dates.date2num(data["Timestamp"])
        points = np.array([timestamps_num, data["ECI"]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = matplotlib.collections.LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(data["ECI"])
        lc.set_linewidth(1)
        line = ax6.add_collection(lc)
        cbar = plt.colorbar(line, ax=ax6, orientation="vertical", pad=0.02)
        cbar.set_label("Environmental Comfort Index (ECI)")
        ax6.set_ylabel("ECI")
    else:
        _dead_sensor_notice(ax6, "DHT Sensor (ECI)")
    ax6.set_title("Environmental Comfort Index" + ("" if dht_valid else " (DHT offline)"))
    ax6.grid()

    # Formatting
    for ax in axs.flat:
        ax.xaxis.set_tick_params(rotation=45)

    fig.suptitle(title, fontsize=16)

    if out_of_date_flag == 1:
        # Add the warning in big red text
        warning_text = "!!! WARNING: FILE IS OUT OF DATE !!!"
        fig.text(
            0.5, 0.92,  # Position: x=0.5 (centered), y=0.92 (below the title)
            warning_text,
            color="red",
            fontsize=14,
            ha="center",
            va="center",
            fontweight="bold"
        )

    
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for the title
    plt.savefig(output_path)
    plt.close()
    ax1.ticklabel_format(style="plain", axis="y")
    ax2.ticklabel_format(style="plain", axis="y")
    ax3.ticklabel_format(style="plain", axis="y")
    ax5.ticklabel_format(style="plain", axis="y")
    ax6.ticklabel_format(style="plain", axis="y")
    ax1.xaxis.set_major_formatter(DateFormatter("%d/%m - %H:%M"))
    ax2.xaxis.set_major_formatter(DateFormatter("%d/%m - %H:%M"))
    ax3.xaxis.set_major_formatter(DateFormatter("%d/%m - %H:%M"))
    ax5.xaxis.set_major_formatter(DateFormatter("%d/%m - %H:%M"))
    ax6.xaxis.set_major_formatter(DateFormatter("%d/%m - %H:%M"))
    print(f"\tSaved plot to {output_path}.")


def save_last_minute_averages(data, predict_data, output_file, sensor_flags=None):
    if sensor_flags is None:
        sensor_flags = {}
    last_minute_data = data[data["Timestamp"] >= (data["Timestamp"].max() - pd.Timedelta(minutes=1))]

    averages = {
        "Temperature (°C/°F)": f"{last_minute_data['Median_Temperature_C'].mean():.2f}°C / {last_minute_data['Median_Temperature_F'].mean():.2f}°F",
        "Predicted Temp (°C/°F)": f"{predict_data['Predicted_Temperature'].mean():.2f}°C / {predict_data['Predicted_Temperature'].mean() * 9/5 + 32:.2f}°F",
        "Humidity (%)": f"{last_minute_data['DHT_Humidity_percent'].mean():.1f}" if sensor_flags.get("dht_humidity") else "Sensor Offline",
        "Pressure (kPa)": f"{last_minute_data['BMP_Pressure_hPa'].mean() / 10:.3f}",
        "Light (lx)": f"{last_minute_data['BH1750_Light_lx'].mean():.1f}" if sensor_flags.get("bh1750") else "Sensor Offline",
    }

    # Create a DataFrame
    averages_df = pd.DataFrame([averages])

    # Convert to HTML
    html_table = averages_df.to_html(index=False, border=1)
    with open(output_file, "w") as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Last Minute Averages</title>
            <style>
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid black; padding: 8px; text-align: center; }}
            </style>
        </head>
        <body>
            {html_table}
        </body>
        </html>
        """)
    print(f"\t\tSaved last 1-minute averages to {output_file}.")
    print("\n-------------")
    for label, value in averages.items():
        print(label, "\t-\t", value)
    print("-------------\n")




def calculate_rolling_averages(data, time_spans):
    """Calculate rolling averages for each time span and save to CSV and HTML."""
    now = datetime.now(UTC)
    averages = {}

    for label, delta in time_spans.items():
        start_time = now - delta
        subset = data[data["Timestamp"] >= start_time]
        averages[label] = subset.mean(numeric_only=True)

    # Convert to DataFrame and save as CSV
    averages_df = pd.DataFrame(averages).T
    averages_df.index.name = "Time_Span"
    averages_df.to_csv(ROLLING_AVERAGES_FILE)
    print(f"\t\tSaved rolling averages to {ROLLING_AVERAGES_FILE}.")

    # Generate HTML table
    html_table = averages_df.to_html(border=1)
    html_file = ROLLING_AVERAGES_FILE.replace(".csv", ".html")
    with open(html_file, "w") as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rolling Averages</title>
            <style>
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid black; padding: 8px; text-align: center; }}
            </style>
        </head>
        <body>
            <h1>Rolling Averages</h1>
            {html_table}
        </body>
        </html>
        """)
    print(f"\t\tSaved rolling averages HTML to {html_file}.")

    # Ensure the file is readable and writable by others
    os.chmod(ROLLING_AVERAGES_FILE, 0o664)
    os.chmod(html_file, 0o664)




def clean_percentage(series):
    """Clean percentage data by removing '%' and converting to float."""
    return series.str.rstrip('%').astype(float)
    
    

def parse_disk_usage(disk_usage_str):
    """Parse multiple disk usage entries and compute total used and total disk space in GB."""
    try:
        if not isinstance(disk_usage_str, str):
            return 0.0, 0.0

        total_used = 0.0
        total_capacity = 0.0

        # Regex to match disk usage entries in the format <used>GB/<total>GB
        disk_pattern = re.compile(r'([\d.]+)GB/([\d.]+)GB')

        # Find all matches in the string
        matches = disk_pattern.findall(disk_usage_str)

        for used, capacity in matches:
            try:
                total_used += float(used)
                total_capacity += float(capacity)
            except ValueError as e:
                print(f"Skipping malformed entry: used={used}, capacity={capacity} - {e}")
                continue

        return total_used, total_capacity
    except Exception as e:
        print(f"Error parsing disk usage string: {disk_usage_str} - {e}")
        return 0.0, 0.0

def plot_system_stats(csv_file=STATS_FILE, output_image="system_stats_plot.png"):
    """Plot CPU usage, memory usage, and CPU temp from the stats CSV."""
    try:
        df = pd.read_csv(csv_file, parse_dates=["Timestamp"])
        df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").tail(2000)
    except Exception as e:
        logging.warning(f"plot_system_stats: could not read {csv_file}: {e}")
        return

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df["Timestamp"], df["CPU_Usage_pct"],    label="CPU Usage (%)",    color="blue")
    ax1.plot(df["Timestamp"], df["Memory_Usage_pct"], label="Memory Usage (%)", color="green")
    ax1.set_ylabel("Usage (%)")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    if "CPU_Temp_C" in df.columns:
        ax2 = ax1.twinx()
        temps = pd.to_numeric(df["CPU_Temp_C"], errors="coerce")
        ax2.plot(df["Timestamp"], temps, label="CPU Temp (°C)", color="red", alpha=0.7)
        ax2.set_ylabel("CPU Temp (°C)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.legend(loc="upper right")

    ax1.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    plt.xticks(rotation=45)
    plt.title("System Stats")
    plt.tight_layout()
    plt.savefig(output_image)
    plt.close()
    logging.info(f"System stats plot saved to {output_image}")

def calculate_dew_point(temp_c, humidity, pressure_hpa=1013.25):
    """Return dew point (°C) for given temperature, relative humidity and pressure.

    Parameters
    ----------
    temp_c : float
        Air temperature in Celsius.
    humidity : float
        Relative humidity in percent (0-100).
    pressure_hpa : float, optional
        Ambient pressure in hPa. Defaults to standard sea level pressure.

    This implementation uses the Magnus formula for saturation vapour pressure
    with a pressure correction recommended by WMO. The additional pressure
    parameter has a minor effect but ensures calculations are physically
    consistent at non‑standard pressures.
    """
    # Clip unrealistic humidity values
    if humidity <= 0:
        return np.nan

    # Saturation vapour pressure (hPa) using the Magnus approximation
    a = 17.62
    b = 243.12
    es = 6.112 * np.exp((a * temp_c) / (b + temp_c))

    # Pressure adjustment factor (Buck 1981)
    f = 1.0007 + 3.46e-6 * pressure_hpa
    e = humidity / 100.0 * es * f

    alpha = np.log(e / 6.112)
    dew_point = (b * alpha) / (a - alpha)
    return dew_point


def get_file_modification_times(directory="."):
    """
    Retrieves the modification times of files starting with 'weather_plot' 
    in the specified directory.
    
    Args:
        directory (str): The path to the directory. Defaults to the current directory.
        
    Returns:
        list of datetime: A list of modification times as datetime objects.
    """
    file_mod_times = []
    for filename in os.listdir(directory):
        if filename.startswith("weather_plot"):  # Check if the file name starts with 'weather_plot'
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):  # Ensure it's a file
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                file_mod_times.append(mod_time)
    return file_mod_times


def construct_time_spans(directory="."):
    """
    Constructs a dictionary of time spans based on the modification times of files.
    
    Args:
        directory (str): The path to the directory. Defaults to the current directory.
        
    Returns:
        dict: A dictionary of time spans to be included.
    """
    now = datetime.now()
    file_mod_times = get_file_modification_times(directory)
    time_spans = {}
    #print(file_mod_times)
    #if any((now - mod_time) > timedelta(hours=2) for mod_time in file_mod_times):
    #    time_spans["1_week"] = timedelta(weeks=1)
    #    
    #if any((now - mod_time) > timedelta(days=1) for mod_time in file_mod_times):
    #    time_spans["1_year"] = timedelta(days=365)  # Effectively no limit    
    #    time_spans["1_month"] = timedelta(weeks=4)  # Effectively no limit                    
    #    
    #if any((now - mod_time) > timedelta(minutes=30) for mod_time in file_mod_times):
    #    time_spans["1_hour"] = timedelta(hours=1)
    #    
    #if any((now - mod_time) > timedelta(hours=1) for mod_time in file_mod_times):
    #    time_spans["1_day"] = timedelta(days=1)


    #time_spans["all_time"] = timedelta(days=3650)
    time_spans["1_month"] = timedelta(weeks=4)
    time_spans["1_week"] = timedelta(weeks=1)
    time_spans["1_day"] = timedelta(days=1)
    time_spans["1_hour"] = timedelta(hours=1)
    time_spans["10_minutes"] = timedelta(minutes=10)

    return time_spans



def main():
    print("Starting server weather processing script...")

    temp_offset =  - 2.3

    # Define image directory and output path for GIF
    IMAGE_DIR = "/media/bigdata/weather_station/images/"
    GIF_OUTPUT = "/media/bigdata/weather_station/hourly_timelapse.gif"
    DAY_GIF_OUTPUT = "/media/bigdata/weather_station/daily_timelapse.gif"
    
    try:
        gather_system_stats()
    except Exception as e:
        logging.warning(f"gather_system_stats failed: {e}")

    # Example usage
    time_spans = construct_time_spans()

    master_data = load_master_data(MASTER_FILE)
    sensor_flags = check_sensor_validity(master_data)

    master_data.loc[master_data["BMP_Pressure_hPa"] < 500, "BMP_Pressure_hPa"] += 500
    master_data.loc[master_data["BMP_Temperature_C"] < 0, "DHT_Temperature_C"] *= -1
    master_data["DHT_Temperature_C"] = master_data["DHT_Temperature_C"] + temp_offset
    master_data["BMP_Temperature_C"] = master_data["BMP_Temperature_C"] + temp_offset

    feature_cols = ["BMP_Temperature_C", "BMP_Pressure_hPa"]
    if sensor_flags["dht_humidity"]:
        feature_cols = ["DHT_Humidity_percent"] + feature_cols
    
    forecaster = WeatherForecaster(
        master_file=MASTER_FILE,
        hidden_dim=128,
        num_layers=6,
        batch_size=250,
        seq_length=500,
        feature_cols=feature_cols,
    )    
    
    # Define the model file path
    model_path = "/media/bigdata/weather_station/weather_model.pth"
    #forecaster.train_model(epochs=2);forecaster.save_model(model_path)
    # Check if the model file exists and is recent
    current_time = datetime.now()
    current_hour = current_time.hour
    if os.path.exists(model_path):
        # Get the last modification time of the file
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
        # Check if the file is older than a day
        if current_time - file_mod_time > timedelta(days=1):
            if 4 <= current_hour < 5:
                print("Model file is older than a day. Retraining the model...")
                forecaster.train_model(epochs=50)
                forecaster.save_model(model_path)
                forecaster.plot_training_loss("training_loss.csv", "training_loss_plot.png")
                forecaster.plot_final_losses("final_losses.csv", "final_losses_plot.png")
            else:
                print("Model file is outdated but it's outside of training hours. Will train at 4AM.")

        else:
            forecaster.load_model(model_path)
    else:
        print("Model file does not exist. Training the model...")
        forecaster.train_model(epochs=300)
        forecaster.save_model(model_path)



    file_infer_time = datetime.fromtimestamp(os.path.getmtime(PREDICT_FILE))
    
    if True:#datetime.now() - file_infer_time > timedelta(minutes=9):
        steps_ahead=60
        # Use load_master_data to get interpolated, resampled data
        recent_data = forecaster.load_master_data()
        timestamps = pd.to_datetime(recent_data["Timestamp"])  # Already interpolated!
        last_timestamp = timestamps.iloc[-1]
        data = recent_data[["DHT_Humidity_percent", "BMP_Temperature_C", "BMP_Pressure_hPa"]].values
        seq_length = forecaster.seq_length
        last_sequence = data[-seq_length:]
        predictions = forecaster.predict_future(last_sequence, steps_ahead=steps_ahead)
        # Infer future timestamps
        interval_seconds = (timestamps.iloc[-1] - timestamps.iloc[-2]).total_seconds()
        future_timestamps = forecaster.infer_timestamps(last_timestamp, steps_ahead, interval_seconds)
        # Save predictions to CSV
        forecaster.save_predictions_to_csv(predictions, future_timestamps, PREDICT_FILE)


    predict_data = load_master_data(PREDICT_FILE)
    predict_data["Predicted_Temperature"] = predict_data["Predicted_Temperature"] + temp_offset
    rolling_window = 10
    timestamps = np.arange(len(master_data["Timestamp"]))
    humidity = master_data["DHT_Humidity_percent"].values
    smooth_timestamps = np.linspace(0, len(timestamps) - 1, len(timestamps))
    
    master_data["Sea_Level_Pressure_hPa"] = master_data["BMP_Pressure_hPa"] * (1 - (master_data["BMP_Altitude_m"] / 44330.77))**-5.255
    master_data["DHT_Temperature_Smoothed"] = master_data["DHT_Temperature_C"].rolling(window=rolling_window, min_periods=1, center=True).mean()
    master_data["BMP_Temperature_Smoothed"] = master_data["BMP_Temperature_C"].rolling(window=rolling_window, min_periods=1, center=True).mean()
    master_data["BMP_Pressure_hPa_Smoothed"] = master_data["BMP_Pressure_hPa"].rolling(window=rolling_window, min_periods=1, center=True).mean()        
    master_data["Sea_Level_Pressure_hPa_Smoothed"] = master_data["Sea_Level_Pressure_hPa"].rolling(window=rolling_window, min_periods=1, center=True).mean()                
    master_data["BH1750_Light_lx_Smoothed"] = master_data["BH1750_Light_lx"].rolling(window=rolling_window, min_periods=1, center=True).mean()                
    master_data["DHT_Humidity_percent_Smoothed"] = master_data["DHT_Humidity_percent"].rolling(window=rolling_window*2, min_periods=1, center=True).mean()                
    
    temp_cols = ["BMP_Temperature_Smoothed"]
    if sensor_flags["dht_temp"]:
        temp_cols.append("DHT_Temperature_Smoothed")
    master_data["Median_Temperature_C"] = master_data[temp_cols].median(axis=1)    
    
    master_data["Median_Temperature_F"] = master_data["Median_Temperature_C"] * 9 / 5 + 32
    master_data["Dew_Point_C"] = master_data.apply(lambda row: calculate_dew_point(row["Median_Temperature_C"], row["DHT_Humidity_percent"], row["BMP_Pressure_hPa"]),axis=1,)
    master_data["Dew_Point_C_smoothed"] = master_data["Dew_Point_C"].rolling(window=rolling_window, center=True).mean()                

    # Replace infinities with NaNs for numeric columns only
    numeric_cols = master_data.select_dtypes(include=[np.number]).columns
    master_data[numeric_cols] = master_data[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Impute missing values in numeric columns with the median
    master_data[numeric_cols] = master_data[numeric_cols].fillna(master_data[numeric_cols].median())

    # Reset index for a clean DataFrame
    master_data.reset_index(drop=True, inplace=True)

    # Verify no remaining infinities in numeric columns
    assert not np.isinf(master_data[numeric_cols].values).any(), "DataFrame still contains infinite values."

        
    # Calculate the difference between max_timestamp and the current time
    current_time = datetime.now(UTC) 
    max_timestamp = master_data["Timestamp"].max()
    time_difference = current_time - max_timestamp
    
    try:
        save_latest_copy(image_dir = IMAGE_DIR)
    except:
        pass


    out_of_date_flag = 0
    if time_difference > timedelta(hours=1):
        print("!!! WARNING !!!")
        print(f"The file is over an hour out of date! Time difference: {time_difference}")
        print("!!! PLEASE CHECK THE FILE STREAM/RPI !!!")
        out_of_date_flag = 1


    for label, delta in time_spans.items():
        # Use the maximum timestamp in the master_data DataFrame
        # Subset the data based on the maximum timestamp
        subset = master_data[master_data["Timestamp"] >= max_timestamp - delta].copy()
        #subset = master_data[master_data["Timestamp"] >= master_data["Timestamp"].max() - delta]
        generate_plots(subset, predict_data, f"/media/bigdata/weather_station/weather_plot_{label}.png", f"Weather Data ({label.replace('_', ' ').title()})", out_of_date_flag)

    print("\tGenerating GIFs...")
    generate_hourly_gif_with_plot(IMAGE_DIR, GIF_OUTPUT, master_data, sensor_flags)
    generate_daily_gif_with_plot(IMAGE_DIR, DAY_GIF_OUTPUT, master_data, sensor_flags)

    generate_summary_plot(master_data, f"/media/bigdata/weather_station/summary_plot.png", sensor_flags)
    #print("\t\tCalculating rolling averages...")
    #calculate_rolling_averages(master_data, time_spans)
    save_last_minute_averages(master_data, predict_data, "/media/bigdata/weather_station/small_summary.html", sensor_flags)

    print("\tDetecting birds...")
    BIRD_IMAGE_DIR = os.path.join(IMAGE_DIR, "birds")
    
    bird_detection.run_detection_pipeline(
        image_dir=IMAGE_DIR,
        output_dir=BIRD_IMAGE_DIR,
        confidence_threshold=0.35,
        log_file="images/birds/processed_images.json",
        hours_back=2,
        target_classes = ['bird', 'squirrel', 'cat', 'rabbit', 'fox', 'deer', 'raccoon', 'skunk', 'coyote', 'mouse', 'vole', 'chipmunk', 'prairie dog', 'badger', 'weasel','hawk', 'owl', 'magpie', 'crow', 'raven', 'turkey', 'woodpecker']
    )
    
    try:
        plot_system_stats()
    except Exception as e:
        logging.warning(f"plot_system_stats failed: {e}")





if __name__ == "__main__":
    main()
