#!/usr/bin/env python3
"""
Alert System - Sends email notifications for weather station conditions and daily summaries.

Implemented alerts:
1. Daily summary of weather data with recent images
2. Weather station data file age alert (>1 hour old)
"""

import os
import datetime
import smtplib
import glob
import numpy as np
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage


# --- Configuration ---
SMTP_USER = "cirrus.noreply@gmail.com"
SMTP_PASS = "jnlaisebvidlrioh"
EMAILS_TO = ["niall.j.miller@gmail.com", "kkatherinegmiller@gmail.com"]
EMAIL_TO  = "niall.j.miller@gmail.com"

WEATHER_DATA_PATH         = "/media/bigdata/weather_station/all_data.csv"
WEATHER_CURRENT_DATA_PATH = "/media/bigdata/weather_station/weather_data.csv"
WEATHER_IMAGE_DIR         = "/media/bigdata/weather_station/images/"
ALERT_LOG_PATH            = "/media/bigdata/weather_station/alerts.log"
SUBSCRIBERS_PATH          = "/media/bigdata/subscribers.txt"
LAST_SUMMARY_FILE         = "/media/bigdata/weather_station/last_daily_summary.txt"

MAX_FILE_AGE_HOURS = 1.0


def get_subscribers():
    try:
        with open(SUBSCRIBERS_PATH) as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return [EMAIL_TO]


# --- Utility Functions ---

def format_value(value):
    """Format a number for display in the email."""
    if isinstance(value, (int, float, np.number)):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.2f}"
    return str(value)


def get_latest_image(image_dir):
    """Return the most recently modified .jpg in image_dir, or None."""
    try:
        files = glob.glob(os.path.join(image_dir, "*.jpg"))
        return max(files, key=os.path.getmtime) if files else None
    except Exception as e:
        print(f"Error getting latest image: {e}")
        return None


def get_weather_summary():
    """Generate summary of weather station data, gracefully handling offline sensors."""
    try:
        df = pd.read_csv(WEATHER_DATA_PATH)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp", ascending=False)
        recent  = df.iloc[0]
        last_24h = df[df["Timestamp"] >= df["Timestamp"].max() - pd.Timedelta(hours=24)]

        def valid(col):
            return col in last_24h.columns and last_24h[col].notna().mean() >= 0.10

        dht_temp_valid = valid("DHT_Temperature_C")
        dht_hum_valid  = valid("DHT_Humidity_percent")
        bh1750_valid   = valid("BH1750_Light_lx")

        bmp_temp_current = recent["BMP_Temperature_C"]
        bmp_temp_max = last_24h["BMP_Temperature_C"].max()
        bmp_temp_min = last_24h["BMP_Temperature_C"].min()
        bmp_temp_avg = last_24h["BMP_Temperature_C"].mean()
        pressure_current = recent["BMP_Pressure_hPa"]
        pressure_avg = last_24h["BMP_Pressure_hPa"].mean()

        if dht_temp_valid:
            dht_temp_current = recent["DHT_Temperature_C"]
            dht_temp_avg = last_24h["DHT_Temperature_C"].mean()
            dht_temp_section = (
                f"DHT Temperature:\n"
                f"  Current: {format_value(dht_temp_current)}°C ({format_value(dht_temp_current * 9/5 + 32)}°F)\n"
                f"  24h Avg: {format_value(dht_temp_avg)}°C ({format_value(dht_temp_avg * 9/5 + 32)}°F)\n"
            )
        else:
            dht_temp_section = "DHT Temperature:\n  Sensor Offline\n"

        if dht_hum_valid:
            humidity_current = recent["DHT_Humidity_percent"]
            humidity_avg = last_24h["DHT_Humidity_percent"].mean()
            humidity_section = (
                f"Humidity:\n"
                f"  Current: {format_value(humidity_current)}%\n"
                f"  24h Avg: {format_value(humidity_avg)}%\n"
            )
        else:
            humidity_section = "Humidity:\n  Sensor Offline\n"

        if bh1750_valid:
            light_current = recent["BH1750_Light_lx"]
            light_avg = last_24h["BH1750_Light_lx"].mean()
            light_max = last_24h["BH1750_Light_lx"].max()
            light_section = (
                f"Light Level:\n"
                f"  Current: {format_value(light_current)} lx\n"
                f"  24h Max: {format_value(light_max)} lx\n"
                f"  24h Avg: {format_value(light_avg)} lx\n"
            )
        else:
            light_section = "Light Level:\n  Sensor Offline\n"

        return f"""
WEATHER STATION SUMMARY
======================
Current Time: {recent['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

BMP Temperature:
  Current: {format_value(bmp_temp_current)}°C ({format_value(bmp_temp_current * 9/5 + 32)}°F)
  24h Max: {format_value(bmp_temp_max)}°C ({format_value(bmp_temp_max * 9/5 + 32)}°F)
  24h Min: {format_value(bmp_temp_min)}°C ({format_value(bmp_temp_min * 9/5 + 32)}°F)
  24h Avg: {format_value(bmp_temp_avg)}°C ({format_value(bmp_temp_avg * 9/5 + 32)}°F)

{dht_temp_section}
{humidity_section}
Pressure:
  Current: {format_value(pressure_current)} hPa
  24h Avg: {format_value(pressure_avg)} hPa

{light_section}"""
    except Exception as e:
        return f"Error generating weather summary: {str(e)}"


# --- Alert Functions ---

def check_weather_data_age():
    """Check if the incoming weather data file is older than MAX_FILE_AGE_HOURS."""
    try:
        if not os.path.exists(WEATHER_CURRENT_DATA_PATH):
            return True, f"WEATHER DATA FILE NOT FOUND: {WEATHER_CURRENT_DATA_PATH} does not exist."
        mod_time = os.path.getmtime(WEATHER_CURRENT_DATA_PATH)
        mod_datetime = datetime.datetime.fromtimestamp(mod_time)
        age_hours = (datetime.datetime.now() - mod_datetime).total_seconds() / 3600
        if age_hours > MAX_FILE_AGE_HOURS:
            return True, (
                f"WEATHER DATA AGE ALERT: data file is {age_hours:.2f} hours old.\n\n"
                f"Last update: {mod_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Threshold: {MAX_FILE_AGE_HOURS} hour(s)."
            )
        return False, ""
    except Exception as e:
        return True, f"ERROR checking weather data age: {str(e)}"


def check_daily_summary():
    """Return True if no summary has been sent in the last 12-30 hours."""
    now = datetime.datetime.now()
    if os.path.exists(LAST_SUMMARY_FILE):
        try:
            with open(LAST_SUMMARY_FILE) as f:
                last_sent = datetime.datetime.fromisoformat(f.read().strip())
            elapsed = now - last_sent
            if elapsed < datetime.timedelta(hours=np.random.uniform(12, 30)):
                print(f"Skipping daily summary - last one sent {elapsed.total_seconds() / 3600:.1f} hours ago")
                return False, "Daily summary sent recently"
        except Exception as e:
            print(f"Error reading last summary timestamp: {e}")
    try:
        with open(LAST_SUMMARY_FILE, "w") as f:
            f.write(now.isoformat())
    except Exception as e:
        print(f"Error updating last summary timestamp: {e}")
    return True, f"Daily summary report for {now.strftime('%Y-%m-%d')}"


# --- Email / Logging ---

def send_email_with_images(subject, body, email_to=None, image_paths=None):
    """Send a plain-text email with optional image attachments."""
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"]    = SMTP_USER
    msg["To"]      = email_to or EMAIL_TO
    msg.attach(MIMEText(body, "plain"))

    for img_path in (image_paths or []):
        if img_path and os.path.exists(img_path):
            try:
                with open(img_path, "rb") as f:
                    img = MIMEImage(f.read())
                img.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(img_path)}"')
                msg.attach(img)
            except Exception as e:
                print(f"Error attaching image {img_path}: {e}")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False


def log_alert(alert_type, message, sent):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "SENT" if sent else "FAILED"
    with open(ALERT_LOG_PATH, "a") as f:
        f.write(f"{timestamp} | {alert_type} | {status} | {message.splitlines()[0]}\n")


# --- Main ---

def main():
    print(f"Starting alert check at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    sub_emails = get_subscribers()
    print(f"Total of email subscribers: {len(sub_emails)}")

    # Weather data age alert
    triggered, message = check_weather_data_age()
    if triggered:
        print(f"WEATHER DATA AGE ALERT: {message}")
        body = (
            f"WEATHER STATION DATA ALERT\n"
            f"=========================\n\n"
            f"{message}\n\n"
            f"This may indicate the weather station Raspberry Pi has stopped working.\n\n"
            f"Generated at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        sent = False
        for addr in EMAILS_TO:
            sent = send_email_with_images("ALERT: Weather Station Data File Age", body, email_to=addr)
        log_alert("WEATHER_DATA_AGE", message, sent)
    else:
        print("Weather data file age normal, no alerts.")

    # Daily summary
    triggered, message = check_daily_summary()
    if triggered:
        print("Generating daily summary report...")
        weather_summary = get_weather_summary()
        weather_image = get_latest_image(WEATHER_IMAGE_DIR)
        subject = f"Daily Weather Summary - {datetime.datetime.now().strftime('%Y-%m-%d')}"
        body = (
            f"DAILY SUMMARY REPORT\n"
            f"===================\n"
            f"Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"\n{weather_summary}\n"
        )
        sent = False
        for addr in sub_emails:
            print(addr)
            sent = send_email_with_images(subject, body, email_to=addr, image_paths=[weather_image])
        log_alert("DAILY_SUMMARY", message, sent)
        print("Daily summary email sent.")

    print("Alert check completed.")


if __name__ == "__main__":
    main()
