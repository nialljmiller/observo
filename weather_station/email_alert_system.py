#!/usr/bin/env python3
"""
Alert System - Sends email notifications for plant station conditions and daily summaries.

Implemented alerts:
1. High temperature alert (40°C or higher)
2. Daily summary of weather and plant data with recent images
3. Plant station data file age alert (>1 hour old)
4. Weather station data file age alert (>1 hour old)

This script can be run manually and later automated.
"""

import os
import sys
import pandas as pd
import numpy as np
import datetime
import smtplib
import glob
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage


def get_subscribers(file_path='/media/bigdata/subscribers.txt'):
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return [EMAIL_TO]



# --- Configuration ---
SMTP_USER = "cirrus.noreply@gmail.com"
SMTP_PASS = "jnlaisebvidlrioh"
SUB_EMAILS_TO = get_subscribers(file_path='/media/bigdata/subscribers.txt')
EMAILS_TO = ["niall.j.miller@gmail.com","kkatherinegmiller@gmail.com"]
EMAIL_TO = "niall.j.miller@gmail.com"
print("Total of email subscribers:", len(SUB_EMAILS_TO))
# File paths
PLANT_DATA_PATH = "/media/bigdata/plant_station/all_plant_data.csv"
WEATHER_DATA_PATH = "/media/bigdata/weather_station/all_data.csv"
PLANT_CURRENT_DATA_PATH = "/media/bigdata/plant_station/plant_data.csv"
WEATHER_CURRENT_DATA_PATH = "/media/bigdata/weather_station/weather_data.csv"
ALERT_LOG_PATH = "/media/bigdata/plant_station/alerts.log"

# Image directories
PLANT_IMAGE_DIR = "/media/bigdata/plant_station/images/"
WEATHER_IMAGE_DIR = "/media/bigdata/weather_station/images/"

# Alert thresholds
HIGH_TEMP_THRESHOLD = 40.0  # in Celsius
MAX_FILE_AGE_HOURS = 1.0    # Maximum file age in hours

# --- Utility Functions ---
def check_daily_summary():
    """
    Trigger for sending the daily summary email.
    This function returns True only if the last summary was sent more than 12 hours ago.
    
    Returns: (bool, str) - Alert triggered, Alert message
    """
    # Define the path to the file that will store the last sent timestamp
    last_summary_file = "/media/bigdata/plant_station/last_daily_summary.txt"
    
    # Current time
    now = datetime.datetime.now()
    
    # Check if the file exists and read the last sent timestamp
    if os.path.exists(last_summary_file):
        try:
            with open(last_summary_file, "r") as f:
                last_sent_str = f.read().strip()
                last_sent = datetime.datetime.fromisoformat(last_sent_str)
                
                # Calculate the time elapsed since the last summary
                time_elapsed = now - last_sent
                
                # If less than 12 hours have passed, don't send a new summary
                if time_elapsed < datetime.timedelta(hours=np.random.uniform(12,30)):
                    print(f"Skipping daily summary - last one sent {time_elapsed.total_seconds() / 3600:.1f} hours ago")
                    return False, f"Daily summary was sent less than 12 hours ago"
        except Exception as e:
            # If there's an error reading the file, log it and proceed to send a summary
            print(f"Error reading last summary timestamp: {e}")
    
    # If we get here, either the file doesn't exist, or more than 12 hours have passed
    # Update the timestamp file
    try:
        with open(last_summary_file, "w") as f:
            f.write(now.isoformat())
    except Exception as e:
        print(f"Error updating last summary timestamp: {e}")
    
    # Return True to trigger sending the summary
    message = f"Daily summary report for {now.strftime('%Y-%m-%d')}"
    return True, message

def get_latest_image(image_dir):
    """Get the most recent image from a directory."""
    try:
        image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
        if not image_files:
            return None
        
        # Get the most recent file based on modification time
        latest_file = max(image_files, key=os.path.getmtime)
        return latest_file
    except Exception as e:
        print(f"Error getting latest image: {e}")
        return None

def format_value(value):
    """Format a number for display in the email."""
    if isinstance(value, (int, float, np.number)):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.2f}"
    return str(value)

def get_plant_summary():
    """Generate summary of plant station data."""
    try:
        # Load plant data
        df = pd.read_csv(PLANT_DATA_PATH)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        
        # Get most recent data and last 24 hours
        df = df.sort_values("Timestamp", ascending=False)
        recent = df.iloc[0]
        last_24h = df[df["Timestamp"] >= (df["Timestamp"].max() - pd.Timedelta(hours=24))]
        
        # Calculate statistics
        temp_current = recent["Temperature_C"]
        temp_max = last_24h["Temperature_C"].max()
        temp_min = last_24h["Temperature_C"].min()
        temp_avg = last_24h["Temperature_C"].mean()
        
        humidity_current = recent["Humidity_percent"]
        humidity_avg = last_24h["Humidity_percent"].mean()
        
        pressure_current = recent["Pressure_hPa"]
        pressure_avg = last_24h["Pressure_hPa"].mean()
        
        # Soil moisture info
        soil_moisture = []
        for i in range(1, 5):
            col = f"Soil_Moisture_{i}"
            if col in recent:
                current = recent[col]
                avg = last_24h[col].mean()
                soil_moisture.append((i, current, avg))
        
        # Create summary text
        summary = f"""
PLANT STATION SUMMARY
=====================
Current Time: {recent['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

Temperature:
  Current: {format_value(temp_current)}°C ({format_value(temp_current * 9/5 + 32)}°F)
  24h Max: {format_value(temp_max)}°C ({format_value(temp_max * 9/5 + 32)}°F)
  24h Min: {format_value(temp_min)}°C ({format_value(temp_min * 9/5 + 32)}°F)
  24h Avg: {format_value(temp_avg)}°C ({format_value(temp_avg * 9/5 + 32)}°F)

Humidity:
  Current: {format_value(humidity_current)}%
  24h Avg: {format_value(humidity_avg)}%

Pressure:
  Current: {format_value(pressure_current)} hPa
  24h Avg: {format_value(pressure_avg)} hPa
"""
        
        # Add soil moisture data if available
        if soil_moisture:
            summary += "\nSoil Moisture:\n"
            for i, current, avg in soil_moisture:
                summary += f"  Sensor {i}: Current {format_value(current)}, 24h Avg {format_value(avg)}\n"
        
        return summary
    except Exception as e:
        return f"Error generating plant summary: {str(e)}"

def get_weather_summary():
    """Generate summary of weather station data."""
    try:
        # Load weather data
        df = pd.read_csv(WEATHER_DATA_PATH)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        
        # Get most recent data and last 24 hours
        df = df.sort_values("Timestamp", ascending=False)
        recent = df.iloc[0]
        last_24h = df[df["Timestamp"] >= (df["Timestamp"].max() - pd.Timedelta(hours=24))]
        
        # Calculate statistics
        bmp_temp_current = recent["BMP_Temperature_C"]
        bmp_temp_max = last_24h["BMP_Temperature_C"].max()
        bmp_temp_min = last_24h["BMP_Temperature_C"].min()
        bmp_temp_avg = last_24h["BMP_Temperature_C"].mean()
        
        dht_temp_current = recent["DHT_Temperature_C"]
        dht_temp_avg = last_24h["DHT_Temperature_C"].mean()
        
        humidity_current = recent["DHT_Humidity_percent"]
        humidity_avg = last_24h["DHT_Humidity_percent"].mean()
        
        pressure_current = recent["BMP_Pressure_hPa"]
        pressure_avg = last_24h["BMP_Pressure_hPa"].mean()
        
        light_current = recent["BH1750_Light_lx"]
        light_avg = last_24h["BH1750_Light_lx"].mean()
        light_max = last_24h["BH1750_Light_lx"].max()
        
        # Create summary text
        summary = f"""
WEATHER STATION SUMMARY
======================
Current Time: {recent['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

BMP Temperature:
  Current: {format_value(bmp_temp_current)}°C ({format_value(bmp_temp_current * 9/5 + 32)}°F)
  24h Max: {format_value(bmp_temp_max)}°C ({format_value(bmp_temp_max * 9/5 + 32)}°F)
  24h Min: {format_value(bmp_temp_min)}°C ({format_value(bmp_temp_min * 9/5 + 32)}°F)
  24h Avg: {format_value(bmp_temp_avg)}°C ({format_value(bmp_temp_avg * 9/5 + 32)}°F)

DHT Temperature:
  Current: {format_value(dht_temp_current)}°C ({format_value(dht_temp_current * 9/5 + 32)}°F)
  24h Avg: {format_value(dht_temp_avg)}°C ({format_value(dht_temp_avg * 9/5 + 32)}°F)

Humidity:
  Current: {format_value(humidity_current)}%
  24h Avg: {format_value(humidity_avg)}%

Pressure:
  Current: {format_value(pressure_current)} hPa
  24h Avg: {format_value(pressure_avg)} hPa

Light Level:
  Current: {format_value(light_current)} lx
  24h Max: {format_value(light_max)} lx
  24h Avg: {format_value(light_avg)} lx
"""
        return summary
    except Exception as e:
        return f"Error generating weather summary: {str(e)}"

# --- Alert Functions ---

def check_high_temperature():
    """
    Check if plant station temperature exceeds threshold.
    Returns: (bool, str) - Alert triggered, Alert message
    """
    try:
        # Load the most recent data
        df = pd.read_csv(PLANT_DATA_PATH)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        
        # Get most recent readings
        recent_data = df.sort_values("Timestamp", ascending=False).head(5)
        max_temp = recent_data["Temperature_C"].max()
        
        if max_temp >= HIGH_TEMP_THRESHOLD:
            timestamp = recent_data.loc[recent_data["Temperature_C"].idxmax(), "Timestamp"]
            message = (f"HIGH TEMPERATURE ALERT: Plant station recorded {max_temp:.1f}°C "
                      f"at {timestamp}\n\n"
                      f"This exceeds the alert threshold of {HIGH_TEMP_THRESHOLD}°C.")
            return True, message
        return False, ""
    except Exception as e:
        return True, f"ERROR checking temperature: {str(e)}"

def check_plant_data_age():
    """
    Check if plant station data file is older than the threshold.
    
    Returns: (bool, str) - Alert triggered, Alert message
    """
    try:
        if not os.path.exists(PLANT_CURRENT_DATA_PATH):
            return True, f"PLANT DATA FILE NOT FOUND: {PLANT_CURRENT_DATA_PATH} does not exist."
        
        # Get file modification time
        mod_time = os.path.getmtime(PLANT_CURRENT_DATA_PATH)
        mod_datetime = datetime.datetime.fromtimestamp(mod_time)
        now = datetime.datetime.now()
        age_hours = (now - mod_datetime).total_seconds() / 3600
        
        if age_hours > MAX_FILE_AGE_HOURS:
            message = (f"PLANT DATA AGE ALERT: Plant station data file is {age_hours:.2f} hours old.\n\n"
                      f"Last update: {mod_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n"
                      f"This exceeds the threshold of {MAX_FILE_AGE_HOURS} hour(s).")
            return True, message
        return False, ""
    except Exception as e:
        return True, f"ERROR checking plant data age: {str(e)}"

def check_weather_data_age():
    """
    Check if weather station data file is older than the threshold.
    
    Returns: (bool, str) - Alert triggered, Alert message
    """
    try:
        if not os.path.exists(WEATHER_CURRENT_DATA_PATH):
            return True, f"WEATHER DATA FILE NOT FOUND: {WEATHER_CURRENT_DATA_PATH} does not exist."
        
        # Get file modification time
        mod_time = os.path.getmtime(WEATHER_CURRENT_DATA_PATH)
        mod_datetime = datetime.datetime.fromtimestamp(mod_time)
        now = datetime.datetime.now()
        age_hours = (now - mod_datetime).total_seconds() / 3600
        
        if age_hours > MAX_FILE_AGE_HOURS:
            message = (f"WEATHER DATA AGE ALERT: Weather station data file is {age_hours:.2f} hours old.\n\n"
                      f"Last update: {mod_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n"
                      f"This exceeds the threshold of {MAX_FILE_AGE_HOURS} hour(s).")
            return True, message
        return False, ""
    except Exception as e:
        return True, f"ERROR checking weather data age: {str(e)}"

# Add more check functions here for future alert conditions

def send_email_with_images(subject, body, email_to = None, image_paths=None):
    """Send email with text and optional attached images."""
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = SMTP_USER

    if email_to == None:
        msg["To"] = EMAIL_TO
        
    msg["To"] = email_to

    # Attach the text body
    msg.attach(MIMEText(body, "plain"))
    
    # Attach images if provided
    if image_paths:
        for i, img_path in enumerate(image_paths):
            if img_path and os.path.exists(img_path):
                try:
                    with open(img_path, 'rb') as img_file:
                        img_data = img_file.read()
                        img = MIMEImage(img_data)
                        img_name = os.path.basename(img_path)
                        img.add_header('Content-Disposition', f'attachment; filename="{img_name}"')
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
    """Log alert to file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "SENT" if sent else "FAILED"
    with open(ALERT_LOG_PATH, "a") as f:
        f.write(f"{timestamp} | {alert_type} | {status} | {message.splitlines()[0]}\n")

def main():
    """Main function to check conditions and send alerts."""
    print(f"Starting alert check at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check for high temperature
    triggered, message = check_high_temperature()
    if triggered:
        print(f"HIGH TEMPERATURE ALERT: {message}")
        subject = "ALERT: Plant Station High Temperature"
        email_body = f"""
PLANT STATION ALERT
===================

{message}

This automatic alert was generated at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        for cur_email in EMAILS_TO:
            sent = send_email_with_images(subject, email_body, email_to = cur_email)

        log_alert("HIGH_TEMP", message, sent)
    else:
        print("Temperature normal, no alerts.")
    
    # Check for plant data age
    triggered, message = check_plant_data_age()
    if triggered:
        print(f"PLANT DATA AGE ALERT: {message}")
        subject = "ALERT: Plant Station Data File Age"
        email_body = f"""
PLANT STATION DATA ALERT
=======================

{message}

This may indicate that the plant station Raspberry Pi has stopped working or is unable to
upload new data. Please check the system.

This automatic alert was generated at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        for cur_email in EMAILS_TO:
            sent = send_email_with_images(subject, email_body, email_to = cur_email)
        
        log_alert("PLANT_DATA_AGE", message, sent)
    else:
        print("Plant data file age normal, no alerts.")
    
    # Check for weather data age
    triggered, message = check_weather_data_age()
    if triggered:
        print(f"WEATHER DATA AGE ALERT: {message}")
        subject = "ALERT: Weather Station Data File Age"
        email_body = f"""
WEATHER STATION DATA ALERT
=========================

{message}

This may indicate that the weather station Raspberry Pi has stopped working or is unable to
upload new data. Please check the system.

This automatic alert was generated at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        for cur_email in EMAILS_TO:
            sent = send_email_with_images(subject, email_body, email_to = cur_email)
        
        log_alert("WEATHER_DATA_AGE", message, sent)
    else:
        print("Weather data file age normal, no alerts.")
    
    # Check for daily summary (always triggers when run)
    triggered, message = check_daily_summary()
    if triggered:
        print("Generating daily summary report...")
        
        # Generate summaries
        plant_summary = get_plant_summary()
        weather_summary = get_weather_summary()
        
        # Get latest images
        plant_image = get_latest_image(PLANT_IMAGE_DIR)
        weather_image = get_latest_image(WEATHER_IMAGE_DIR)
        
        # Build email body
        subject = f"Daily Plant & Weather Summary - {datetime.datetime.now().strftime('%Y-%m-%d')}"
        email_body = f"""
DAILY SUMMARY REPORT
===================
Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{weather_summary}

{plant_summary}

Note: The most recent images from both stations are attached to this email.
        """
        for cur_email in SUB_EMAILS_TO:
            print(cur_email)
            sent = send_email_with_images(subject, email_body, email_to = cur_email, image_paths= [weather_image, plant_image])

        log_alert("DAILY_SUMMARY", message, sent)
        print("Daily summary email sent.")
    
    # Add more conditions here for future alerts
        
    print("Alert check completed.")

if __name__ == "__main__":
    main()
