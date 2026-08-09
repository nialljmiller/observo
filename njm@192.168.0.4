import time
import board
import busio
import Adafruit_BMP.BMP085 as BMP085
import adafruit_dht
import adafruit_bh1750
import csv
import os
from datetime import datetime
import subprocess
import sys
import pandas as pd
import psutil  # For CPU and memory usage

def get_cpu_temp():
    # Reads the CPU temperature
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = int(f.read()) / 1000  # Convert millidegrees to Celsius
        return temp
    except FileNotFoundError:
        return "Unavailable"

def get_cpu_usage():
    # Gets the CPU usage as a percentage
    return psutil.cpu_percent(interval=1)

def get_memory_usage():
    # Gets the memory usage as a percentage
    memory = psutil.virtual_memory()
    return memory.percent

sys.stdout.reconfigure(line_buffering=True)

# Initialize sensors
bmp_sensor = BMP085.BMP085()
dht_sensor = adafruit_dht.DHT11(board.D4)
i2c = busio.I2C(board.SCL, board.SDA)
light_sensor = adafruit_bh1750.BH1750(i2c)

# File paths
system_csv_file = "/home/njm/system_usage.csv"
system_server_csv_path = "/media/bigdata/weather_station/system_usage.csv"  # File path on server
local_csv = "/home/njm/weather_data.csv"  # File on Raspberry Pi
server_csv_path = "/media/bigdata/weather_station/weather_data.csv"  # File path on server
server_address = "nill@nillmill.ddns.net"  # Server address

# Ensure local CSV exists
if not os.path.exists(local_csv):
    with open(local_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Timestamp", "BMP_Temperature_C", "BMP_Pressure_hPa", "BMP_Altitude_m",
                         "DHT_Temperature_C", "DHT_Humidity_percent", "BH1750_Light_lx"])

# Check if the file exists, and if not, write the header
if not os.path.exists(system_csv_file):
    with open(system_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "CPU Temperature (°C)", "CPU Usage (%)", "Memory Usage (%)"])

print("Weather Station Initialized! Harvesting data...\n")
write_timer = time.time()

delete_counter = 0

while True:
    try:
        # Gather data
        timestamp = datetime.now(pd.Timestamp.utcnow().tzinfo)
        #datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        temperature_bmp = bmp_sensor.read_temperature()
        pressure = bmp_sensor.read_pressure() / 100  # Convert to hPa
        altitude = bmp_sensor.read_altitude()
        temperature_dht = dht_sensor.temperature
        humidity = dht_sensor.humidity
        light_level = light_sensor.lux

        cpu_temp = get_cpu_temp()
        cpu_usage = get_cpu_usage()
        memory_usage = get_memory_usage()


        # Log data locally
        with open(local_csv, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([timestamp, temperature_bmp, pressure, altitude,
                             temperature_dht, humidity, light_level])

        # Append the readings to the CSV file
        with open(system_csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, cpu_temp, cpu_usage, memory_usage])


        print(f"Data logged at {timestamp}")
        print(f"BMP Temperature: {temperature_bmp:.2f} °C, Pressure: {pressure:.2f} hPa, Altitude: {altitude:.2f} m")
        print(f"DHT Temperature: {temperature_dht:.2f} °C, Humidity: {humidity:.2f} %")
        print(f"BH1750 Light: {light_level:.2f} lx\n")
                # Populate the variables
    
        # Print the results
        print(f"CPU Temperature: {cpu_temp}°C")
        print(f"CPU Usage: {cpu_usage}%")
        print(f"Memory Usage: {memory_usage}%")

        # Transfer data to server every minute
        time_to_write = time.time() - write_timer
        print(time_to_write)

        if time_to_write > 60:  # Every minute
            write_timer = time.time()
            print("Transferring data to the server...")
            try:
                subprocess.run(
                    ["scp", local_csv, f"{server_address}:{server_csv_path}"],
                    check=True
                )

                subprocess.run(
                    ["scp", system_csv_file, f"{server_address}:{system_server_csv_path}"],
                    check=True
                )
                print("Data successfully transferred to the server.")
                delete_counter = delete_counter + 1
            
                if delete_counter > 600:
                    os.remove("weather_station.log")
                    delete_counter = 0
                    # Clear local CSV file after successful transfer
                    with open(local_csv, "w", newline="") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(["Timestamp", "BMP_Temperature_C", "BMP_Pressure_hPa", "BMP_Altitude_m",
                                         "DHT_Temperature_C", "DHT_Humidity_percent", "BH1750_Light_lx"])


                    with open(system_csv_file, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(["Timestamp", "CPU Temperature (°C)", "CPU Usage (%)", "Memory Usage (%)"])

                        print("Local data cleared to save space.\n")

            except subprocess.CalledProcessError as e:
                print(f"Error transferring data to the server: {e}\n")

        time.sleep(10)  # Wait 10 seconds before the next reading

    except RuntimeError as e:
        # Handle sensor read errors
        print(f"Sensor error: {e}")
        time.sleep(2)
    except Exception as e:
        print(f"Unexpected error: {e}")
        break

