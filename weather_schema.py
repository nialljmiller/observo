#!/usr/bin/env python3
"""Canonical schema for Tempestas/Observo weather-station data v2.

Version 2 adds the BME680 while retaining the DHT22 as an independent sensor.
All source measurements are stored. ``Combined_*`` contains the equal-weight
candidate combination produced by the Pi; ``Ambient_*`` remains the current
station-level/default reading selected by the Pi.

This is a clean-deployment schema. No legacy DHT11/BMP/BH1750 or earlier
DHT22-only columns are migrated.
"""

WEATHER_COLUMNS = [
    "Timestamp",
    "Ambient_Temperature_C",
    "Ambient_Humidity_percent",
    "Combined_Temperature_C",
    "Combined_Humidity_percent",
    "DHT22_Temperature_C",
    "DHT22_Humidity_percent",
    "BME680_Temperature_C",
    "BME680_Humidity_percent",
    "BME680_Pressure_hPa",
    "BME680_Gas_Resistance_ohm",
    "BME680_Altitude_m",
    "Camera_Lux",
    "Camera_Lux_Age_s",
]

WEATHER_NUMERIC_COLUMNS = [
    column for column in WEATHER_COLUMNS if column != "Timestamp"
]

SYSTEM_COLUMNS = [
    "Timestamp",
    "CPU_Temperature_C",
    "CPU_Usage_percent",
    "Memory_Usage_percent",
    "Disk_Free_GB",
    "Pi_Throttled_Hex",
]

SYSTEM_NUMERIC_COLUMNS = [
    "CPU_Temperature_C",
    "CPU_Usage_percent",
    "Memory_Usage_percent",
    "Disk_Free_GB",
]

# Diagnostic ranges only. Out-of-range values are warned about but retained so
# unusual conditions or a failing sensor are visible rather than silently lost.
PLAUSIBILITY_RANGES = {
    "Ambient_Temperature_C": (-60.0, 85.0),
    "Ambient_Humidity_percent": (0.0, 100.0),
    "Combined_Temperature_C": (-60.0, 85.0),
    "Combined_Humidity_percent": (0.0, 100.0),
    "DHT22_Temperature_C": (-40.0, 80.0),
    "DHT22_Humidity_percent": (0.0, 100.0),
    "BME680_Temperature_C": (-40.0, 85.0),
    "BME680_Humidity_percent": (0.0, 100.0),
    "BME680_Pressure_hPa": (300.0, 1100.0),
    "BME680_Gas_Resistance_ohm": (1.0, 100_000_000.0),
    "BME680_Altitude_m": (-1000.0, 10000.0),
    "Camera_Lux": (0.0, 1.0e7),
    "Camera_Lux_Age_s": (0.0, 86400.0),
    "CPU_Temperature_C": (-20.0, 100.0),
    "CPU_Usage_percent": (0.0, 100.0),
    "Memory_Usage_percent": (0.0, 100.0),
    "Disk_Free_GB": (0.0, 1.0e7),
}
