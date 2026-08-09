#!/usr/bin/env python3
"""Canonical schemas for the Tempestas/Observo weather station.

Keep raw measurements source-specific.  ``Ambient_Temperature_C`` is the
station-level combined/selected ambient temperature produced by the Pi.
Pressure fields are reserved now and may be NaN until a pressure sensor is
installed.
"""

WEATHER_COLUMNS = [
    "Timestamp",
    "Ambient_Temperature_C",
    "DHT22_Temperature_C",
    "DHT22_Humidity_percent",
    "PressureSensor_Temperature_C",
    "Pressure_hPa",
    "Pressure_Altitude_m",
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

# These are diagnostic plausibility ranges, not hard validity limits.  Values
# outside them are ingested but generate warnings so unusual weather or a bad
# sensor cannot silently disappear from the data set.
PLAUSIBILITY_RANGES = {
    "Ambient_Temperature_C": (-60.0, 80.0),
    "DHT22_Temperature_C": (-40.0, 80.0),
    "DHT22_Humidity_percent": (0.0, 100.0),
    "PressureSensor_Temperature_C": (-60.0, 100.0),
    "Pressure_hPa": (250.0, 1150.0),
    "Pressure_Altitude_m": (-1000.0, 10000.0),
    "Camera_Lux": (0.0, 1.0e7),
    "Camera_Lux_Age_s": (0.0, 86400.0),
    "CPU_Temperature_C": (-20.0, 100.0),
    "CPU_Usage_percent": (0.0, 100.0),
    "Memory_Usage_percent": (0.0, 100.0),
    "Disk_Free_GB": (0.0, 1.0e7),
}
