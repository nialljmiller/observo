#!/usr/bin/env python3
"""Process and plot Tempestas weather-station data v2.

The canonical v2 stream stores DHT22 and BME680 temperature/humidity
independently, a candidate equal-weight combination of each pair, BME680
pressure/gas/altitude, and camera metadata. ``Ambient_*`` remains the current
station-level/default value chosen by the Pi.

Unavailable sensors remain unavailable: this processor never fabricates
environmental measurements, and failure of one sensor does not stop products
from the others being updated. UTC measurements are displayed in
America/New_York.
"""

import csv
import glob
import json
import logging
import os
import shutil
from datetime import datetime, timedelta, timezone
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from PIL import Image, ImageDraw, ImageFont
from zoneinfo import ZoneInfo

from weather_forcast import WeatherForecaster
from weather_schema import WEATHER_COLUMNS, WEATHER_NUMERIC_COLUMNS

try:
    import bird_detection
except Exception as exc:
    bird_detection = None
    _BIRD_IMPORT_ERROR = exc
else:
    _BIRD_IMPORT_ERROR = None

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

BASE_DIR = "/media/bigdata/weather_station"
MASTER_FILE = os.path.join(BASE_DIR, "all_data.csv")
MASTER_FILE_JSON = os.path.join(BASE_DIR, "all_data.json")
PREDICT_FILE = os.path.join(BASE_DIR, "predictions.csv")
MODEL_FILE = os.path.join(BASE_DIR, "weather_model_v2_bme680.pth")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
HOURLY_GIF = os.path.join(BASE_DIR, "hourly_timelapse.gif")
DAILY_GIF = os.path.join(BASE_DIR, "daily_timelapse.gif")
PI_SYSTEM_FILE = os.path.join(BASE_DIR, "all_system_data.csv")
SERVER_STATS_FILE = os.path.join(BASE_DIR, "my_pc_stats.csv")

LOCAL_TZ = ZoneInfo("America/New_York")
SMOOTH_WINDOW = "5min"
HUMIDITY_SMOOTH_WINDOW = "5min"
MAX_CAMERA_LUX_AGE_S = 15 * 60
FORECAST_MIN_RESAMPLED_ROWS = 7 * 24 * 60  # one week of 1-minute data
FORECAST_SEQUENCE_MINUTES = 360  # six hours of recent context
FORECAST_STEPS = 60

TIME_SPANS = {
    "1_month": timedelta(weeks=4),
    "1_week": timedelta(weeks=1),
    "1_day": timedelta(days=1),
    "1_hour": timedelta(hours=1),
    "10_minutes": timedelta(minutes=10),
}


def _safe_atomic_text_write(path: str, text: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_master_data(fp: str) -> pd.DataFrame:
    if not os.path.exists(fp):
        raise FileNotFoundError(fp)

    data = pd.read_csv(fp, on_bad_lines="warn")
    missing = [c for c in WEATHER_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"Master weather file missing canonical column(s): {missing}")

    data = data[WEATHER_COLUMNS].copy()
    data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce", utc=True)
    for col in WEATHER_NUMERIC_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    data = data.drop_duplicates(subset=["Timestamp"], keep="last").reset_index(drop=True)
    return data


def generate_json_from_data(data: pd.DataFrame, path: str) -> None:
    output = data.copy()
    output["Timestamp"] = output["Timestamp"].map(
        lambda x: x.isoformat() if pd.notna(x) else None
    )
    records = json.loads(output.to_json(orient="records"))
    _safe_atomic_text_write(path, json.dumps(records, separators=(",", ":")))


def _sensor_current(
    df: pd.DataFrame,
    column: str,
    *,
    max_age: str = "2min",
    window: str = "15min",
    min_count: int = 2,
) -> bool:
    """Return True only when a sensor has produced genuinely recent data.

    This intentionally measures *current* health rather than the fraction of valid
    values in several hours of history.  A failed sensor therefore goes offline
    quickly, and a recovered sensor comes back online after a couple of good rows.
    """
    if df.empty or column not in df.columns:
        return False

    latest_stream = df["Timestamp"].max()
    recent = df[df["Timestamp"] >= latest_stream - pd.Timedelta(window)]
    values = pd.to_numeric(recent[column], errors="coerce")
    valid_mask = values.notna()
    if int(valid_mask.sum()) < min_count:
        return False

    last_valid = recent.loc[valid_mask, "Timestamp"].max()
    return (latest_stream - last_valid) <= pd.Timedelta(max_age)


def check_sensor_validity(df: pd.DataFrame) -> dict:
    flags = {
        "ambient_temp": _sensor_current(df, "Ambient_Temperature_C"),
        "ambient_humidity": _sensor_current(df, "Ambient_Humidity_percent"),
        "combined_temp": _sensor_current(df, "Combined_Temperature_C"),
        "combined_humidity": _sensor_current(df, "Combined_Humidity_percent"),
        "dht22_temp": _sensor_current(df, "DHT22_Temperature_C"),
        "dht22_humidity": _sensor_current(df, "DHT22_Humidity_percent"),
        "bme680_temp": _sensor_current(df, "BME680_Temperature_C"),
        "bme680_humidity": _sensor_current(df, "BME680_Humidity_percent"),
        "bme680_pressure": _sensor_current(df, "BME680_Pressure_hPa"),
        "bme680_gas": _sensor_current(df, "BME680_Gas_Resistance_ohm"),
        "bme680_altitude": _sensor_current(df, "BME680_Altitude_m"),
        "camera_lux": _sensor_current(
            df, "Camera_Lux", max_age="15min", window="30min", min_count=1
        ),
    }
    flags["ambient"] = flags["ambient_temp"] and flags["ambient_humidity"]
    flags["dht22"] = flags["dht22_temp"] and flags["dht22_humidity"]
    flags["bme680"] = (
        flags["bme680_temp"]
        and flags["bme680_humidity"]
        and flags["bme680_pressure"]
    )

    for name, okay in flags.items():
        if name in {"ambient", "dht22", "bme680"}:
            continue
        logging.info("Sensor %-20s %s", name, "available" if okay else "unavailable")
    return flags


def _time_rolling_mean(data: pd.DataFrame, column: str, window: str) -> pd.Series:
    s = pd.Series(
        pd.to_numeric(data[column], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(data["Timestamp"]),
    )
    return s.rolling(window=window, min_periods=1, center=True).mean().to_numpy()


def calculate_dew_point(temp_c, humidity):
    """Magnus dew point in °C from air temperature and RH."""
    t = np.asarray(temp_c, dtype=float)
    rh = np.asarray(humidity, dtype=float)
    out = np.full(np.broadcast(t, rh).shape, np.nan, dtype=float)
    valid = np.isfinite(t) & np.isfinite(rh) & (rh > 0) & (rh <= 100)
    if np.any(valid):
        a, b = 17.62, 243.12
        gamma = np.log(rh[valid] / 100.0) + (a * t[valid]) / (b + t[valid])
        out[valid] = (b * gamma) / (a - gamma)
    return out


def calculate_heat_index_c(temp_c, humidity):
    """NOAA/Rothfusz heat index, used only in its normal warm/humid domain."""
    t_c = np.asarray(temp_c, dtype=float)
    rh = np.asarray(humidity, dtype=float)
    t_f = t_c * 9.0 / 5.0 + 32.0
    result_f = t_f.copy()
    valid = np.isfinite(t_f) & np.isfinite(rh) & (t_f >= 80.0) & (rh >= 40.0)
    if np.any(valid):
        T = t_f[valid]
        H = rh[valid]
        hi = (
            -42.379
            + 2.04901523 * T
            + 10.14333127 * H
            - 0.22475541 * T * H
            - 0.00683783 * T**2
            - 0.05481717 * H**2
            + 0.00122874 * T**2 * H
            + 0.00085282 * T * H**2
            - 0.00000199 * T**2 * H**2
        )
        result_f[valid] = hi
    return (result_f - 32.0) * 5.0 / 9.0


def calculate_specific_humidity_gkg(temp_c, humidity, pressure_hpa):
    """Specific humidity in g/kg. Requires measured ambient pressure."""
    t = np.asarray(temp_c, dtype=float)
    rh = np.asarray(humidity, dtype=float)
    p = np.asarray(pressure_hpa, dtype=float)
    out = np.full(np.broadcast(t, rh, p).shape, np.nan, dtype=float)
    valid = (
        np.isfinite(t)
        & np.isfinite(rh)
        & np.isfinite(p)
        & (rh > 0)
        & (rh <= 100)
        & (p > 0)
    )
    if np.any(valid):
        es = 6.112 * np.exp((17.62 * t[valid]) / (243.12 + t[valid]))
        e = (rh[valid] / 100.0) * es
        denom = p[valid] - 0.378 * e
        good = denom > 0
        vals = np.full_like(e, np.nan)
        vals[good] = 1000.0 * 0.622 * e[good] / denom[good]
        out[valid] = vals
    return out


def derive_metrics(data: pd.DataFrame, flags: dict) -> pd.DataFrame:
    data = data.copy()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    smooth_columns = [
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
    ]
    for col in smooth_columns:
        data[f"{col}_Smoothed"] = _time_rolling_mean(
            data,
            col,
            HUMIDITY_SMOOTH_WINDOW if "Humidity" in col else SMOOTH_WINDOW,
        )

    camera_age = pd.to_numeric(data["Camera_Lux_Age_s"], errors="coerce")
    data["Camera_Lux_Recent"] = data["Camera_Lux"].where(
        camera_age.notna() & (camera_age <= MAX_CAMERA_LUX_AGE_S)
    )
    data["Camera_Lux_Recent_Smoothed"] = _time_rolling_mean(
        data.assign(Camera_Lux_Recent=data["Camera_Lux_Recent"]),
        "Camera_Lux_Recent",
        SMOOTH_WINDOW,
    )

    data["Median_Temperature_C"] = data["Ambient_Temperature_C"]
    data["Median_Temperature_F"] = data["Ambient_Temperature_C"] * 9.0 / 5.0 + 32.0

    # Derived quantities always follow the station-level Ambient_* policy. If
    # Combined_* is promoted on the Pi later, no server-side redesign is needed.
    data["Dew_Point_C"] = calculate_dew_point(
        data["Ambient_Temperature_C"].to_numpy(),
        data["Ambient_Humidity_percent"].to_numpy(),
    )
    data["Heat_Index_C"] = calculate_heat_index_c(
        data["Ambient_Temperature_C"].to_numpy(),
        data["Ambient_Humidity_percent"].to_numpy(),
    )
    data["Specific_Humidity_gkg"] = calculate_specific_humidity_gkg(
        data["Ambient_Temperature_C"].to_numpy(),
        data["Ambient_Humidity_percent"].to_numpy(),
        data["BME680_Pressure_hPa"].to_numpy(),
    )

    data["Dew_Point_C_Smoothed"] = _time_rolling_mean(
        data, "Dew_Point_C", SMOOTH_WINDOW
    )
    data["Heat_Index_C_Smoothed"] = _time_rolling_mean(
        data, "Heat_Index_C", SMOOTH_WINDOW
    )
    data["Specific_Humidity_gkg_Smoothed"] = _time_rolling_mean(
        data, "Specific_Humidity_gkg", SMOOTH_WINDOW
    )
    return data


def _dead_sensor_notice(ax, text):
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color="0.35",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="0.95", alpha=0.9),
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _configure_time_axis(ax, span: timedelta):
    if span <= timedelta(hours=2):
        fmt = "%H:%M"
    elif span <= timedelta(days=2):
        fmt = "%m-%d\n%H:%M"
    else:
        fmt = "%m-%d"
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt, tz=LOCAL_TZ))
    ax.tick_params(axis="x", rotation=30)


def _safe_set_ylim(ax, series_list, pad_frac=0.07, minimum_pad=0.2):
    arrays = []
    for s in series_list:
        vals = pd.to_numeric(pd.Series(s), errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            arrays.append(vals)
    if not arrays:
        return
    vals = np.concatenate(arrays)
    low, high = float(np.min(vals)), float(np.max(vals))
    pad = max((high - low) * pad_frac, minimum_pad)
    ax.set_ylim(low - pad, high + pad)


def _prepare_prediction_overlay(predict_data: pd.DataFrame, last_real_timestamp) -> pd.DataFrame:
    if predict_data is None or predict_data.empty:
        return pd.DataFrame(columns=["Timestamp", "Predicted_Temperature"])
    p = predict_data.copy()
    p["Timestamp"] = pd.to_datetime(p["Timestamp"], errors="coerce", utc=True)
    p["Predicted_Temperature"] = pd.to_numeric(
        p["Predicted_Temperature"], errors="coerce"
    )
    p = p.dropna(subset=["Timestamp", "Predicted_Temperature"])
    p = p[p["Timestamp"] > last_real_timestamp]
    return p.sort_values("Timestamp")


def generate_plots(data, predict_data, output_path, title, out_of_date_flag, flags):
    if data.empty:
        logging.warning("No data for plot %s", output_path)
        return

    data = data.copy()
    span = max(
        data["Timestamp"].iloc[-1] - data["Timestamp"].iloc[0],
        pd.Timedelta(minutes=1),
    )
    prediction = _prepare_prediction_overlay(
        predict_data, data["Timestamp"].iloc[-1]
    )

    def has_data(column):
        return column in data.columns and pd.to_numeric(
            data[column], errors="coerce"
        ).notna().any()

    fig, axs = plt.subplots(4, 2, figsize=(15, 15))

    # 1. Temperature: DHT22, BME680, and their candidate combination.
    ax = axs[0, 0]
    plotted = False
    if has_data("DHT22_Temperature_C_Smoothed"):
        ax.plot(
            data["Timestamp"], data["DHT22_Temperature_C_Smoothed"],
            linewidth=1.6, alpha=0.85, label="DHT22",
        )
        plotted = True
    if has_data("BME680_Temperature_C_Smoothed"):
        ax.plot(
            data["Timestamp"], data["BME680_Temperature_C_Smoothed"],
            linewidth=1.6, alpha=0.85, label="BME680",
        )
        plotted = True
    if has_data("Combined_Temperature_C_Smoothed"):
        ax.plot(
            data["Timestamp"], data["Combined_Temperature_C_Smoothed"],
            linewidth=2.6, label="Combined candidate",
        )
        plotted = True

    if not prediction.empty and has_data("Ambient_Temperature_C"):
        valid_temp = pd.to_numeric(
            data["Ambient_Temperature_C"], errors="coerce"
        ).dropna()
        if not valid_temp.empty:
            idx = valid_temp.index[-1]
            p = pd.concat(
                [
                    pd.DataFrame({
                        "Timestamp": [data.loc[idx, "Timestamp"]],
                        "Predicted_Temperature": [valid_temp.iloc[-1]],
                    }),
                    prediction,
                ],
                ignore_index=True,
            )
            ax.plot(
                p["Timestamp"], p["Predicted_Temperature"],
                linestyle="--", label="LSTM forecast",
            )
            plotted = True

    if plotted:
        ax.set_ylabel("°C")
        ax.legend(loc="best")
        _safe_set_ylim(
            ax,
            [
                data["DHT22_Temperature_C"],
                data["BME680_Temperature_C"],
                data["Combined_Temperature_C"],
            ],
        )
    else:
        _dead_sensor_notice(ax, "No temperature measurements")
    ax.set_title("Temperature — DHT22, BME680 and Combined")
    ax.grid(alpha=0.3)

    # 2. Relative humidity: DHT22, BME680, and their candidate combination.
    ax = axs[0, 1]
    plotted = False
    if has_data("DHT22_Humidity_percent_Smoothed"):
        ax.plot(
            data["Timestamp"], data["DHT22_Humidity_percent_Smoothed"],
            linewidth=1.6, alpha=0.85, label="DHT22",
        )
        plotted = True
    if has_data("BME680_Humidity_percent_Smoothed"):
        ax.plot(
            data["Timestamp"], data["BME680_Humidity_percent_Smoothed"],
            linewidth=1.6, alpha=0.85, label="BME680",
        )
        plotted = True
    if has_data("Combined_Humidity_percent_Smoothed"):
        ax.plot(
            data["Timestamp"], data["Combined_Humidity_percent_Smoothed"],
            linewidth=2.6, label="Combined candidate",
        )
        plotted = True
    if plotted:
        ax.set_ylim(0, 100)
        ax.set_ylabel("% RH")
        ax.legend(loc="best")
    else:
        _dead_sensor_notice(ax, "No humidity measurements")
    ax.set_title("Relative Humidity — DHT22, BME680 and Combined")
    ax.grid(alpha=0.3)

    # 3. BME680 pressure.
    ax = axs[1, 0]
    if has_data("BME680_Pressure_hPa_Smoothed"):
        ax.plot(
            data["Timestamp"], data["BME680_Pressure_hPa"],
            alpha=0.20, label="Raw",
        )
        ax.plot(
            data["Timestamp"], data["BME680_Pressure_hPa_Smoothed"],
            linewidth=2, label="BME680",
        )
        ax.set_ylabel("hPa")
        ax.legend(loc="best")
        _safe_set_ylim(ax, [data["BME680_Pressure_hPa"]], minimum_pad=0.5)
    else:
        _dead_sensor_notice(ax, "BME680 pressure unavailable")
    ax.set_title("Barometric Pressure")
    ax.grid(alpha=0.3)

    # 4. Camera lux metadata.
    ax = axs[1, 1]
    if data["Camera_Lux_Recent"].notna().any():
        ax.plot(data["Timestamp"], data["Camera_Lux_Recent"], alpha=0.25)
        ax.plot(
            data["Timestamp"], data["Camera_Lux_Recent_Smoothed"],
            linewidth=2, label="Camera metadata",
        )
        ax.set_ylabel("Lux (camera metadata)")
        ax.legend(loc="best")
        positive = data["Camera_Lux_Recent"].dropna()
        if len(positive) and (positive > 0).all():
            ax.set_yscale("log")
    else:
        _dead_sensor_notice(ax, "No recent camera lux metadata")
    ax.set_title("Camera-Reported Light Level")
    ax.grid(alpha=0.3)

    # 5. Heat index.
    ax = axs[2, 0]
    if has_data("Heat_Index_C_Smoothed"):
        ax.plot(
            data["Timestamp"], data["Heat_Index_C_Smoothed"],
            label="Heat index / air temperature",
        )
        ax.set_ylabel("°C")
        ax.legend(loc="best")
    else:
        _dead_sensor_notice(ax, "Heat index unavailable")
    ax.set_title("Heat Index")
    ax.grid(alpha=0.3)

    # 6. Dew point.
    ax = axs[2, 1]
    if has_data("Dew_Point_C_Smoothed"):
        ax.plot(
            data["Timestamp"], data["Dew_Point_C_Smoothed"],
            label="Dew point",
        )
        ax.set_ylabel("°C")
        ax.legend(loc="best")
    else:
        _dead_sensor_notice(ax, "Dew point unavailable")
    ax.set_title("Dew Point")
    ax.grid(alpha=0.3)

    # 7. Specific humidity from station Ambient_* + measured BME680 pressure.
    ax = axs[3, 0]
    if has_data("Specific_Humidity_gkg_Smoothed"):
        ax.plot(
            data["Timestamp"], data["Specific_Humidity_gkg_Smoothed"],
            label="Specific humidity",
        )
        ax.set_ylabel("g/kg")
        ax.legend(loc="best")
    else:
        _dead_sensor_notice(ax, "Specific humidity unavailable")
    ax.set_title("Specific Humidity")
    ax.grid(alpha=0.3)

    # 8. Raw BME680 gas-sensor resistance. Do not present this as VOC
    # concentration/AQI without a separate calibration/model.
    ax = axs[3, 1]
    if has_data("BME680_Gas_Resistance_ohm_Smoothed"):
        gas_kohm = pd.to_numeric(
            data["BME680_Gas_Resistance_ohm_Smoothed"], errors="coerce"
        ) / 1000.0
        ax.plot(
            data["Timestamp"], gas_kohm,
            linewidth=2, label="BME680 gas resistance",
        )
        ax.set_ylabel("Resistance (kΩ)")
        ax.legend(loc="best")
        _safe_set_ylim(ax, [gas_kohm], minimum_pad=0.2)
    else:
        _dead_sensor_notice(ax, "BME680 gas measurement unavailable")
    ax.set_title("BME680 Gas Resistance")
    ax.grid(alpha=0.3)

    for ax in axs.flat:
        if ax.has_data():
            _configure_time_axis(ax, span.to_pytimedelta())

    latest_local = data["Timestamp"].iloc[-1].tz_convert(LOCAL_TZ)
    fig.suptitle(
        f"{title}\nLatest: {latest_local:%Y-%m-%d %H:%M:%S %Z}",
        fontsize=14,
    )
    if out_of_date_flag:
        fig.text(
            0.5, 0.5, "DATA STREAM STALE",
            ha="center", va="center", fontsize=48, alpha=0.12, rotation=30,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    tmp = f"{output_path}.tmp.png"
    fig.savefig(tmp, dpi=120)
    plt.close(fig)
    os.replace(tmp, output_path)


def generate_summary_plot(data, output_path, flags):
    if data.empty:
        return
    recent = data[
        data["Timestamp"] >= data["Timestamp"].max() - pd.Timedelta(hours=24)
    ].copy()
    fig, ax_t = plt.subplots(figsize=(10, 6))
    lines, labels = [], []

    if recent["Ambient_Temperature_C_Smoothed"].notna().any():
        line = ax_t.plot(
            recent["Timestamp"], recent["Ambient_Temperature_C_Smoothed"],
            label="Ambient temperature",
        )
        lines += line
        labels += [item.get_label() for item in line]
    ax_t.set_ylabel("Temperature (°C)")
    ax_t.grid(alpha=0.3)

    if recent["Ambient_Humidity_percent_Smoothed"].notna().any():
        ax_h = ax_t.twinx()
        line = ax_h.plot(
            recent["Timestamp"], recent["Ambient_Humidity_percent_Smoothed"],
            label="Ambient humidity", alpha=0.75,
        )
        lines += line
        labels += [item.get_label() for item in line]
        ax_h.set_ylabel("Relative humidity (%)")
        ax_h.set_ylim(0, 100)

    if lines:
        ax_t.legend(lines, labels, loc="best")
    else:
        ax_t.text(
            0.5, 0.5, "No recent temperature/humidity data",
            transform=ax_t.transAxes, ha="center", va="center",
        )

    ax_t.xaxis.set_major_formatter(
        mdates.DateFormatter("%m-%d\n%H:%M", tz=LOCAL_TZ)
    )
    ax_t.tick_params(axis="x", rotation=30)
    ax_t.set_title("Tempestas — latest 24 hours")
    fig.tight_layout()
    tmp = f"{output_path}.tmp.png"
    fig.savefig(tmp, dpi=120)
    plt.close(fig)
    os.replace(tmp, output_path)


def save_last_minute_averages(data, predict_data, output_file, flags):
    if data.empty:
        return
    latest = data["Timestamp"].max()
    recent = data[data["Timestamp"] >= latest - pd.Timedelta(minutes=1)]
    if recent.empty:
        recent = data.tail(1)

    temp_c = pd.to_numeric(
        recent["Ambient_Temperature_C"], errors="coerce"
    ).mean()
    temp_f = temp_c * 9 / 5 + 32 if np.isfinite(temp_c) else np.nan
    humidity = pd.to_numeric(
        recent["Ambient_Humidity_percent"], errors="coerce"
    ).mean()
    pressure = pd.to_numeric(
        recent["BME680_Pressure_hPa"], errors="coerce"
    ).mean()
    gas_ohm = pd.to_numeric(
        recent["BME680_Gas_Resistance_ohm"], errors="coerce"
    ).mean()
    lux = (
        recent["Camera_Lux_Recent"].mean()
        if recent["Camera_Lux_Recent"].notna().any()
        else np.nan
    )

    forecast = _prepare_prediction_overlay(predict_data, latest)
    pred = forecast["Predicted_Temperature"].iloc[0] if not forecast.empty else np.nan
    local_time = latest.tz_convert(LOCAL_TZ)

    rows = [
        ("Updated", local_time.strftime("%Y-%m-%d %H:%M:%S %Z")),
        (
            "Temperature",
            f"{temp_c:.2f} °C / {temp_f:.2f} °F"
            if np.isfinite(temp_c) else "Unavailable",
        ),
        (
            "Humidity",
            f"{humidity:.1f} %" if np.isfinite(humidity) else "Unavailable",
        ),
        (
            "Pressure",
            f"{pressure:.2f} hPa" if np.isfinite(pressure) else "Unavailable",
        ),
        (
            "BME680 gas resistance",
            f"{gas_ohm / 1000.0:.2f} kΩ"
            if np.isfinite(gas_ohm) else "Unavailable",
        ),
        (
            "Camera light",
            f"{lux:.1f} lx" if np.isfinite(lux) else "Unavailable",
        ),
        (
            "Next forecast",
            f"{pred:.2f} °C" if np.isfinite(pred) else "Not yet available",
        ),
    ]
    table_rows = "\n".join(
        f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows
    )
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Weather Summary</title>
<style>
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid black; padding: 8px; text-align: center; }}
</style>
</head>
<body>
<table>
{table_rows}
</table>
</body>
</html>
"""
    _safe_atomic_text_write(output_file, html)


def save_latest_copy(image_dir, output_name="latest.jpg"):
    image_files = [
        p for p in glob.glob(os.path.join(image_dir, "*.jpg"))
        if os.path.basename(p) != output_name
    ]
    if not image_files:
        return False
    latest_file = max(image_files, key=os.path.getmtime)
    tmp = os.path.join(image_dir, f".{output_name}.tmp")
    shutil.copy2(latest_file, tmp)
    os.replace(tmp, os.path.join(image_dir, output_name))
    return True


def _parse_image_timestamp(path):
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        # Pi currently names images using its UTC system clock.
        return pd.Timestamp(datetime.strptime(name, "%Y%m%d_%H%M%S"), tz="UTC")
    except ValueError:
        return None


def _font(size):
    for candidate in ["DejaVuSans.ttf", "arial.ttf"]:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _generate_gif_with_plot(image_dir, output_gif, data, lookback, frame_duration_ms):
    now = pd.Timestamp.now(tz="UTC")
    start = now - pd.Timedelta(lookback)
    images = []
    for path in glob.glob(os.path.join(image_dir, "*.jpg")):
        ts = _parse_image_timestamp(path)
        if ts is not None and start <= ts <= now:
            images.append((path, ts))
    images.sort(key=lambda item: item[1])
    if len(images) < 2:
        logging.info("Not enough recent images for %s", output_gif)
        return

    frames = []
    data_window = data[(data["Timestamp"] >= start) & (data["Timestamp"] <= now)]
    if len(data_window) > 1000:
        data_window = data_window.iloc[:: max(1, len(data_window) // 1000)]

    for img_path, ts in images:
        try:
            with Image.open(img_path) as source:
                img = source.convert("RGB").copy()
        except Exception as exc:
            logging.warning("Skipping unreadable image %s: %s", img_path, exc)
            continue

        local_ts = ts.tz_convert(LOCAL_TZ)
        draw = ImageDraw.Draw(img)
        text = local_ts.strftime("%Y-%m-%d %H:%M:%S %Z")
        draw.rectangle((5, 5, min(img.width - 5, 520), 45), fill=(0, 0, 0))
        draw.text((12, 10), text, font=_font(24), fill=(255, 255, 255))

        subset = data_window[data_window["Timestamp"] <= ts]
        fig, ax = plt.subplots(figsize=(8, 3))
        plotted = False
        if (
            not subset.empty
            and subset["Ambient_Temperature_C_Smoothed"].notna().any()
        ):
            ax.plot(
                subset["Timestamp"], subset["Ambient_Temperature_C_Smoothed"],
                label="Temperature",
            )
            ax.set_ylabel("°C")
            plotted = True

        if (
            not subset.empty
            and subset["Ambient_Humidity_percent_Smoothed"].notna().any()
        ):
            ax2 = ax.twinx()
            ax2.plot(
                subset["Timestamp"], subset["Ambient_Humidity_percent_Smoothed"],
                alpha=0.6, label="RH",
            )
            ax2.set_ylabel("% RH")
            ax2.set_ylim(0, 100)
            plotted = True

        if not plotted:
            ax.text(
                0.5, 0.5, "No weather data",
                transform=ax.transAxes, ha="center", va="center",
            )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=LOCAL_TZ))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        plt.close(fig)
        buf.seek(0)
        with Image.open(buf) as pimg:
            plot_img = pimg.convert("RGB").copy()

        new_h = int(plot_img.height * img.width / plot_img.width)
        plot_img = plot_img.resize((img.width, new_h), Image.Resampling.LANCZOS)
        combined = Image.new("RGB", (img.width, img.height + new_h))
        combined.paste(img, (0, 0))
        combined.paste(plot_img, (0, img.height))
        frames.append(combined)

    if len(frames) < 2:
        return
    tmp = f"{output_gif}.tmp.gif"
    frames[0].save(
        tmp,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
    )
    os.replace(tmp, output_gif)


def generate_hourly_gif_with_plot(image_dir, output_gif, data, sensor_flags=None):
    _generate_gif_with_plot(image_dir, output_gif, data, "1h", 150)


def generate_daily_gif_with_plot(image_dir, output_gif, data, sensor_flags=None):
    _generate_gif_with_plot(image_dir, output_gif, data, "24h", 250)


def gather_system_stats(output_file=None):
    output_file = output_file or SERVER_STATS_FILE
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_pct = psutil.virtual_memory().percent
    try:
        temps = psutil.sensors_temperatures()
        cpu_temp = temps.get("coretemp", [])[0].current if temps.get("coretemp") else np.nan
    except Exception:
        cpu_temp = np.nan
    write_header = not os.path.exists(output_file) or os.path.getsize(output_file) == 0
    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Timestamp", "CPU_Usage_pct", "Memory_Usage_pct", "CPU_Temp_C"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            round(cpu_usage, 1),
            round(memory_pct, 1),
            round(cpu_temp, 1) if np.isfinite(cpu_temp) else "",
        ])


def plot_system_stats(csv_file=None, output_image=None):
    csv_file = csv_file or SERVER_STATS_FILE
    output_image = output_image or os.path.join(BASE_DIR, "system_stats_plot.png")
    if not os.path.exists(csv_file):
        return
    df = pd.read_csv(csv_file)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").tail(2000)
    if df.empty:
        return
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df["Timestamp"], pd.to_numeric(df["CPU_Usage_pct"], errors="coerce"), label="CPU usage")
    ax1.plot(df["Timestamp"], pd.to_numeric(df["Memory_Usage_pct"], errors="coerce"), label="Memory usage")
    ax1.set_ylabel("Usage (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper left")
    if "CPU_Temp_C" in df.columns:
        ax2 = ax1.twinx()
        ax2.plot(df["Timestamp"], pd.to_numeric(df["CPU_Temp_C"], errors="coerce"), alpha=0.7, label="CPU temp")
        ax2.set_ylabel("CPU temperature (°C)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=LOCAL_TZ))
    ax1.tick_params(axis="x", rotation=30)
    ax1.set_title("Cirrus system statistics")
    fig.tight_layout()
    tmp = f"{output_image}.tmp.png"
    fig.savefig(tmp, dpi=120)
    plt.close(fig)
    os.replace(tmp, output_image)


def plot_pi_system_stats(csv_file=None, output_image=None):
    csv_file = csv_file or PI_SYSTEM_FILE
    output_image = output_image or os.path.join(BASE_DIR, "pi_system_stats_plot.png")
    if not os.path.exists(csv_file):
        return
    df = pd.read_csv(csv_file)
    required = {"Timestamp", "CPU_Temperature_C", "CPU_Usage_percent", "Memory_Usage_percent"}
    if not required.issubset(df.columns):
        return
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").tail(4000)
    if df.empty:
        return
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df["Timestamp"], pd.to_numeric(df["CPU_Usage_percent"], errors="coerce"), label="CPU usage")
    ax1.plot(df["Timestamp"], pd.to_numeric(df["Memory_Usage_percent"], errors="coerce"), label="Memory usage")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Usage (%)")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(df["Timestamp"], pd.to_numeric(df["CPU_Temperature_C"], errors="coerce"), label="Pi CPU temp", alpha=0.75)
    ax2.set_ylabel("CPU temperature (°C)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=LOCAL_TZ))
    ax1.tick_params(axis="x", rotation=30)
    ax1.set_title("Raspberry Pi system statistics")
    fig.tight_layout()
    tmp = f"{output_image}.tmp.png"
    fig.savefig(tmp, dpi=120)
    plt.close(fig)
    os.replace(tmp, output_image)


def prepare_forecast(master_file: str) -> pd.DataFrame:
    """Return a current forecast or an empty frame when insufficient data exist."""
    empty = pd.DataFrame(columns=["Timestamp", "Predicted_Temperature"])
    try:
        forecaster = WeatherForecaster(
            master_file=master_file,
            hidden_dim=96,
            num_layers=2,
            batch_size=128,
            seq_length=FORECAST_SEQUENCE_MINUTES,
            feature_cols=["Ambient_Temperature_C", "Ambient_Humidity_percent"],
            target_col="Ambient_Temperature_C",
        )
        recent = forecaster.load_master_data()
    except Exception as exc:
        logging.info("Forecast unavailable: %s", exc)
        return empty

    if len(recent) < FORECAST_MIN_RESAMPLED_ROWS:
        logging.info(
            "Forecast disabled until %d valid one-minute rows exist (currently %d)",
            FORECAST_MIN_RESAMPLED_ROWS,
            len(recent),
        )
        return empty

    now_local = datetime.now(LOCAL_TZ)
    need_train = not os.path.exists(MODEL_FILE)
    if os.path.exists(MODEL_FILE):
        model_age = datetime.now(timezone.utc) - datetime.fromtimestamp(
            os.path.getmtime(MODEL_FILE), tz=timezone.utc
        )
        if model_age > timedelta(days=1) and now_local.hour == 4:
            need_train = True

    if not need_train:
        try:
            forecaster.load_model(MODEL_FILE)
        except Exception as exc:
            logging.warning("Existing v2 forecast model unusable; retraining: %s", exc)
            need_train = True

    if need_train:
        try:
            forecaster.train_model(epochs=30)
            forecaster.save_model(MODEL_FILE)
            forecaster.plot_training_loss(
                "training_loss.csv", os.path.join(BASE_DIR, "training_loss_plot.png")
            )
        except Exception as exc:
            logging.warning("Forecast training failed: %s", exc)
            return empty

    try:
        recent = forecaster.load_master_data()
        seq = recent[forecaster.feature_cols].to_numpy(dtype=np.float32)[-forecaster.seq_length :]
        predictions = forecaster.predict_future(seq, steps_ahead=FORECAST_STEPS)
        timestamps = pd.to_datetime(recent["Timestamp"], utc=True)
        interval = (timestamps.iloc[-1] - timestamps.iloc[-2]).total_seconds()
        future = forecaster.infer_timestamps(timestamps.iloc[-1], FORECAST_STEPS, interval)
        forecaster.save_predictions_to_csv(predictions, future, PREDICT_FILE)
        return pd.DataFrame({"Timestamp": future, "Predicted_Temperature": predictions})
    except Exception as exc:
        logging.warning("Forecast inference failed: %s", exc)
        return empty


def run_bird_detection():
    if bird_detection is None:
        logging.warning("Bird detection unavailable: %s", _BIRD_IMPORT_ERROR)
        return
    try:
        bird_detection.run_detection_pipeline(
            image_dir=IMAGE_DIR,
            output_dir=os.path.join(IMAGE_DIR, "birds"),
            confidence_threshold=0.35,
            log_file=os.path.join(IMAGE_DIR, "birds", "processed_images.json"),
            hours_back=2,
            target_classes=[
                "bird", "squirrel", "cat", "rabbit", "fox", "deer", "raccoon",
                "skunk", "coyote", "mouse", "vole", "chipmunk", "prairie dog",
                "badger", "weasel", "hawk", "owl", "magpie", "crow", "raven",
                "turkey", "woodpecker",
            ],
        )
    except Exception as exc:
        logging.warning("Bird detection failed: %s", exc)


def main():
    logging.info("Starting server weather processing")
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    try:
        gather_system_stats()
    except Exception as exc:
        logging.warning("gather_system_stats failed: %s", exc)

    try:
        master_data = load_master_data(MASTER_FILE)
    except Exception as exc:
        logging.error("Cannot load weather master: %s", exc)
        return 1

    if master_data.empty:
        logging.warning("Weather master contains no rows; nothing to plot")
        return 0

    flags = check_sensor_validity(master_data)

    # Derivation is row-wise/NaN-safe. A failed DHT22 must not stop BME680 gas
    # or pressure products, and a failed BME680 must not stop DHT22 products.
    master_data = derive_metrics(master_data, flags)
    generate_json_from_data(master_data[WEATHER_COLUMNS], MASTER_FILE_JSON)

    if flags["ambient_temp"] and flags["ambient_humidity"]:
        predict_data = prepare_forecast(MASTER_FILE)
    else:
        logging.info(
            "Forecast disabled while current Ambient temperature/humidity is unavailable"
        )
        predict_data = pd.DataFrame(
            columns=["Timestamp", "Predicted_Temperature"]
        )

    now_utc = pd.Timestamp.now(tz="UTC")
    max_timestamp = master_data["Timestamp"].max()
    lag = now_utc - max_timestamp
    stale = lag > pd.Timedelta(hours=1)
    if stale:
        logging.warning("Weather stream is stale by %s", lag)

    try:
        save_latest_copy(IMAGE_DIR)
    except Exception as exc:
        logging.warning("save_latest_copy failed: %s", exc)

    # Always regenerate every panel from whatever measurements are genuinely
    # available. One sensor outage is a degraded channel, not a processor crash.
    for label, delta in TIME_SPANS.items():
        subset = master_data[
            master_data["Timestamp"] >= max_timestamp - pd.Timedelta(delta)
        ].copy()
        generate_plots(
            subset,
            predict_data,
            os.path.join(BASE_DIR, f"weather_plot_{label}.png"),
            f"Weather Data ({label.replace('_', ' ').title()})",
            int(stale),
            flags,
        )

    generate_summary_plot(
        master_data, os.path.join(BASE_DIR, "summary_plot.png"), flags
    )
    save_last_minute_averages(
        master_data,
        predict_data,
        os.path.join(BASE_DIR, "small_summary.html"),
        flags,
    )

    try:
        generate_hourly_gif_with_plot(IMAGE_DIR, HOURLY_GIF, master_data, flags)
        generate_daily_gif_with_plot(IMAGE_DIR, DAILY_GIF, master_data, flags)
    except Exception as exc:
        logging.warning("GIF generation failed: %s", exc)

    try:
        plot_system_stats()
        plot_pi_system_stats()
    except Exception as exc:
        logging.warning("System-stat plotting failed: %s", exc)

    run_bird_detection()
    logging.info("Weather processing complete")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
