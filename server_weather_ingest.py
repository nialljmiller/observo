#!/usr/bin/env python3
"""Robust ingestion for Tempestas/Observo weather-station data v2.

This is the clean BME680 + DHT22 ingestion path. It intentionally does not
migrate or preserve any earlier weather schema.

Design goals
------------
* Never truncate or delete an incoming upload before it is validated.
* Require the exact v2 schema before touching the master data.
* Tolerate the Pi re-uploading the same local history by de-duplicating on UTC
  timestamp.
* Retain every DHT22 and BME680 raw environmental measurement.
* Ingest Pi health telemetry (including ``Pi_Throttled_Hex``) separately.
* Use an inter-process lock and atomic master-file replacement so readers never
  see a partially rewritten master CSV.
"""

from __future__ import annotations

import fcntl
import io
import logging
import os
import re
import tempfile
import time
from pathlib import Path

import pandas as pd

from weather_schema import (
    PLAUSIBILITY_RANGES,
    SYSTEM_COLUMNS,
    SYSTEM_NUMERIC_COLUMNS,
    WEATHER_COLUMNS,
    WEATHER_NUMERIC_COLUMNS,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BASE_DIR = Path("/media/bigdata/weather_station")

WEATHER_INCOMING_FILE = BASE_DIR / "weather_data.csv"
SYSTEM_INCOMING_FILE = BASE_DIR / "system_usage.csv"

WEATHER_MASTER_FILE = BASE_DIR / "all_data.csv"
SYSTEM_MASTER_FILE = BASE_DIR / "all_system_data.csv"

LOCK_FILE = BASE_DIR / ".server_weather_ingest.lock"

# SCP writes a destination file progressively.  A very recent file may still
# be in flight, so leave it for the next watchdog cycle rather than risk reading
# a partial upload.
MIN_INPUT_AGE_S = 2.0

# Avoid silently accepting a truncated file that contains only a fragment of a
# row but happens to parse.  Header-only files are valid and simply contain no
# measurements.
REJECT_UNEXPECTED_COLUMNS = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOG = logging.getLogger("weather_ingest")


class InputNotReady(RuntimeError):
    """The incoming file may still be changing; retry on the next cycle."""


class InputValidationError(RuntimeError):
    """The incoming file is stable but not valid for the canonical schema."""


# -----------------------------------------------------------------------------
# Filesystem safety
# -----------------------------------------------------------------------------

def atomic_write_csv(df: pd.DataFrame, destination: Path) -> None:
    """Write ``df`` beside ``destination`` then atomically replace it."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temp_path = Path(temp_name)

    try:
        with os.fdopen(fd, "w", newline="") as handle:
            df.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass


def read_stable_bytes(path: Path) -> bytes:
    """Return a consistent snapshot of ``path`` or defer if SCP may be writing."""
    if not path.exists():
        raise FileNotFoundError(path)

    before = path.stat()
    age = time.time() - before.st_mtime
    if age < MIN_INPUT_AGE_S:
        raise InputNotReady(
            f"{path.name} is only {age:.2f}s old; upload may still be in progress"
        )

    with path.open("rb") as handle:
        # This lock protects against cooperating local writers.  SCP itself does
        # not promise to honor flock, so the before/after stat check is still
        # required below.
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        payload = handle.read()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)

    if identity_before != identity_after:
        raise InputNotReady(f"{path.name} changed while it was being read")

    if len(payload) != before.st_size:
        raise InputNotReady(
            f"{path.name} read length {len(payload)} != stat size {before.st_size}"
        )

    return payload


# -----------------------------------------------------------------------------
# Schema and value validation
# -----------------------------------------------------------------------------

def _canonical_utc_timestamp(series: pd.Series, source_name: str) -> pd.Series:
    text = series.astype("string").str.strip()
    timezone_aware = text.str.contains(
        r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True, na=False
    )
    if not timezone_aware.all():
        examples = text[~timezone_aware].head(5).tolist()
        raise InputValidationError(
            f"{source_name}: timezone-less Timestamp value(s) are not accepted; "
            f"examples={examples}. Pi measurements must include Z or an explicit UTC offset."
        )

    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    bad = parsed.isna()
    if bad.any():
        examples = series[bad].astype(str).head(5).tolist()
        raise InputValidationError(
            f"{source_name}: {int(bad.sum())} invalid Timestamp value(s); "
            f"examples={examples}"
        )

    # Fixed UTC text avoids mixed tz-aware/naive pandas behavior downstream and
    # gives deterministic de-duplication keys.
    return parsed.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _coerce_numeric_strict(
    frame: pd.DataFrame,
    columns: list[str],
    source_name: str,
) -> None:
    for column in columns:
        raw = frame[column]
        converted = pd.to_numeric(raw, errors="coerce")

        # Blank cells/NaN are legitimate for optional sensors.  Non-blank text
        # that cannot be converted is corruption and rejects the batch.
        invalid = raw.notna() & converted.isna()
        if invalid.any():
            examples = raw[invalid].astype(str).head(5).tolist()
            raise InputValidationError(
                f"{source_name}: column {column!r} has "
                f"{int(invalid.sum())} non-numeric value(s); examples={examples}"
            )

        frame[column] = converted


def _warn_plausibility(frame: pd.DataFrame, source_name: str) -> None:
    for column, limits in PLAUSIBILITY_RANGES.items():
        if column not in frame.columns:
            continue
        low, high = limits
        values = pd.to_numeric(frame[column], errors="coerce")
        bad = values.notna() & ((values < low) | (values > high))
        if bad.any():
            examples = values[bad].head(5).tolist()
            LOG.warning(
                "%s: %d value(s) in %s outside diagnostic range [%s, %s]; "
                "ingesting them unchanged; examples=%s",
                source_name,
                int(bad.sum()),
                column,
                low,
                high,
                examples,
            )


def parse_canonical_csv(
    payload: bytes,
    *,
    expected_columns: list[str],
    numeric_columns: list[str],
    source_name: str,
) -> pd.DataFrame:
    if not payload.strip():
        raise InputValidationError(f"{source_name}: file is empty")

    try:
        frame = pd.read_csv(
            io.BytesIO(payload),
            encoding="utf-8",
            on_bad_lines="error",
        )
    except Exception as exc:
        raise InputValidationError(f"{source_name}: CSV parse failed: {exc}") from exc

    actual = list(frame.columns)
    missing = [column for column in expected_columns if column not in actual]
    unexpected = [column for column in actual if column not in expected_columns]

    if missing:
        raise InputValidationError(
            f"{source_name}: missing required column(s): {missing}; got {actual}"
        )
    if unexpected and REJECT_UNEXPECTED_COLUMNS:
        raise InputValidationError(
            f"{source_name}: unexpected column(s): {unexpected}; expected "
            f"{expected_columns}"
        )

    # Canonical order, regardless of source ordering.
    frame = frame[expected_columns].copy()

    if frame.empty:
        return frame

    frame["Timestamp"] = _canonical_utc_timestamp(frame["Timestamp"], source_name)
    _coerce_numeric_strict(frame, numeric_columns, source_name)

    if "Pi_Throttled_Hex" in frame.columns:
        values = frame["Pi_Throttled_Hex"].astype("string")
        bad = values.notna() & ~values.str.fullmatch(r"0x[0-9a-fA-F]+", na=False)
        if bad.any():
            examples = values[bad].head(5).tolist()
            raise InputValidationError(
                f"{source_name}: invalid Pi_Throttled_Hex value(s); "
                f"examples={examples}"
            )
        frame["Pi_Throttled_Hex"] = values.str.lower()

    duplicate_count = int(frame.duplicated(subset=["Timestamp"], keep="last").sum())
    if duplicate_count:
        LOG.warning(
            "%s: %d duplicate timestamp row(s) inside upload; keeping last",
            source_name,
            duplicate_count,
        )
        frame = frame.drop_duplicates(subset=["Timestamp"], keep="last")

    frame = frame.sort_values("Timestamp").reset_index(drop=True)
    _warn_plausibility(frame, source_name)
    return frame


# -----------------------------------------------------------------------------
# Master-data handling
# -----------------------------------------------------------------------------

def ensure_clean_master(path: Path, expected_columns: list[str]) -> None:
    """Create a canonical master, replacing any old-schema master in place."""
    if not path.exists():
        LOG.info("Creating new master file: %s", path)
        atomic_write_csv(pd.DataFrame(columns=expected_columns), path)
        return

    try:
        existing_header = list(pd.read_csv(path, nrows=0).columns)
    except Exception as exc:
        LOG.warning("Could not read %s header (%s); starting clean", path, exc)
        atomic_write_csv(pd.DataFrame(columns=expected_columns), path)
        return

    if existing_header != expected_columns:
        LOG.warning(
            "Replacing old/incompatible master %s. Existing schema=%s; new schema=%s",
            path,
            existing_header,
            expected_columns,
        )
        atomic_write_csv(pd.DataFrame(columns=expected_columns), path)


def load_master(path: Path, expected_columns: list[str]) -> pd.DataFrame:
    ensure_clean_master(path, expected_columns)
    try:
        frame = pd.read_csv(path, on_bad_lines="error")
    except Exception as exc:
        # A canonical master becoming unreadable is not something to paper over:
        # preserving the incoming upload and failing loudly is safer than
        # discarding known-good historical rows.
        raise RuntimeError(f"Cannot read canonical master {path}: {exc}") from exc

    if list(frame.columns) != expected_columns:
        raise RuntimeError(f"Canonical master schema changed unexpectedly: {path}")
    return frame


def merge_into_master(
    incoming: pd.DataFrame,
    *,
    master_path: Path,
    expected_columns: list[str],
    source_name: str,
) -> int:
    if incoming.empty:
        LOG.info("%s: upload contains header only; nothing to ingest", source_name)
        return 0

    master = load_master(master_path, expected_columns)

    if not master.empty:
        # Re-normalize timestamp text from disk before comparison.  This also
        # catches manually edited/broken master timestamps loudly.
        master["Timestamp"] = _canonical_utc_timestamp(
            master["Timestamp"], f"{master_path.name} master"
        )

    existing_keys = set(master["Timestamp"]) if not master.empty else set()
    new_rows = incoming[~incoming["Timestamp"].isin(existing_keys)].copy()
    duplicate_count = len(incoming) - len(new_rows)

    if new_rows.empty:
        LOG.info(
            "%s: no new timestamps; %d uploaded row(s) already present",
            source_name,
            duplicate_count,
        )
        return 0

    if master.empty:
        combined = new_rows
    else:
        combined = pd.concat([master, new_rows], ignore_index=True)

    combined = combined.sort_values("Timestamp").reset_index(drop=True)
    atomic_write_csv(combined[expected_columns], master_path)

    LOG.info(
        "%s: master now has %d row(s); appended %d new row(s); "
        "%d uploaded row(s) already present",
        source_name,
        len(combined),
        len(new_rows),
        duplicate_count,
    )
    return len(new_rows)


def ingest_one(
    incoming_path: Path,
    master_path: Path,
    expected_columns: list[str],
    numeric_columns: list[str],
) -> int:
    source_name = incoming_path.name

    try:
        payload = read_stable_bytes(incoming_path)
    except FileNotFoundError:
        LOG.warning("%s does not exist; skipping", incoming_path)
        return 0
    except InputNotReady as exc:
        LOG.info("Deferring %s: %s", source_name, exc)
        return 0

    try:
        frame = parse_canonical_csv(
            payload,
            expected_columns=expected_columns,
            numeric_columns=numeric_columns,
            source_name=source_name,
        )
    except InputValidationError as exc:
        # Critical robustness rule: the uploaded file remains untouched, so it
        # can be inspected/recovered and a later good upload can replace it.
        LOG.error("Rejecting %s without modifying it: %s", source_name, exc)
        return 0

    return merge_into_master(
        frame,
        master_path=master_path,
        expected_columns=expected_columns,
        source_name=source_name,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    # One ingest process at a time.  LOCK_NB prevents watchdog overlap from
    # queueing multiple expensive full-master rewrites.
    with LOCK_FILE.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            LOG.warning("Another server_weather_ingest process is active; exiting cleanly")
            return 0

        # Version-2 clean deployment: incompatible master schemas are intentionally discarded.
        ensure_clean_master(WEATHER_MASTER_FILE, WEATHER_COLUMNS)
        ensure_clean_master(SYSTEM_MASTER_FILE, SYSTEM_COLUMNS)

        weather_new = ingest_one(
            WEATHER_INCOMING_FILE,
            WEATHER_MASTER_FILE,
            WEATHER_COLUMNS,
            WEATHER_NUMERIC_COLUMNS,
        )
        system_new = ingest_one(
            SYSTEM_INCOMING_FILE,
            SYSTEM_MASTER_FILE,
            SYSTEM_COLUMNS,
            SYSTEM_NUMERIC_COLUMNS,
        )

        LOG.info(
            "Ingestion cycle complete: weather_new=%d, system_new=%d",
            weather_new,
            system_new,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
