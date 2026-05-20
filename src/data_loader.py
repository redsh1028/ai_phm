"""
TDMS + Operation CSV data loader.

Discovers train/validation runs by detecting *_Vibration/ directories
and matching *_Operation.csv files.  TDMS files contain only vibration
channels (CH1–CH4); auxiliary channels (Torque, RPM, Temperatures) are
read from the companion CSV.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from nptdms import TdmsFile

from src.config import (
    ACQUISITION_DURATION_SEC,
    ACQUISITION_INTERVAL_SEC,
    AUX_INTERVAL_SEC,
    CSV_ENCODING,
    VIBRATION_CHANNELS,
    PipelineConfig,
)
from src.utils import get_logger, sorted_naturally

logger = get_logger(__name__)


# ── Data containers ────────────────────────────────────────────────────────

@dataclass
class TdmsRecord:
    """Data extracted from a single TDMS file (one acquisition point)."""

    filepath: str
    vibration: Dict[str, np.ndarray] = field(default_factory=dict)  # ch_key → array
    file_index: int = 0  # 0-based index within the run


@dataclass
class OperationData:
    """Parsed Operation CSV for a single bearing run."""

    time_sec: np.ndarray      # shape (N,) – absolute timestamps in seconds
    torque: np.ndarray         # Nm
    rpm: np.ndarray            # RPM
    temp_front: np.ndarray     # ℃
    temp_rear: np.ndarray      # ℃


@dataclass
class BearingRun:
    """All data for a single bearing degradation run."""

    name: str                          # e.g. "Train1"
    records: List[TdmsRecord] = field(default_factory=list)
    operation: Optional[OperationData] = None


# ── Column finder (handles leading spaces & encoding issues) ──────────────

def _find_col(df: pd.DataFrame, *patterns: str) -> Optional[str]:
    """Return the first column name that contains any of the patterns."""
    for col in df.columns:
        col_stripped = col.strip().lower()
        for pat in patterns:
            if pat.lower() in col_stripped:
                return col
    return None


# ── Operation CSV loader ──────────────────────────────────────────────────

def load_operation_csv(csv_path: str) -> OperationData:
    """Parse a Train*_Operation.csv / Validation*_Operation.csv file."""
    # Try multiple encodings
    df = None
    for enc in (CSV_ENCODING, "utf-8", "euc-kr", "latin-1"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if df is None:
        raise RuntimeError(f"Could not decode {csv_path} with any known encoding.")

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Locate columns
    time_col = _find_col(df, "time")
    torque_col = _find_col(df, "torque")
    rpm_col = _find_col(df, "motor speed", "rpm")
    tf_col = _find_col(df, "tc sp front", "front")
    tr_col = _find_col(df, "tc sp rear", "rear")

    def _safe_arr(col_name: Optional[str]) -> np.ndarray:
        if col_name is None or col_name not in df.columns:
            return np.array([], dtype=np.float32)
        return pd.to_numeric(df[col_name], errors="coerce").values.astype(np.float32)

    return OperationData(
        time_sec=_safe_arr(time_col),
        torque=_safe_arr(torque_col),
        rpm=_safe_arr(rpm_col),
        temp_front=_safe_arr(tf_col),
        temp_rear=_safe_arr(tr_col),
    )


def get_operation_window(
    op: OperationData,
    file_index: int,
    interval_sec: int = ACQUISITION_INTERVAL_SEC,
    duration_sec: int = ACQUISITION_DURATION_SEC,
    aux_interval_sec: int = AUX_INTERVAL_SEC,
) -> Dict[str, np.ndarray]:
    """
    Extract the operation-data window corresponding to a specific TDMS file.

    TDMS file *file_index* (0-based) covers the time window:
        [file_index * interval_sec, file_index * interval_sec + duration_sec]

    The operation CSV is sampled at *aux_interval_sec* intervals.

    Returns dict with keys: torque, rpm, temp_front, temp_rear.
    """
    t_start = file_index * interval_sec
    t_end = t_start + duration_sec

    if len(op.time_sec) == 0:
        return {"torque": np.array([]), "rpm": np.array([]),
                "temp_front": np.array([]), "temp_rear": np.array([])}

    mask = (op.time_sec >= t_start) & (op.time_sec <= t_end)

    return {
        "torque": op.torque[mask] if len(op.torque) else np.array([]),
        "rpm": op.rpm[mask] if len(op.rpm) else np.array([]),
        "temp_front": op.temp_front[mask] if len(op.temp_front) else np.array([]),
        "temp_rear": op.temp_rear[mask] if len(op.temp_rear) else np.array([]),
    }


def get_mean_rpm_for_file(
    op: OperationData,
    file_index: int,
    interval_sec: int = ACQUISITION_INTERVAL_SEC,
    duration_sec: int = ACQUISITION_DURATION_SEC,
) -> float:
    """Return mean RPM for the window matching a specific TDMS file, or 1000.0 fallback."""
    window = get_operation_window(op, file_index, interval_sec, duration_sec)
    rpm_arr = window["rpm"]
    if len(rpm_arr) > 0:
        val = float(np.nanmean(rpm_arr))
        if val > 0 and not np.isnan(val):
            return val
    return 1000.0  # fallback


# ── TDMS file loader ──────────────────────────────────────────────────────

def load_tdms_file(filepath: str, file_index: int = 0) -> TdmsRecord:
    """Parse a single TDMS file and return a :class:`TdmsRecord`."""
    tdms = TdmsFile.read(filepath)
    record = TdmsRecord(filepath=filepath, file_index=file_index)

    # Vibration channels
    for ch_key in VIBRATION_CHANNELS:
        found = False
        for group in tdms.groups():
            for ch in group.channels():
                if ch.name.strip().upper() == ch_key.upper():
                    record.vibration[ch_key] = ch.data.astype(np.float32)
                    found = True
                    break
            if found:
                break
        if not found:
            logger.warning("Missing vibration channel %s in %s", ch_key, filepath)

    return record


# ── Run discovery ─────────────────────────────────────────────────────────

def _discover_bearing_runs(parent_dir: str) -> List[Tuple[str, str, Optional[str]]]:
    """
    Discover bearing runs under *parent_dir*.

    Returns list of (run_name, vibration_dir, operation_csv_or_None).
    Looks for *_Vibration/ directories and matching *_Operation.csv files.
    """
    if not os.path.isdir(parent_dir):
        logger.error("Directory does not exist: %s", parent_dir)
        return []

    entries = sorted_naturally(os.listdir(parent_dir))
    vib_dirs: Dict[str, str] = {}
    op_csvs: Dict[str, str] = {}

    for entry in entries:
        full = os.path.join(parent_dir, entry)
        if os.path.isdir(full) and "_vibration" in entry.lower():
            # Extract run name: "Train1_Vibration" → "Train1"
            run_name = re.sub(r"_vibration$", "", entry, flags=re.IGNORECASE)
            vib_dirs[run_name.lower()] = full
        elif os.path.isfile(full) and "_operation" in entry.lower() and entry.lower().endswith(".csv"):
            run_name = re.sub(r"_operation\.csv$", "", entry, flags=re.IGNORECASE)
            op_csvs[run_name.lower()] = full

    results: List[Tuple[str, str, Optional[str]]] = []
    for key, vdir in sorted(vib_dirs.items()):
        csv_path = op_csvs.get(key)
        display_name = os.path.basename(vdir).replace("_Vibration", "").replace("_vibration", "")
        results.append((display_name, vdir, csv_path))

    return results


# ── Run-level loader ──────────────────────────────────────────────────────

def load_bearing_run(
    run_name: str,
    vibration_dir: str,
    operation_csv: Optional[str] = None,
) -> BearingRun:
    """Load all TDMS files and operation CSV for a single bearing run."""
    tdms_files = [f for f in os.listdir(vibration_dir) if f.lower().endswith(".tdms")]
    if not tdms_files:
        raise FileNotFoundError(f"No TDMS files found in {vibration_dir}")

    tdms_files = sorted_naturally(tdms_files)
    logger.info("Loading run '%s' – %d TDMS files from %s", run_name, len(tdms_files), vibration_dir)

    records: List[TdmsRecord] = []
    for idx, fname in enumerate(tdms_files):
        fpath = os.path.join(vibration_dir, fname)
        records.append(load_tdms_file(fpath, file_index=idx))

    # Load operation CSV
    op_data = None
    if operation_csv and os.path.isfile(operation_csv):
        logger.info("  Loading operation data from %s", operation_csv)
        op_data = load_operation_csv(operation_csv)
    else:
        logger.warning("  No operation CSV found for run '%s'.", run_name)

    return BearingRun(name=run_name, records=records, operation=op_data)


# ── Dataset-level loader ──────────────────────────────────────────────────

def load_all_runs(parent_dir: str) -> List[BearingRun]:
    """Load all bearing runs under *parent_dir*."""
    discovered = _discover_bearing_runs(parent_dir)
    if not discovered:
        logger.warning("No bearing runs found under %s", parent_dir)
        return []

    runs: List[BearingRun] = []
    for run_name, vib_dir, op_csv in discovered:
        try:
            runs.append(load_bearing_run(run_name, vib_dir, op_csv))
        except FileNotFoundError as exc:
            logger.warning("Skipping %s: %s", run_name, exc)

    logger.info("Loaded %d runs: %s", len(runs), [r.name for r in runs])
    return runs
