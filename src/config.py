"""
Configuration module for the KSPHM-KIMM 2026 PHM Data Challenge.

Centralises every tunable parameter so that nothing is hard-coded
elsewhere in the pipeline.

Actual data layout (discovered from the provided dataset)
─────────────────────────────────────────────────────────
  <data_dir>/
    train/
      Train1_Operation.csv          # aux channels (0.1 Hz = 10-s intervals)
      Train1_Vibration/
        000001.tdms … 000126.tdms   # CH1–CH4 vibration only
      Train2_Operation.csv
      Train2_Vibration/
      Train3_Operation.csv
      Train3_Vibration/
      Train4_Operation.csv
      Train4_Vibration/
    validation/                      # same sub-structure
      ...
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ── Vibration channel mapping (inside TDMS) ───────────────────────────────
VIBRATION_CHANNELS: List[str] = ["CH1", "CH2", "CH3", "CH4"]

VIBRATION_CHANNEL_DESCRIPTIONS: Dict[str, str] = {
    "CH1": "Front Vertical Vibration",
    "CH2": "Front Axial Vibration",
    "CH3": "Rear Vertical Vibration",
    "CH4": "Rear Axial Vibration",
}

# ── Operation CSV column mapping ──────────────────────────────────────────
# Column names as they appear in the CSVs (with leading spaces)
OP_COL_TIME = "Time[sec]"
OP_COL_TORQUE = "Torque[Nm]"         # may have leading space
OP_COL_RPM = "Motor speed[rpm]"       # may have leading space
OP_COL_TEMP_FRONT = "TC SP Front"     # partial match (encoding issues with ℃)
OP_COL_TEMP_REAR = "TC SP Rear"       # partial match

# Sanitised feature-name prefixes for operation columns
OP_FEATURE_NAMES: Dict[str, str] = {
    "torque": "Torque_Nm",
    "rpm": "Motor_speed_rpm",
    "temp_front": "TC_SP_Front_C",
    "temp_rear": "TC_SP_Rear_C",
}

# ── Sampling parameters ───────────────────────────────────────────────────
VIBRATION_SAMPLE_RATE: float = 25_600.0          # Hz
AUX_SAMPLE_RATE: float = 0.1                      # Hz  (one sample every 10 s)
AUX_INTERVAL_SEC: int = 10                         # seconds between operation CSV rows
ACQUISITION_INTERVAL_SEC: int = 600                # seconds between acquisitions
ACQUISITION_DURATION_SEC: int = 60                 # 1-minute recording window

# ── Bearing fault frequencies at 1 000 RPM (reference) ─────────────────────
FAULT_FREQS_AT_1000RPM: Dict[str, float] = {
    "BPFI": 140.0,
    "BPFO": 93.0,
    "BSF":  78.0,
    "Cage": 6.7,
}

# ── Fault-frequency harmonics & bandwidth ──────────────────────────────────
FAULT_HARMONICS: List[int] = [1, 2, 3]           # 1× = fundamental
FAULT_BAND_HZ: float = 5.0                        # ±5 Hz around each frequency

# ── Label generation ──────────────────────────────────────────────────────
DEFAULT_FAILURE_OFFSET_SEC: int = 300              # seconds after last file

# ── Partial-run validation percentages ─────────────────────────────────────
PARTIAL_RUN_CUTOFFS: List[float] = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

# ── Calibration search grid ───────────────────────────────────────────────
CALIBRATION_RANGE: Tuple[float, float, float] = (0.70, 1.31, 0.01)  # start, stop, step

# ── Random seed ───────────────────────────────────────────────────────────
RANDOM_SEED: int = 42

# ── CSV encoding (Korean Windows) ─────────────────────────────────────────
CSV_ENCODING: str = "cp949"


@dataclass
class PipelineConfig:
    """Run-level configuration, overridable via CLI flags."""

    data_dir: str = os.environ.get("PHM_DATA_DIR", ".")
    output_dir: str = "outputs"

    # Sub-directories under data_dir
    train_subdir: str = "train"
    validation_subdir: str = "validation"

    # Label generation
    failure_offset_sec: int = DEFAULT_FAILURE_OFFSET_SEC

    # Model training
    random_seed: int = RANDOM_SEED
    n_jobs: int = -1  # use all CPUs

    # Feature extraction
    fault_band_hz: float = FAULT_BAND_HZ
    fault_harmonics: List[int] = field(default_factory=lambda: list(FAULT_HARMONICS))

    # Calibration
    calibration_range: Tuple[float, float, float] = CALIBRATION_RANGE

    # ── Derived paths ──────────────────────────────────────────────────────
    @property
    def train_dir(self) -> str:
        return os.path.join(self.data_dir, self.train_subdir)

    @property
    def validation_dir(self) -> str:
        return os.path.join(self.data_dir, self.validation_subdir)

    @property
    def model_dir(self) -> str:
        return os.path.join(self.output_dir, "models")

    @property
    def prediction_dir(self) -> str:
        return os.path.join(self.output_dir, "predictions")

    @property
    def report_dir(self) -> str:
        return os.path.join(self.output_dir, "reports")

    def ensure_dirs(self) -> None:
        """Create output directories if they do not exist."""
        for d in (self.model_dir, self.prediction_dir, self.report_dir):
            os.makedirs(d, exist_ok=True)
