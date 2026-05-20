"""
Dataset construction module.

Builds feature tables (DataFrames) for training and validation bearing runs,
generates RUL labels for training data, and saves everything to CSV.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import ACQUISITION_INTERVAL_SEC, PipelineConfig
from src.data_loader import BearingRun, load_all_runs
from src.features import extract_record_features
from src.utils import get_logger

logger = get_logger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Label generation
# ────────────────────────────────────────────────────────────────────────────

def generate_rul_labels(
    num_files: int,
    failure_offset_sec: int,
    interval_sec: int = ACQUISITION_INTERVAL_SEC,
) -> np.ndarray:
    """
    Generate RUL labels (seconds) for a training run.

    Parameters
    ----------
    num_files : int
        Number of TDMS files (acquisition points) in the run.
    failure_offset_sec : int
        Seconds after the last recorded sample to assume failure occurred.
    interval_sec : int
        Seconds between consecutive acquisitions (default 600).

    Returns
    -------
    np.ndarray
        Shape ``(num_files,)`` with RUL in seconds for each sample.
    """
    last_sample_time = (num_files - 1) * interval_sec
    failure_time = last_sample_time + failure_offset_sec
    rul = np.array(
        [failure_time - i * interval_sec for i in range(num_files)],
        dtype=np.float32,
    )
    return rul


# ────────────────────────────────────────────────────────────────────────────
# Feature table builders
# ────────────────────────────────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add rolling mean and std for key features grouped by run_id."""
    cols_to_roll = [
        "CH1_rms", "CH1_kurtosis", "CH2_rms", "CH3_rms", "CH4_rms",
        "CH1_mean", "CH2_mean", "CH3_mean", "CH4_mean",
        "TC_SP_Front_C_mean", "Motor_speed_rpm_mean", "Torque_Nm_mean",
        "CH3_Cage_1x", "CH1_BPFI_1x"
    ]
    
    # Only keep columns that exist in df
    valid_cols = [c for c in cols_to_roll if c in df.columns]
    if not valid_cols:
        return df

    # We must sort by run_id and sample_index to ensure rolling is chronological
    df = df.sort_values(["run_id", "sample_index"]).reset_index(drop=True)

    grouped = df.groupby("run_id")[valid_cols]
    
    # Calculate rolling statistics
    rolled_mean = grouped.rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
    rolled_std = grouped.rolling(window=window, min_periods=1).std().reset_index(level=0, drop=True)
    
    # Rename columns
    rolled_mean.columns = [f"{c}_roll_mean" for c in valid_cols]
    rolled_std.columns = [f"{c}_roll_std" for c in valid_cols]
    
    # Concat back to df
    df = pd.concat([df, rolled_mean, rolled_std], axis=1)
    
    # bfill standard deviations which are NaN for window size 1 (first sample)
    for c in rolled_std.columns:
        df[c] = df.groupby("run_id")[c].bfill().fillna(0.0)

    return df

def build_feature_table(
    runs: List[BearingRun],
    cfg: PipelineConfig,
    include_labels: bool = False,
    rpm_predictor = None,
) -> pd.DataFrame:
    """
    Build a feature DataFrame from a list of bearing runs.

    Parameters
    ----------
    runs : list[BearingRun]
        Loaded bearing runs.
    cfg : PipelineConfig
        Pipeline configuration.
    include_labels : bool
        If True, append an ``RUL_sec`` column using :func:`generate_rul_labels`.

    Returns
    -------
    pd.DataFrame
        One row per acquisition sample across all runs.
    """
    all_rows: List[Dict[str, float]] = []

    for run in tqdm(runs, desc="Building features", unit="run"):
        n_records = len(run.records)
        rul = (
            generate_rul_labels(n_records, cfg.failure_offset_sec)
            if include_labels
            else None
        )

        for idx, rec in enumerate(
            tqdm(run.records, desc=f"  {run.name}", leave=False, unit="file")
        ):
            feats = extract_record_features(
                rec, run, cfg, sample_index=idx, total_samples=n_records,
                rpm_predictor=rpm_predictor
            )
            feats["run_id"] = run.name  # type: ignore[assignment]
            if include_labels and rul is not None:
                feats["RUL_sec"] = float(rul[idx])
                feats["remaining_fraction_label"] = float(
                    rul[idx] / rul[0]
                ) if rul[0] > 0 else 0.0
            all_rows.append(feats)

    df = pd.DataFrame(all_rows)
    df = add_rolling_features(df, window=5)
    return df


# ────────────────────────────────────────────────────────────────────────────
# Full pipeline: build + save
# ────────────────────────────────────────────────────────────────────────────

def build_and_save_datasets(cfg: PipelineConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build feature tables for train and validation data and save to CSV.

    Returns
    -------
    (train_df, val_df)
    """
    cfg.ensure_dirs()

    # ── Training data ──────────────────────────────────────────────────────
    logger.info("Loading training data from %s …", cfg.train_dir)
    train_runs = load_all_runs(cfg.train_dir)
    if not train_runs:
        raise FileNotFoundError(
            f"No training runs found under {cfg.train_dir}. "
            "Check your data layout."
        )
    logger.info("Loaded %d training runs.", len(train_runs))
    train_df = build_feature_table(train_runs, cfg, include_labels=True)
    train_csv = os.path.join(cfg.report_dir, "train_features.csv")
    train_df.to_csv(train_csv, index=False)
    logger.info("Saved training features → %s  (%d rows × %d cols)",
                train_csv, *train_df.shape)

    # ── Validation data ────────────────────────────────────────────────────
    val_df = pd.DataFrame()
    if os.path.isdir(cfg.validation_dir):
        logger.info("Loading validation data from %s …", cfg.validation_dir)
        val_runs = load_all_runs(cfg.validation_dir)
        if val_runs:
            val_df = build_feature_table(val_runs, cfg, include_labels=False)
            val_csv = os.path.join(cfg.report_dir, "validation_features.csv")
            val_df.to_csv(val_csv, index=False)
            logger.info("Saved validation features → %s  (%d rows × %d cols)",
                        val_csv, *val_df.shape)
        else:
            logger.warning("No validation runs found under %s", cfg.validation_dir)
    else:
        logger.warning(
            "Validation directory does not exist: %s – skipping.", cfg.validation_dir
        )

    return train_df, val_df
