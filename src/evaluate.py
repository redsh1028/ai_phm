"""
Scoring & evaluation utilities.

Implements the official KSPHM-KIMM 2026 challenge metric and helpers
for Leave-One-Bearing-Out cross-validation, partial-run validation,
and calibration-factor search.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import PARTIAL_RUN_CUTOFFS, PipelineConfig
from src.utils import get_logger

logger = get_logger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Official scoring function
# ────────────────────────────────────────────────────────────────────────────

def a_rul_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Compute the official challenge score.

    Er = 100 * (ActRUL - PredRUL) / ActRUL

    If Er <= 0:  A_RUL = exp(-ln(0.5) * Er / 20)
    Else:        A_RUL = exp(+ln(0.5) * Er / 50)

    Returns the mean A_RUL across all samples.

    Parameters
    ----------
    actual : array-like
        Ground-truth RUL values (seconds).
    predicted : array-like
        Predicted RUL values (seconds).
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    # Guard against division by zero when actual RUL is 0
    safe_actual = np.where(actual == 0, 1e-6, actual)

    er = 100.0 * (safe_actual - predicted) / safe_actual

    ln05 = np.log(0.5)
    a_scores = np.where(
        er <= 0,
        np.exp(-ln05 * er / 20.0),
        np.exp(ln05 * er / 50.0),
    )

    return float(np.mean(a_scores))


def a_rul_score_single(actual: float, predicted: float) -> float:
    """Score for a single sample (convenience wrapper)."""
    return a_rul_score(np.array([actual]), np.array([predicted]))


# ────────────────────────────────────────────────────────────────────────────
# Calibration factor search
# ────────────────────────────────────────────────────────────────────────────

def search_calibration_factor(
    actual: np.ndarray,
    raw_predicted: np.ndarray,
    cfg: PipelineConfig,
) -> Tuple[float, float]:
    """
    Grid-search for the calibration factor that maximises the official score.

    final_pred = raw_pred * calibration_factor

    Returns
    -------
    (best_factor, best_score)
    """
    start, stop, step = cfg.calibration_range
    factors = np.arange(start, stop, step)
    best_factor = 1.0
    best_score = -1.0

    for f in factors:
        calibrated = raw_predicted * f
        score = a_rul_score(actual, calibrated)
        if score > best_score:
            best_score = score
            best_factor = f

    logger.info("Best calibration factor: %.2f  →  score %.4f", best_factor, best_score)
    return float(best_factor), float(best_score)


# ────────────────────────────────────────────────────────────────────────────
# Leave-One-Bearing-Out CV
# ────────────────────────────────────────────────────────────────────────────

def leave_one_bearing_out_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    model_factory,
    cfg: PipelineConfig,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    """
    Perform Leave-One-Bearing-Out cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        Training feature table with ``run_id`` and *label_col* columns.
    feature_cols : list[str]
        Names of feature columns.
    label_col : str
        Label column name (e.g. ``RUL_sec``).
    model_factory : callable
        A zero-argument callable that returns a fresh (unfitted) model/pipeline.
    cfg : PipelineConfig
        Pipeline configuration.

    Returns
    -------
    (overall_score, per_run_scores, all_actual, all_predicted)
    """
    runs = df["run_id"].unique()
    per_run_scores: Dict[str, float] = {}
    all_actual_list: List[np.ndarray] = []
    all_pred_list: List[np.ndarray] = []

    for held_out in runs:
        train_mask = df["run_id"] != held_out
        val_mask = df["run_id"] == held_out

        X_train = df.loc[train_mask, feature_cols].values.astype(np.float32)
        y_train = df.loc[train_mask, label_col].values.astype(np.float32)
        X_val = df.loc[val_mask, feature_cols].values.astype(np.float32)
        y_val = df.loc[val_mask, label_col].values.astype(np.float32)

        model = model_factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val).astype(np.float32)

        # Clip negative predictions
        y_pred = np.clip(y_pred, 0, None)

        score = a_rul_score(y_val, y_pred)
        per_run_scores[held_out] = score

        all_actual_list.append(y_val)
        all_pred_list.append(y_pred)

    all_actual = np.concatenate(all_actual_list)
    all_predicted = np.concatenate(all_pred_list)
    overall_score = a_rul_score(all_actual, all_predicted)

    return overall_score, per_run_scores, all_actual, all_predicted


# ────────────────────────────────────────────────────────────────────────────
# Partial-run validation
# ────────────────────────────────────────────────────────────────────────────

def partial_run_validation(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    model_factory,
    cfg: PipelineConfig,
    cutoffs: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Simulate validation by cutting held-out runs at different percentages.

    For each cutoff, the last available sample's predicted RUL is compared
    with the true RUL.

    Returns
    -------
    pd.DataFrame
        Columns: run_id, cutoff, true_rul, predicted_rul, a_rul_score
    """
    if cutoffs is None:
        cutoffs = list(PARTIAL_RUN_CUTOFFS)

    runs = df["run_id"].unique()
    results: List[Dict] = []

    for held_out in runs:
        train_mask = df["run_id"] != held_out
        val_df = df.loc[df["run_id"] == held_out].sort_values("sample_index")

        X_train = df.loc[train_mask, feature_cols].values.astype(np.float32)
        y_train = df.loc[train_mask, label_col].values.astype(np.float32)

        model = model_factory()
        model.fit(X_train, y_train)

        n_total = len(val_df)
        for cutoff in cutoffs:
            n_cut = max(1, int(n_total * cutoff))
            partial = val_df.iloc[:n_cut]
            last_row = partial.iloc[[-1]]

            X_last = last_row[feature_cols].values.astype(np.float32)
            pred_rul = float(np.clip(model.predict(X_last), 0, None)[0])
            true_rul = float(last_row[label_col].values[0])

            score = a_rul_score_single(true_rul, pred_rul)

            results.append({
                "run_id": held_out,
                "cutoff": cutoff,
                "true_rul": true_rul,
                "predicted_rul": pred_rul,
                "a_rul_score": score,
            })

    return pd.DataFrame(results)
