#!/usr/bin/env python3
"""
Prediction script for the KSPHM-KIMM 2026 PHM Data Challenge.

Usage
-----
    python src/predict.py --data_dir /data --output_dir outputs

The script will:
  1. Load previously trained models and metadata from ``outputs/models/``.
  2. Extract features from validation TDMS data.
  3. Generate per-run RUL predictions (last available sample).
  4. Apply calibration factor.
  5. Save results to CSV and Excel.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.config import PipelineConfig
from src.data_loader import load_all_runs
from src.dataset import build_feature_table
from src.models import ensemble_predict
from src.utils import get_logger

logger = get_logger("predict")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict bearing RUL for validation data."
    )
    parser.add_argument(
        "--data_dir", type=str, default=".",
        help="Root data directory containing validation/ sub-folder.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Directory containing trained models and for saving predictions.",
    )
    parser.add_argument(
        "--validation_dir", type=str, default="validation",
        help="Name of the validation subdirectory under data_dir.",
    )
    return parser.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    cfg = PipelineConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        validation_subdir=args.validation_dir,
    )
    cfg.ensure_dirs()

    # ── 1. Load trained models & metadata ──────────────────────────────────
    meta_path = os.path.join(cfg.model_dir, "meta.json")
    if not os.path.isfile(meta_path):
        logger.error("meta.json not found at %s. Did you run train.py first?", meta_path)
        sys.exit(1)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    feature_cols: List[str] = meta["feature_cols"]
    weights: Dict[str, float] = meta["ensemble_weights"]
    calibration_factor: float = meta["calibration_factor"]
    logger.info("Loaded metadata: %d features, calibration_factor=%.2f",
                len(feature_cols), calibration_factor)

    # Load model pipelines
    models: Dict[str, object] = {}
    for model_name in weights:
        model_path = os.path.join(cfg.model_dir, f"{model_name}.joblib")
        if not os.path.isfile(model_path):
            logger.warning("Model file not found: %s – skipping.", model_path)
            continue
        models[model_name] = joblib.load(model_path)
        logger.info("Loaded model: %s", model_name)

    if not models:
        logger.error("No models could be loaded. Exiting.")
        sys.exit(1)

    # Load RPM predictor if available
    rpm_predictor = None
    rpm_model_path = os.path.join(cfg.model_dir, "rpm_predictor.joblib")
    if os.path.isfile(rpm_model_path):
        rpm_predictor = joblib.load(rpm_model_path)
        logger.info("Loaded RPM predictor: %s", rpm_model_path)

    # Load Temp classifier if available
    temp_classifier = None
    temp_model_path = os.path.join(cfg.model_dir, "temp_classifier.joblib")
    if os.path.isfile(temp_model_path):
        temp_classifier = joblib.load(temp_model_path)
        logger.info("Loaded Temp classifier: %s", temp_model_path)

    # ── 2. Load & extract validation features ──────────────────────────────
    logger.info("Loading validation data from %s …", cfg.validation_dir)
    val_runs = load_all_runs(cfg.validation_dir)
    if not val_runs:
        logger.error("No validation runs found under %s. Exiting.", cfg.validation_dir)
        sys.exit(1)

    logger.info("Building features for %d validation runs …", len(val_runs))
    val_df = build_feature_table(val_runs, cfg, include_labels=False, rpm_predictor=rpm_predictor, temp_classifier=temp_classifier)

    val_csv = os.path.join(cfg.report_dir, "validation_features.csv")
    val_df.to_csv(val_csv, index=False)
    logger.info("Saved validation features → %s  (%d rows × %d cols)",
                val_csv, *val_df.shape)

    # ── 3. Predict ─────────────────────────────────────────────────────────
    run_ids = val_df["run_id"].unique()
    prediction_rows: List[Dict] = []
    all_individual_preds: List[Dict] = []

    for run_id in run_ids:
        run_mask = val_df["run_id"] == run_id
        run_df = val_df.loc[run_mask].sort_values("sample_index")

        # Use ALL available samples for context, but predict at the last sample
        last_row = run_df.iloc[[-1]]

        # Ensure feature columns exist (fill missing with NaN)
        X_last = pd.DataFrame(columns=feature_cols)
        for col in feature_cols:
            if col in last_row.columns:
                X_last[col] = last_row[col].values
            else:
                X_last[col] = [np.nan]
        X_last = X_last.values.astype(np.float32)

        # Individual model predictions
        ind_preds: Dict[str, np.ndarray] = {}
        row_data: Dict = {"run_id": run_id}

        for model_name, model in models.items():
            pred = float(np.clip(model.predict(X_last), 0, None)[0])
            ind_preds[model_name] = np.array([pred])
            row_data[f"pred_{model_name}"] = pred

        # Ensemble (only over loaded models)
        active_weights = {k: weights[k] for k in models if k in weights}
        w_total = sum(active_weights.values())
        active_weights = {k: v / w_total for k, v in active_weights.items()}

        ens_pred = ensemble_predict(ind_preds, active_weights)
        raw_rul = float(ens_pred[0])
        calibrated_rul = max(0.0, raw_rul * calibration_factor)

        row_data["ensemble_raw"] = raw_rul
        row_data["calibration_factor"] = calibration_factor
        row_data["predicted_RUL_sec"] = calibrated_rul

        prediction_rows.append(row_data)
        logger.info("  %s: raw=%.1f  calibrated=%.1f sec", run_id, raw_rul, calibrated_rul)

    # ── 4. Save predictions ────────────────────────────────────────────────
    pred_df = pd.DataFrame(prediction_rows)

    # Full CSV with all details
    full_csv = os.path.join(cfg.prediction_dir, "validation_predictions.csv")
    pred_df.to_csv(full_csv, index=False)
    logger.info("Full predictions saved → %s", full_csv)

    # Clean Excel for submission
    submission_df = pred_df[["run_id", "predicted_RUL_sec"]].copy()
    submission_df.columns = ["Dataset", "Predicted_RUL_sec"]

    xlsx_path = os.path.join(cfg.prediction_dir, "team_validation.xlsx")
    submission_df.to_excel(xlsx_path, index=False, sheet_name="Predictions")
    logger.info("Submission file saved → %s", xlsx_path)

    logger.info("")
    logger.info("=" * 60)
    logger.info("PREDICTION COMPLETE")
    logger.info("=" * 60)
    logger.info("\n%s", submission_df.to_string(index=False))


if __name__ == "__main__":
    main()
