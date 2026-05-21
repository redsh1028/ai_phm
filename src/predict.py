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
    logger.info("=" * 60)
    logger.info("STEP 1: Loading MoE models & metadata")
    logger.info("=" * 60)

    import joblib
    import json
    
    # Load Normal Expert
    meta_normal_path = os.path.join(cfg.model_normal_dir, "meta.json")
    if not os.path.isfile(meta_normal_path):
        logger.error("Missing normal expert metadata. Exiting.")
        sys.exit(1)
    with open(meta_normal_path, "r") as f:
        meta_normal = json.load(f)
    
    models_normal = {}
    for name in meta_normal["ensemble_weights"]:
        model_path = os.path.join(cfg.model_normal_dir, f"{name}.joblib")
        if os.path.isfile(model_path):
            models_normal[name] = joblib.load(model_path)
            
    # Load Thermal Expert
    meta_thermal_path = os.path.join(cfg.model_thermal_dir, "meta.json")
    if not os.path.isfile(meta_thermal_path):
        logger.error("Missing thermal expert metadata. Exiting.")
        sys.exit(1)
    with open(meta_thermal_path, "r") as f:
        meta_thermal = json.load(f)
        
    models_thermal = {}
    for name in meta_thermal["ensemble_weights"]:
        model_path = os.path.join(cfg.model_thermal_dir, f"{name}.joblib")
        if os.path.isfile(model_path):
            models_thermal[name] = joblib.load(model_path)

    # Load RPM predictor
    rpm_predictor = None
    rpm_model_path = os.path.join(cfg.model_dir, "rpm_predictor.joblib")
    if os.path.isfile(rpm_model_path):
        rpm_predictor = joblib.load(rpm_model_path)

    # Load Temp classifier
    temp_classifier = None
    temp_model_path = os.path.join(cfg.model_dir, "temp_classifier.joblib")
    if os.path.isfile(temp_model_path):
        temp_classifier = joblib.load(temp_model_path)

    # ── 2. Load & extract validation features ──────────────────────────────
    logger.info("Loading validation data from %s …", cfg.validation_dir)
    val_runs = load_all_runs(cfg.validation_dir)
    if not val_runs:
        logger.error("No validation runs found. Exiting.")
        sys.exit(1)

    val_df = build_feature_table(val_runs, cfg, include_labels=False, rpm_predictor=rpm_predictor, temp_classifier=temp_classifier)

    # ── 3. Predict with MoE Routing ─────────────────────────────────────────
    run_ids = val_df["run_id"].unique()
    prediction_rows = []

    for run_id in run_ids:
        run_mask = val_df["run_id"] == run_id
        run_df = val_df.loc[run_mask].sort_values("sample_index")
        last_row = run_df.iloc[[-1]]
        
        is_high_temp = float(last_row["is_high_temp_mode"].values[0])
        
        if is_high_temp > 0.5:
            expert_name = "THERMAL"
            models = models_thermal
            meta = meta_thermal
        else:
            expert_name = "NORMAL"
            models = models_normal
            meta = meta_normal
            
        feature_cols = meta["feature_cols"]
        weights = meta["ensemble_weights"]
        calibration_factor = meta["calibration_factor"]

        import numpy as np
        import pandas as pd
        
        X_last = pd.DataFrame(columns=feature_cols)
        for col in feature_cols:
            if col in last_row.columns:
                X_last[col] = last_row[col].values
            else:
                X_last[col] = [np.nan]
        X_last = X_last.values.astype(np.float32)

        ind_preds = {}
        for model_name, model in models.items():
            pred = float(np.clip(model.predict(X_last), 0, None)[0])
            ind_preds[model_name] = np.array([pred])

        active_weights = {k: weights[k] for k in models if k in weights}
        w_total = sum(active_weights.values())
        active_weights = {k: v / w_total for k, v in active_weights.items()}

        ens_pred = ensemble_predict(ind_preds, active_weights)
        raw_rul = float(ens_pred[0])
        calibrated_rul = max(0.0, raw_rul * calibration_factor)

        prediction_rows.append({
            "run_id": run_id,
            "expert": expert_name,
            "ensemble_raw": raw_rul,
            "calibration_factor": calibration_factor,
            "predicted_RUL_sec": calibrated_rul
        })
        logger.info(f"  {run_id}: expert={expert_name} raw={raw_rul:.1f}  calibrated={calibrated_rul:.1f} sec")

    # ── 4. Save predictions ────────────────────────────────────────────────
    pred_df = pd.DataFrame(prediction_rows)
    submission_df = pred_df[["run_id", "predicted_RUL_sec"]].copy()
    submission_df.columns = ["Dataset", "Predicted_RUL_sec"]
    xlsx_path = os.path.join(cfg.prediction_dir, "team_validation.xlsx")
    submission_df.to_excel(xlsx_path, index=False, sheet_name="Predictions")
    logger.info("Submission file saved → %s", xlsx_path)

if __name__ == "__main__":
    main()
