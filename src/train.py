#!/usr/bin/env python3
"""
Training script for the KSPHM-KIMM 2026 PHM Data Challenge.

Usage
-----
    python src/train.py --data_dir /data --output_dir outputs --failure_offset_sec 300

The script will:
  1. Load training TDMS files and extract features.
  2. Train all available models.
  3. Run Leave-One-Bearing-Out cross-validation.
  4. Search for the best calibration factor.
  5. Save trained models, CV results, feature importances, and a summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

# Ensure project root is on sys.path so ``src`` can be imported
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.config import PipelineConfig
from src.data_loader import load_all_runs
from src.dataset import build_feature_table
from src.evaluate import (
    a_rul_score,
    leave_one_bearing_out_cv,
    partial_run_validation,
    search_calibration_factor,
)
from src.models import (
    compute_ensemble_weights,
    ensemble_predict,
    get_feature_importance,
    get_model_catalogue,
)
from src.utils import get_logger

logger = get_logger("train")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train bearing RUL prediction models."
    )
    parser.add_argument(
        "--data_dir", type=str, default=".",
        help="Root data directory containing train/ and validation/ sub-folders.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Directory for models, predictions, and reports.",
    )
    parser.add_argument(
        "--failure_offset_sec", type=int, default=300,
        help="Seconds after last recorded file to assume failure.",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=-1,
        help="Number of parallel jobs (-1 = all CPUs).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
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
        failure_offset_sec=args.failure_offset_sec,
        n_jobs=args.n_jobs,
        random_seed=args.seed,
    )
    cfg.ensure_dirs()

    t0 = time.time()

    # ── 1. Load & extract features ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Loading training data from %s", cfg.train_dir)
    logger.info("=" * 60)

    train_runs = load_all_runs(cfg.train_dir)
    if not train_runs:
        logger.error("No training runs found under %s. Exiting.", cfg.train_dir)
        sys.exit(1)

    logger.info("Found %d training runs: %s",
                len(train_runs), [r.name for r in train_runs])

    train_df = build_feature_table(train_runs, cfg, include_labels=True)
    train_csv = os.path.join(cfg.report_dir, "train_features.csv")
    train_df.to_csv(train_csv, index=False)
    logger.info("Saved training features → %s  (%d rows × %d cols)",
                train_csv, *train_df.shape)

    # ── Determine feature columns ─────────────────────────────────────────
    meta_cols = {"run_id", "RUL_sec", "remaining_fraction_label"}
    feature_cols = [c for c in train_df.columns if c not in meta_cols]
    logger.info("Using %d features.", len(feature_cols))

    # ── 1.5 Train RPM Predictor ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1.5: Training RPM Predictor")
    logger.info("=" * 60)
    import joblib
    from lightgbm import LGBMRegressor
    rpm_features = [c for c in train_df.columns if not any(x in c for x in ['1x', '2x', '3x', '4x', 'Motor_speed', 'Torque', 'TC_SP', 'roll', 'RUL', 'run_id', 'remaining_fraction_label'])]
    logger.info("Using %d base features for RPM prediction", len(rpm_features))
    X_rpm = train_df[rpm_features]
    y_rpm = train_df['Motor_speed_rpm_mean']
    rpm_predictor = LGBMRegressor(n_estimators=100, random_state=cfg.random_seed, verbose=-1)
    rpm_predictor.fit(X_rpm, y_rpm)
    rpm_model_path = os.path.join(cfg.model_dir, "rpm_predictor.joblib")
    joblib.dump(rpm_predictor, rpm_model_path)
    logger.info("Saved RPM predictor -> %s", rpm_model_path)

    # ── 1.6 Train High Temp Classifier ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1.6: Training High Temp Classifier")
    logger.info("=" * 60)
    from lightgbm import LGBMClassifier
    y_temp = train_df['is_high_temp_mode']
    temp_classifier = LGBMClassifier(n_estimators=100, random_state=cfg.random_seed, verbose=-1)
    temp_classifier.fit(X_rpm, y_temp)
    temp_model_path = os.path.join(cfg.model_dir, "temp_classifier.joblib")
    joblib.dump(temp_classifier, temp_model_path)
    logger.info("Saved Temp classifier -> %s", temp_model_path)

    
    # ── 2. Define Expert Training Function ─────────────────────────────────
    def train_expert(mode_name, expert_df, feature_cols, cfg, out_dir, run_lobocv=True):
        logger.info("*" * 60)
        logger.info(f"TRAINING EXPERT: {mode_name}")
        logger.info("*" * 60)

        catalogue = get_model_catalogue(cfg)
        
        cv_results = {}
        all_cv_actual = {}
        all_cv_pred = {}
        ens_score = 0.0
        best_factor = 1.0
        best_cal_score = 0.0
        weights = {k: 1.0/len(catalogue) for k in catalogue.keys()}
        model_scores = {k: 0.0 for k in catalogue.keys()}

        if run_lobocv and len(expert_df['run_id'].unique()) > 1:
            for name, pipeline in catalogue.items():
                from sklearn.base import clone
                overall, per_run, y_act, y_pred = leave_one_bearing_out_cv(
                    expert_df, feature_cols, "RUL_sec", lambda p=pipeline: clone(p), cfg
                )
                cv_results[name] = {"overall_score": overall, "per_run_scores": per_run}
                all_cv_actual[name] = y_act
                all_cv_pred[name] = y_pred
                model_scores[name] = overall
            
            weights = compute_ensemble_weights(model_scores)
            ref_actual = list(all_cv_actual.values())[0]
            ens_pred = ensemble_predict(all_cv_pred, weights)
            ens_score = a_rul_score(ref_actual, ens_pred)
            best_factor, best_cal_score = search_calibration_factor(ref_actual, ens_pred, cfg)
            logger.info(f"[{mode_name}] CV Calibrated Score: {best_cal_score:.4f} (factor: {best_factor:.2f})")
            
            # Partial run validation
            best_model_name = max(model_scores, key=model_scores.get)
            best_pipeline = catalogue[best_model_name]
            from sklearn.base import clone as sk_clone
            partial_df = partial_run_validation(
                expert_df, feature_cols, "RUL_sec", lambda: sk_clone(best_pipeline), cfg
            )
            partial_csv = os.path.join(cfg.report_dir, f"partial_run_validation_{mode_name}.csv")
            partial_df.to_csv(partial_csv, index=False)
            logger.info(f"[{mode_name}] Partial-run validation saved")
        else:
            logger.warning(f"[{mode_name}] Not enough runs for LOBOCV. Using default weights and factor 1.0.")

        # Train final models on ALL expert data
        X_all = expert_df[feature_cols].values.astype(np.float32)
        y_all = expert_df["RUL_sec"].values.astype(np.float32)
        importances_all = {}

        for name, pipeline in catalogue.items():
            from sklearn.base import clone
            model = clone(pipeline)
            model.fit(X_all, y_all)
            model_path = os.path.join(out_dir, f"{name}.joblib")
            joblib.dump(model, model_path)
            imp = get_feature_importance(model, feature_cols)
            if imp is not None:
                importances_all[name] = imp

        meta = {
            "feature_cols": feature_cols,
            "ensemble_weights": weights,
            "calibration_factor": best_factor,
            "cv_scores": model_scores,
            "ensemble_cv_score": ens_score,
            "calibrated_ensemble_score": best_cal_score,
        }
        meta_path = os.path.join(out_dir, "meta.json")
        import json
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
            
        return meta

    # ── 3. Train Experts ───────────────────────────────────────────────────
    df_normal = train_df[train_df['is_high_temp_mode'] == 0.0].copy()
    df_thermal = train_df[train_df['is_high_temp_mode'] == 1.0].copy()
    
    meta_normal = train_expert("normal", df_normal, feature_cols, cfg, cfg.model_normal_dir, run_lobocv=True)
    meta_thermal = train_expert("thermal", df_thermal, feature_cols, cfg, cfg.model_thermal_dir, run_lobocv=False)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE  (%.1f s)", elapsed)
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
