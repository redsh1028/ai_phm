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

    # ── 2. Get model catalogue ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Model catalogue")
    logger.info("=" * 60)

    catalogue = get_model_catalogue(cfg)
    logger.info("Models: %s", list(catalogue.keys()))

    # ── 3. Leave-One-Bearing-Out CV ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Leave-One-Bearing-Out cross-validation")
    logger.info("=" * 60)

    cv_results: Dict[str, Dict] = {}
    all_cv_actual: Dict[str, np.ndarray] = {}
    all_cv_pred: Dict[str, np.ndarray] = {}

    for name, pipeline in catalogue.items():
        logger.info("── CV for %s ──", name)

        def _factory(p=pipeline):
            """Clone-like factory that returns a fresh pipeline."""
            from sklearn.base import clone
            return clone(p)

        overall, per_run, y_act, y_pred = leave_one_bearing_out_cv(
            train_df, feature_cols, "RUL_sec", _factory, cfg,
        )

        cv_results[name] = {
            "overall_score": overall,
            "per_run_scores": per_run,
        }
        all_cv_actual[name] = y_act
        all_cv_pred[name] = y_pred

        logger.info("  Overall A_RUL score: %.4f", overall)
        for run_id, score in per_run.items():
            logger.info("    %s: %.4f", run_id, score)

    # ── 4. Ensemble ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Ensemble")
    logger.info("=" * 60)

    model_scores = {k: v["overall_score"] for k, v in cv_results.items()}
    weights = compute_ensemble_weights(model_scores)
    logger.info("Ensemble weights: %s",
                {k: f"{v:.3f}" for k, v in weights.items()})

    # Build ensemble CV predictions by averaging per-sample
    # All models should have the same y_actual (same fold order)
    ref_actual = list(all_cv_actual.values())[0]
    ens_pred = ensemble_predict(all_cv_pred, weights)
    ens_score = a_rul_score(ref_actual, ens_pred)
    logger.info("Ensemble A_RUL score (CV): %.4f", ens_score)

    # ── 5. Calibration factor search ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5: Calibration factor search")
    logger.info("=" * 60)

    best_factor, best_cal_score = search_calibration_factor(
        ref_actual, ens_pred, cfg
    )
    logger.info("Selected calibration factor: %.2f  →  score %.4f",
                best_factor, best_cal_score)

    # ── 6. Partial-run validation ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: Partial-run validation")
    logger.info("=" * 60)

    # Use the best single model for partial-run validation to save time
    best_model_name = max(model_scores, key=model_scores.get)  # type: ignore[arg-type]
    best_pipeline = catalogue[best_model_name]

    from sklearn.base import clone as sk_clone

    partial_df = partial_run_validation(
        train_df, feature_cols, "RUL_sec",
        lambda: sk_clone(best_pipeline),
        cfg,
    )
    partial_csv = os.path.join(cfg.report_dir, "partial_run_validation.csv")
    partial_df.to_csv(partial_csv, index=False)
    logger.info("Partial-run validation saved → %s", partial_csv)
    logger.info("\n%s", partial_df.to_string(index=False))

    # ── 7. Train final models on ALL training data ─────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 7: Training final models on full training set")
    logger.info("=" * 60)

    X_all = train_df[feature_cols].values.astype(np.float32)
    y_all = train_df["RUL_sec"].values.astype(np.float32)

    trained_models: Dict[str, object] = {}
    importances_all: Dict[str, Dict[str, float]] = {}

    for name, pipeline in catalogue.items():
        logger.info("Training %s on full data …", name)
        from sklearn.base import clone as sk_clone2
        model = sk_clone2(pipeline)
        model.fit(X_all, y_all)
        trained_models[name] = model

        # Save model
        model_path = os.path.join(cfg.model_dir, f"{name}.joblib")
        joblib.dump(model, model_path)
        logger.info("  Saved → %s", model_path)

        # Feature importance
        imp = get_feature_importance(model, feature_cols)
        if imp is not None:
            importances_all[name] = imp

    # Save ensemble weights & calibration factor
    meta = {
        "feature_cols": feature_cols,
        "ensemble_weights": weights,
        "calibration_factor": best_factor,
        "cv_scores": model_scores,
        "ensemble_cv_score": ens_score,
        "calibrated_ensemble_score": best_cal_score,
    }
    meta_path = os.path.join(cfg.model_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved meta → %s", meta_path)

    # ── 8. Save reports ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 8: Saving reports")
    logger.info("=" * 60)

    # CV results
    cv_rows = []
    for name, res in cv_results.items():
        for run_id, score in res["per_run_scores"].items():
            cv_rows.append({
                "model": name,
                "held_out_run": run_id,
                "a_rul_score": score,
            })
        cv_rows.append({
            "model": name,
            "held_out_run": "OVERALL",
            "a_rul_score": res["overall_score"],
        })
    # Add ensemble row
    cv_rows.append({
        "model": "Ensemble",
        "held_out_run": "OVERALL",
        "a_rul_score": ens_score,
    })
    cv_rows.append({
        "model": "Ensemble (calibrated)",
        "held_out_run": "OVERALL",
        "a_rul_score": best_cal_score,
    })
    cv_df = pd.DataFrame(cv_rows)
    cv_csv = os.path.join(cfg.report_dir, "cv_results.csv")
    cv_df.to_csv(cv_csv, index=False)
    logger.info("CV results saved → %s", cv_csv)

    # Feature importance
    if importances_all:
        imp_rows = []
        for model_name, imp_dict in importances_all.items():
            for feat, val in imp_dict.items():
                imp_rows.append({"model": model_name, "feature": feat, "importance": val})
        imp_df = pd.DataFrame(imp_rows)
        imp_csv = os.path.join(cfg.report_dir, "feature_importances.csv")
        imp_df.to_csv(imp_csv, index=False)
        logger.info("Feature importances saved → %s", imp_csv)

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE  (%.1f s)", elapsed)
    logger.info("=" * 60)
    logger.info("Model scores (Leave-One-Bearing-Out CV):")
    for name, score in sorted(model_scores.items(), key=lambda x: -x[1]):
        logger.info("  %-25s  A_RUL = %.4f  weight = %.3f",
                     name, score, weights[name])
    logger.info("  %-25s  A_RUL = %.4f", "Ensemble", ens_score)
    logger.info("  %-25s  A_RUL = %.4f  (factor = %.2f)",
                "Ensemble (calibrated)", best_cal_score, best_factor)
    logger.info("")
    logger.info("Saved artefacts:")
    logger.info("  Models       : %s", cfg.model_dir)
    logger.info("  Reports      : %s", cfg.report_dir)
    logger.info("  Features CSV : %s", train_csv)
    logger.info("  CV results   : %s", cv_csv)


if __name__ == "__main__":
    main()
