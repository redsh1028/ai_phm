"""
Model definitions and ensemble logic.

All models are wrapped in sklearn Pipelines with imputation and optional
scaling.  The module gracefully degrades when optional packages
(xgboost, lightgbm) are absent.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

from src.config import PipelineConfig
from src.utils import get_logger

logger = get_logger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Optional imports
# ────────────────────────────────────────────────────────────────────────────

def _try_import(module_name: str):
    """Return module if available, else None."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


# ────────────────────────────────────────────────────────────────────────────
# Model catalogue
# ────────────────────────────────────────────────────────────────────────────

def get_model_catalogue(cfg: PipelineConfig) -> Dict[str, Pipeline]:
    """
    Return a dictionary of model_name → sklearn Pipeline.

    Each pipeline has:
      1. SimpleImputer (median strategy)
      2. StandardScaler (optional — helps some models, harmless for trees)
      3. Regressor

    Parameters
    ----------
    cfg : PipelineConfig
        Pipeline configuration (for seed, n_jobs).

    Returns
    -------
    dict
        Model name → Pipeline.
    """
    seed = cfg.random_seed
    n_jobs = cfg.n_jobs
    models: Dict[str, Pipeline] = {}

    # ── Scikit-learn built-ins ─────────────────────────────────────────────
    models["RandomForest"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", TransformedTargetRegressor(
            regressor=RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=5,
                random_state=seed,
                n_jobs=n_jobs,
            ),
            func=np.log1p,
            inverse_func=np.expm1
        )),
    ])

    models["ExtraTrees"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", TransformedTargetRegressor(
            regressor=ExtraTreesRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=5,
                random_state=seed,
                n_jobs=n_jobs,
            ),
            func=np.log1p,
            inverse_func=np.expm1
        )),
    ])

    models["GradientBoosting"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", TransformedTargetRegressor(
            regressor=GradientBoostingRegressor(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=5,
                learning_rate=0.05,
                random_state=seed,
            ),
            func=np.log1p,
            inverse_func=np.expm1
        )),
    ])

    models["HistGradientBoosting"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", TransformedTargetRegressor(
            regressor=HistGradientBoostingRegressor(
                max_iter=300,
                max_depth=5,
                min_samples_leaf=5,
                learning_rate=0.05,
                random_state=seed,
            ),
            func=np.log1p,
            inverse_func=np.expm1
        )),
    ])

    # ── Optional: XGBoost ──────────────────────────────────────────────────
    xgb = _try_import("xgboost")
    if xgb is not None:
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", xgb.XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                n_jobs=n_jobs,
                verbosity=0,
            )),
        ])
        logger.info("XGBoost available – added to catalogue.")
    else:
        logger.info("XGBoost not installed – skipping.")

    # ── Optional: LightGBM ─────────────────────────────────────────────────
    lgb = _try_import("lightgbm")
    if lgb is not None:
        models["LightGBM"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", TransformedTargetRegressor(
                regressor=lgb.LGBMRegressor(
                    n_estimators=300,
                    max_depth=5,
                    min_child_samples=5,
                    learning_rate=0.05,
                    random_state=seed,
                    n_jobs=n_jobs,
                    verbose=-1,
                ),
                func=np.log1p,
                inverse_func=np.expm1
            )),
        ])
        logger.info("LightGBM available – added to catalogue.")
    else:
        logger.info("LightGBM not installed – skipping.")

    return models


# ────────────────────────────────────────────────────────────────────────────
# Feature importance extraction
# ────────────────────────────────────────────────────────────────────────────

def get_feature_importance(
    pipeline: Pipeline,
    feature_names: List[str],
) -> Optional[Dict[str, float]]:
    """
    Extract feature importances from the fitted pipeline's last step.
    Returns None if the model does not expose importances.
    """
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return dict(zip(feature_names, importances))
    return None


# ────────────────────────────────────────────────────────────────────────────
# Ensemble helpers
# ────────────────────────────────────────────────────────────────────────────

def compute_ensemble_weights(
    model_scores: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute normalised ensemble weights from per-model validation scores.

    Higher score → higher weight.  Scores <= 0 are clipped to a small ε.
    """
    eps = 1e-6
    raw = {k: max(v, eps) for k, v in model_scores.items()}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def ensemble_predict(
    predictions: Dict[str, np.ndarray],
    weights: Dict[str, float],
) -> np.ndarray:
    """Weighted average ensemble prediction."""
    names = list(predictions.keys())
    w = np.array([weights[n] for n in names], dtype=np.float64)
    w /= w.sum()
    preds = np.column_stack([predictions[n] for n in names])
    return (preds * w[np.newaxis, :]).sum(axis=1)
