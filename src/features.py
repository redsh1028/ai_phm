"""
Feature extraction module.

Computes time-domain, frequency-domain, and bearing-fault-frequency features
for each vibration channel (from TDMS), plus summary statistics for auxiliary
channels (from operation CSV), plus run-level temporal features.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats
from scipy.fft import rfft, rfftfreq, next_fast_len
from scipy.signal import hilbert

from src.config import (
    FAULT_FREQS_AT_1000RPM,
    OP_FEATURE_NAMES,
    VIBRATION_CHANNELS,
    VIBRATION_SAMPLE_RATE,
    PipelineConfig,
)
from src.data_loader import BearingRun, TdmsRecord, get_mean_rpm_for_file, get_operation_window
from src.utils import get_logger

logger = get_logger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Time-domain features
# ────────────────────────────────────────────────────────────────────────────

def _time_domain_features(sig: np.ndarray, prefix: str) -> Dict[str, float]:
    """Compute time-domain statistical features for *sig*."""
    features: Dict[str, float] = {}
    N = len(sig)
    if N == 0:
        return features

    # Cast to float64 for accumulation precision
    sig64 = sig.astype(np.float64)

    mean_val = float(np.mean(sig64))
    std_val = float(np.std(sig64, ddof=1)) if N > 1 else 0.0
    rms_val = float(np.sqrt(np.mean(sig64 ** 2)))
    abs_mean = float(np.mean(np.abs(sig64)))
    max_val = float(np.max(sig64))
    min_val = float(np.min(sig64))
    peak2peak = max_val - min_val
    peak_abs = float(np.max(np.abs(sig64)))

    features[f"{prefix}_mean"] = mean_val
    features[f"{prefix}_std"] = std_val
    features[f"{prefix}_rms"] = rms_val
    features[f"{prefix}_max"] = max_val
    features[f"{prefix}_min"] = min_val
    features[f"{prefix}_peak2peak"] = peak2peak
    features[f"{prefix}_abs_mean"] = abs_mean
    features[f"{prefix}_energy"] = float(np.sum(sig64 ** 2))

    # Higher-order statistics
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features[f"{prefix}_skewness"] = float(sp_stats.skew(sig64, nan_policy="omit"))
        features[f"{prefix}_kurtosis"] = float(sp_stats.kurtosis(sig64, nan_policy="omit"))

    # Condition indicators
    features[f"{prefix}_crest_factor"] = peak_abs / rms_val if rms_val > 0 else np.nan
    features[f"{prefix}_impulse_factor"] = peak_abs / abs_mean if abs_mean > 0 else np.nan
    features[f"{prefix}_shape_factor"] = rms_val / abs_mean if abs_mean > 0 else np.nan

    # Clearance factor: peak / (mean(sqrt(|x|)))^2
    mean_sqrt_abs = float(np.mean(np.sqrt(np.abs(sig64))))
    features[f"{prefix}_clearance_factor"] = (
        peak_abs / (mean_sqrt_abs ** 2) if mean_sqrt_abs > 0 else np.nan
    )

    return features


# ────────────────────────────────────────────────────────────────────────────
# FFT computation (shared by freq-domain and fault-freq features)
# ────────────────────────────────────────────────────────────────────────────

def _compute_fft(
    sig: np.ndarray,
    fs: float = VIBRATION_SAMPLE_RATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute one-sided FFT magnitudes and frequency bins.

    Returns (magnitudes, freqs), both as float64 arrays.
    Returns empty arrays if signal is too short.
    """
    N = len(sig)
    if N < 2:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # ── Phase 2: Envelope Analysis (Hilbert Transform) ──
    # Pad to fast length for speed, then truncate back
    fast_len = next_fast_len(N)
    analytic = hilbert(sig, N=fast_len)
    envelope = np.abs(analytic[:N])
    # Remove DC component
    envelope -= np.mean(envelope)

    fft_vals = rfft(envelope)
    magnitudes = np.abs(fft_vals).astype(np.float64)
    freqs = rfftfreq(N, d=1.0 / fs)
    return magnitudes, freqs


# ────────────────────────────────────────────────────────────────────────────
# Frequency-domain features
# ────────────────────────────────────────────────────────────────────────────

def _freq_domain_features(
    magnitudes: np.ndarray,
    freqs: np.ndarray,
    prefix: str,
) -> Dict[str, float]:
    """
    Compute frequency-domain features from pre-computed FFT magnitudes.

    Parameters
    ----------
    magnitudes : np.ndarray
        One-sided FFT magnitude spectrum (including DC at index 0).
    freqs : np.ndarray
        Corresponding frequency bins.
    prefix : str
        Feature name prefix (e.g. 'CH1').
    """
    features: Dict[str, float] = {}
    if len(magnitudes) < 2:
        return features

    # Remove DC component for spectral shape features
    magnitudes_no_dc = magnitudes[1:]
    freqs_no_dc = freqs[1:]

    total_energy = float(np.sum(magnitudes_no_dc ** 2))
    features[f"{prefix}_spectral_energy"] = total_energy

    if total_energy == 0:
        features[f"{prefix}_spectral_centroid"] = np.nan
        features[f"{prefix}_spectral_bandwidth"] = np.nan
        features[f"{prefix}_spectral_rolloff"] = np.nan
        features[f"{prefix}_dominant_freq"] = np.nan
        features[f"{prefix}_dominant_mag"] = np.nan
        return features

    # Spectral centroid
    mag_sum = float(np.sum(magnitudes_no_dc))
    centroid = float(np.sum(freqs_no_dc * magnitudes_no_dc) / mag_sum) if mag_sum > 0 else 0.0
    features[f"{prefix}_spectral_centroid"] = centroid

    # Spectral bandwidth (std around centroid)
    bw = float(
        np.sqrt(np.sum(magnitudes_no_dc * (freqs_no_dc - centroid) ** 2) / mag_sum)
    ) if mag_sum > 0 else 0.0
    features[f"{prefix}_spectral_bandwidth"] = bw

    # Spectral rolloff (85 %)
    cumsum = np.cumsum(magnitudes_no_dc)
    rolloff_idx = int(np.searchsorted(cumsum, 0.85 * cumsum[-1]))
    rolloff_idx = min(rolloff_idx, len(freqs_no_dc) - 1)
    features[f"{prefix}_spectral_rolloff"] = float(freqs_no_dc[rolloff_idx])

    # Dominant frequency & magnitude
    peak_idx = int(np.argmax(magnitudes_no_dc))
    features[f"{prefix}_dominant_freq"] = float(freqs_no_dc[peak_idx])
    features[f"{prefix}_dominant_mag"] = float(magnitudes_no_dc[peak_idx])

    return features


# ────────────────────────────────────────────────────────────────────────────
# Bearing fault-frequency band energy
# ────────────────────────────────────────────────────────────────────────────

def _band_energy(
    magnitudes: np.ndarray,
    freqs: np.ndarray,
    centre: float,
    bandwidth: float,
) -> float:
    """Sum of squared magnitudes in *centre ± bandwidth*."""
    mask = (freqs >= centre - bandwidth) & (freqs <= centre + bandwidth)
    return float(np.sum(magnitudes[mask] ** 2))


def _fault_freq_features(
    magnitudes: np.ndarray,
    freqs: np.ndarray,
    prefix: str,
    rpm: float,
    cfg: PipelineConfig,
) -> Dict[str, float]:
    """
    Band energy around bearing fault frequencies corrected for actual RPM.

    Parameters
    ----------
    magnitudes : np.ndarray
        Pre-computed FFT magnitude spectrum.
    freqs : np.ndarray
        Corresponding frequency bins.
    prefix : str
        Feature name prefix.
    rpm : float
        Mean RPM for this acquisition window.
    cfg : PipelineConfig
        Pipeline configuration (for fault_harmonics, fault_band_hz).
    """
    features: Dict[str, float] = {}
    if len(magnitudes) < 2:
        return features

    rpm_ratio = rpm / 1000.0 if rpm and rpm > 0 else 1.0

    for fault_name, ref_freq in FAULT_FREQS_AT_1000RPM.items():
        actual_freq = ref_freq * rpm_ratio
        for harmonic in cfg.fault_harmonics:
            centre = actual_freq * harmonic
            be = _band_energy(magnitudes, freqs, centre, cfg.fault_band_hz)
            tag = f"{prefix}_{fault_name}_{harmonic}x"
            features[tag] = be

    return features


# ────────────────────────────────────────────────────────────────────────────
# Auxiliary-channel features (from operation CSV window)
# ────────────────────────────────────────────────────────────────────────────

def _aux_channel_features(arr: np.ndarray, prefix: str) -> Dict[str, float]:
    """Summary statistics for a slow-sampled auxiliary channel window."""
    features: Dict[str, float] = {}
    N = len(arr)
    if N == 0:
        return features

    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return features

    features[f"{prefix}_mean"] = float(np.mean(valid))
    features[f"{prefix}_std"] = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
    features[f"{prefix}_min"] = float(np.min(valid))
    features[f"{prefix}_max"] = float(np.max(valid))
    features[f"{prefix}_last"] = float(valid[-1])

    # Slope via linear regression
    if len(valid) >= 2:
        t = np.arange(len(valid), dtype=np.float64)
        slope, _, _, _, _ = sp_stats.linregress(t, valid.astype(np.float64))
        features[f"{prefix}_slope"] = float(slope)
    else:
        features[f"{prefix}_slope"] = np.nan

    return features


# ────────────────────────────────────────────────────────────────────────────
# Public API: extract features for one record
# ────────────────────────────────────────────────────────────────────────────

def extract_record_features(
    record: TdmsRecord,
    run: BearingRun,
    cfg: PipelineConfig,
    sample_index: int = 0,
    total_samples: int = 1,
    rpm_predictor = None,
    temp_classifier = None,
) -> Dict[str, float]:
    """
    Extract a complete feature vector from a single :class:`TdmsRecord`.

    Parameters
    ----------
    record : TdmsRecord
        Parsed TDMS data for one acquisition.
    run : BearingRun
        Parent run (provides operation CSV data).
    cfg : PipelineConfig
        Pipeline configuration.
    sample_index : int
        0-based index of this record within its run.
    total_samples : int
        Total number of records in the run (for normalised index).

    Returns
    -------
    dict
        Feature name → value mapping.
    """
    feats: Dict[str, float] = {}

    # ── Vibration channels: Base Features & FFT Caching ────────────────────
    fft_cache = {}
    for ch_key in VIBRATION_CHANNELS:
        prefix = ch_key
        arr = record.vibration.get(ch_key)
        if arr is None or len(arr) == 0:
            logger.warning("No data for %s in %s – skipping.", ch_key, record.filepath)
            continue

        # Time-domain features
        feats.update(_time_domain_features(arr, prefix))

        # Compute FFT and freq-domain features
        magnitudes, freqs = _compute_fft(arr, fs=VIBRATION_SAMPLE_RATE)
        feats.update(_freq_domain_features(magnitudes, freqs, prefix))
        fft_cache[ch_key] = (magnitudes, freqs)

    # ── Determine RPM ──────────────────────────────────────────────────────
    rpm = 1000.0  # default fallback
    if run.operation is not None:
        rpm = get_mean_rpm_for_file(run.operation, sample_index)
    elif rpm_predictor is not None:
        import pandas as pd
        # Prepare row for rpm predictor
        expected_cols = getattr(rpm_predictor, "feature_names_in_", None)
        if expected_cols is not None:
            row_dict = {k: v for k, v in feats.items() if k in expected_cols}
            df_row = pd.DataFrame([row_dict])
            for col in expected_cols:
                if col not in df_row.columns:
                    df_row[col] = 0.0
            pred_rpm = float(rpm_predictor.predict(df_row[expected_cols])[0])
            rpm = max(700.0, min(950.0, pred_rpm))  # Clip to realistic range

    # ── Fault Frequencies (RPM dependent) ──────────────────────────────────
    for ch_key, (magnitudes, freqs) in fft_cache.items():
        feats.update(_fault_freq_features(magnitudes, freqs, ch_key, rpm, cfg))

    # ── Auxiliary channels (from operation CSV) ────────────────────────────
    if run.operation is not None:
        op_window = get_operation_window(run.operation, sample_index)
        # Torque
        feats.update(_aux_channel_features(op_window["torque"], OP_FEATURE_NAMES["torque"]))
        # RPM
        feats.update(_aux_channel_features(op_window["rpm"], OP_FEATURE_NAMES["rpm"]))
        # Temperature front
        feats.update(_aux_channel_features(op_window["temp_front"], OP_FEATURE_NAMES["temp_front"]))
        # Temperature rear
        feats.update(_aux_channel_features(op_window["temp_rear"], OP_FEATURE_NAMES["temp_rear"]))

    # ── RPM value ──────────────────────────────────────────────────────────
    feats["rpm_mean"] = rpm

    # ── Temporal / run-level features ──────────────────────────────────────
    # NOTE: normalized_index intentionally removed — it always equals 1.0 at
    # the prediction point (last sample), creating a train/inference mismatch.
    # sample_index and elapsed_time_sec are absolute and leakage-free.
    feats["sample_index"] = float(sample_index)
    feats["elapsed_time_sec"] = float(sample_index * 600)
    
    if run.operation is not None:
        feats["is_high_temp_mode"] = 1.0 if feats.get("TC_SP_Front_C_mean", 0) > 75.0 else 0.0
    elif temp_classifier is not None:
        import pandas as pd
        expected_cols = getattr(temp_classifier, "feature_names_in_", None)
        if expected_cols is not None:
            row_dict = {k: v for k, v in feats.items() if k in expected_cols}
            df_row = pd.DataFrame([row_dict])
            for col in expected_cols:
                if col not in df_row.columns:
                    df_row[col] = 0.0
            pred_temp_mode = float(temp_classifier.predict(df_row[expected_cols])[0])
            feats["is_high_temp_mode"] = pred_temp_mode
        else:
            feats["is_high_temp_mode"] = 0.0
    else:
        feats["is_high_temp_mode"] = 0.0

    return feats
