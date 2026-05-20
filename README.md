# KSPHM-KIMM 2026 PHM Data Challenge — Bearing RUL Prediction

A complete pipeline for predicting the **Remaining Useful Life (RUL)** of bearings using feature engineering and ensemble regression, built for the KSPHM-KIMM PHM Data Challenge 2026.

---

## Problem Definition

Given run-to-failure vibration data from bearings on a test rig, predict the RUL (in seconds) for validation bearings whose data does not reach failure.

**Vibration channels (25.6 kHz, stored in TDMS files):**
| Channel | Description               |
|---------|---------------------------|
| CH1     | Front Vertical Vibration  |
| CH2     | Front Axial Vibration     |
| CH3     | Rear Vertical Vibration   |
| CH4     | Rear Axial Vibration      |

**Auxiliary channels (0.1 Hz = 10 s intervals, stored in Operation CSVs):**
| Column            | Description           |
|-------------------|-----------------------|
| Time[sec]         | Timestamp in seconds  |
| Torque[Nm]        | Torque measurement    |
| Motor speed[rpm]  | Shaft RPM             |
| TC SP Front[℃]   | Front temperature     |
| TC SP Rear[℃]    | Rear temperature      |

**Acquisition cycle:** Every 10 minutes, 1 minute of data is recorded.

---

## Data Layout

```
<data_dir>/
  train/
    Train1_Operation.csv       # Auxiliary sensor data (0.1 Hz)
    Train1_Vibration/          # Vibration TDMS files
      000001.tdms
      000002.tdms
      ...
    Train2_Operation.csv
    Train2_Vibration/
    Train3_Operation.csv
    Train3_Vibration/
    Train4_Operation.csv
    Train4_Vibration/
  validation/                   # Same structure
    Validation1_Operation.csv
    Validation1_Vibration/
    ...
```

Each `*_Vibration/` folder is one bearing degradation run.
Each `.tdms` file is one acquisition point (sorted by filename).
Each `*_Operation.csv` contains the slow-sampled auxiliary channels for the corresponding run.

---

## Pipeline Overview

### 1. Feature Extraction (`src/features.py`)

For each TDMS file, a single feature vector is extracted (**156 features total**):

**Time-domain (per vibration channel × 4 channels = 56 features):**
mean, std, RMS, max, min, peak-to-peak, skewness, kurtosis, crest factor, impulse factor, shape factor, clearance factor, absolute mean, energy.

**Frequency-domain (per vibration channel × 4 channels = 24 features):**
spectral centroid, spectral bandwidth, spectral rolloff, dominant frequency, dominant magnitude, total spectral energy.

**Bearing fault-frequency band energy (per channel × 4 faults × 3 harmonics = 48 features):**
- Frequencies corrected by RPM from operation CSV: `f_actual = f_ref × RPM / 1000`
- Reference at 1000 RPM: BPFI=140 Hz, BPFO=93 Hz, BSF=78 Hz, Cage=6.7 Hz
- Band energy computed at 1×, 2×, and 3× harmonics (±5 Hz bandwidth)

**Auxiliary channels (from Operation CSV, per channel × 6 stats = 24 features):**
mean, std, min, max, last value, slope over the 1-minute window.

**Temporal features (3 features):**
sample_index, elapsed_time_sec, normalized_index, plus rpm_mean.

### 2. Label Generation (`src/dataset.py`)

For training runs (run-to-failure):
```
failure_time = (num_files - 1) × 600 + failure_offset_sec
RUL(i) = failure_time - i × 600
```
Default `failure_offset_sec = 300`.

### 3. Validation Strategy (`src/evaluate.py`)

**Leave-One-Bearing-Out CV:**
Hold out one complete training run, train on the remaining runs, evaluate on the held-out run. Repeat for each run.

**Partial-Run Validation:**
For each held-out run, simulate validation by cutting at 30%, 40%, 50%, 60%, 70%, 80%. Predict RUL at the cutoff point and compare with ground truth.

### 4. Models (`src/models.py`)

All wrapped in `sklearn.Pipeline` with `SimpleImputer` + `StandardScaler`:

| Model                         | Status             |
|-------------------------------|--------------------|
| RandomForestRegressor         | Always available   |
| ExtraTreesRegressor           | Always available   |
| GradientBoostingRegressor     | Always available   |
| HistGradientBoostingRegressor | Always available   |
| XGBRegressor                  | If xgboost installed |
| LGBMRegressor                 | If lightgbm installed |

### 5. Ensemble

- Train all available models.
- Compute per-model weights from CV scores.
- Final prediction = weighted average of individual predictions.

### 6. Calibration (`src/evaluate.py`)

Calibration factor search over [0.75, 1.05] to maximise the official score:
```
final_pred = raw_pred × calibration_factor
```

### 7. Official Scoring

```
Er = 100 × (ActRUL - PredRUL) / ActRUL

If Er ≤ 0:  A_RUL = exp(-ln(0.5) × Er / 20)
If Er > 0:  A_RUL = exp(+ln(0.5) × Er / 50)

Score = mean(A_RUL)
```

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
# Optional:
pip install xgboost lightgbm
```

### Training

```bash
python3 src/train.py --data_dir . --output_dir outputs --failure_offset_sec 300
```

**Outputs:**
- `outputs/models/` — Trained model files (`.joblib`) and `meta.json`
- `outputs/reports/train_features.csv` — Extracted training features
- `outputs/reports/cv_results.csv` — Cross-validation results
- `outputs/reports/partial_run_validation.csv` — Partial-run validation
- `outputs/reports/feature_importances.csv` — Feature importance rankings

### Prediction

```bash
python3 src/predict.py --data_dir . --output_dir outputs
```

**Outputs:**
- `outputs/predictions/validation_predictions.csv` — Full predictions with per-model details
- `outputs/predictions/team_validation.xlsx` — Submission file (Dataset, Predicted_RUL_sec)

---

## Project Structure

```
src/
  config.py        # All tunable parameters
  data_loader.py   # TDMS + Operation CSV loading & channel discovery
  features.py      # Time/frequency/fault-frequency feature extraction
  dataset.py       # Feature table construction & RUL labelling
  models.py        # Model catalogue & ensemble logic
  evaluate.py      # Scoring, LOBO CV, calibration search
  train.py         # Training entry point
  predict.py       # Prediction entry point
  utils.py         # Logging & natural sorting

outputs/
  models/          # Trained models & metadata
  predictions/     # Submission files
  reports/         # CSVs & analysis
```

---

## Required Libraries

| Package      | Purpose                          |
|--------------|----------------------------------|
| numpy        | Numerical computation            |
| pandas       | DataFrames                       |
| scipy        | FFT, statistics                  |
| scikit-learn | Models, pipelines, preprocessing |
| nptdms       | TDMS file reading                |
| openpyxl     | Excel output                     |
| joblib       | Model serialisation              |
| tqdm         | Progress bars                    |
| xgboost      | (optional) XGBoost regressor     |
| lightgbm     | (optional) LightGBM regressor    |

---

## Reproducibility

- All random seeds are controlled via `--seed` (default 42).
- Models are deterministic given the same seed.
- Feature extraction is purely numerical — no randomness.
