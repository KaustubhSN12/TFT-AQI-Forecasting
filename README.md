# 🛰️ Temporal Fusion Transformer Based AQI Forecasting

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

**Temporal Fusion Transformer v3 · Autoregressive Multi-Quantile Prediction · 10 Indian Cities**

[🚀 Live Demo](#deployment) · [📄 Research Paper](#research-paper) · [📊 Results](#results) · [🛠️ Installation](#installation)

</div>

---

## 📌 Overview

This repository presents an end-to-end deep learning pipeline for **24-hour probabilistic Air Quality Index (AQI) forecasting** across 10 major Indian cities. The model leverages the **Temporal Fusion Transformer (TFT)** architecture with an autoregressive decoder, achieving state-of-the-art performance with **R² = 0.85** and **RMSE = 3.20**.

> *"The most impactful finding: replacing StandardScaler with RobustScaler alone reduced RMSE by 43.9% — demonstrating that data pipeline decisions outweigh architectural complexity."*

---

## ✨ Key Highlights

```
┌─────────────────────────────────────────────────────────────────┐
│                    TFT-v3 ACHIEVEMENTS                          │
├─────────────────┬───────────────────────────────────────────────┤
│  R²             │  0.85  — explains 85.84% of AQI variance    │
│  RMSE           │  3.20    — vs 91.82 baseline (↓ 96.5%)        │
│  MAE            │  2.55    — vs 75.46 baseline (↓ 96.6%)        │
│  MAPE           │  5.3%    — vs 791.58% baseline (↓ 99.3%)      │
│  PI Coverage    │  ~80%    — perfectly calibrated intervals      │
│  Cities         │  10 Indian metro cities                        │
│  Horizon        │  24 hours ahead                               │
│  Quantiles      │  Q10 / Q50 / Q90                              │
└─────────────────┴───────────────────────────────────────────────┘
```

---

## 🏙️ Cities Covered

```
╔══════════════════════════════════════════════════════════════╗
║                    GEOGRAPHIC COVERAGE                       ║
╠══════════════════════╦═══════════════╦════════════════════════╣
║  CITY                ║  STATION      ║  REGION                ║
╠══════════════════════╬═══════════════╬════════════════════════╣
║  🏛️  Delhi           ║  Alipur       ║  Indo-Gangetic Plain   ║
║  🏙️  Noida           ║  Sector-1     ║  Indo-Gangetic Plain   ║
║  🏗️  Ghaziabad       ║  Indirapuram  ║  Indo-Gangetic Plain   ║
║  🕌  Lucknow         ║  Lalbagh      ║  Indo-Gangetic Plain   ║
║  🎓  Patna           ║  Muradpur     ║  Indo-Gangetic Plain   ║
║  🌊  Mumbai          ║  Bandra       ║  West Coast            ║
║  🎨  Kolkata         ║  Bidhannagar  ║  Eastern India         ║
║  ⛪  Chennai         ║  Manali       ║  East Coast            ║
║  💊  Hyderabad       ║  Central Univ ║  Deccan Plateau        ║
║  🌿  Bengaluru       ║  Bapuji Nagar ║  Deccan Plateau        ║
╚══════════════════════╩═══════════════╩════════════════════════╝
```

---

## 🏗️ Model Architecture

```
                    TFT-v3 ARCHITECTURE
                    ════════════════════

INPUT (72h × 18 features)
         │
         ▼
┌─────────────────────┐
│   Input Projection  │  Linear → LayerNorm → ELU → Dropout
│   (18 → 128 dims)   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Variable Selection │  Dynamically weights all 18 input features
│  Network (VSN)      │  per timestep — fully interpretable
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   2-Layer LSTM      │  Sequential hidden representations
│   Encoder           │  across 72-hour lookback window
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Gated Residual     │  Adaptive non-linear filtering
│  Network (GRN)      │  with skip connections
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Multi-Head         │  4 attention heads
│  Self-Attention     │  Long-range temporal dependencies
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────────────────┐
│           AUTOREGRESSIVE DECODER                │
│                                                 │
│  Step 1 → Q10, Q50, Q90 ──┐                    │
│  Step 2 → Q10, Q50, Q90 ◄─┘─┐                  │
│  Step 3 → Q10, Q50, Q90 ◄───┘─┐                │
│  ...                           │                │
│  Step 24→ Q10, Q50, Q90 ◄─────┘                │
│                                                 │
│  Each step conditions on previous prediction    │
└─────────────────────────────────────────────────┘
          │
          ▼
OUTPUT: 24h × [Q10, Q50, Q90]
```

---

## 📊 Results

### Model Version Progression

```
RMSE IMPROVEMENT ACROSS VERSIONS
══════════════════════════════════════════════════════

v1 Baseline  ████████████████████████████████████  91.82
             StandardScaler · MSE Loss · No early stopping

v2 Improved  ████████████████████  51.51  (↓ 43.9%)
             RobustScaler · Quantile Loss · GRN Block

v3 Optimized ██  3.20  (↓ 96.5% from v1)
             Autoregressive Decoder · Teacher Forcing · FP16

══════════════════════════════════════════════════════
```

### Multi-Model Benchmark

```
MODEL COMPARISON (same dataset, same evaluation conditions)
═══════════════════════════════════════════════════════════════

                    MAE      RMSE      MAPE      R²
                    ───      ────      ────      ──
🥇 TFT-v3 (Ours)   2.55     3.20      5.3%    0.85  ◄ BEST
🥈 XGBoost         54.41    65.48    260.7%   0.3275
🥉 GRU             59.91    76.99    240.8%   0.0701
   LSTM            63.53    81.50    204.8%  -0.0418

TFT-v3 outperforms XGBoost by 95.3% on MAE and 95.1% on RMSE
═══════════════════════════════════════════════════════════════
```

### Statistical Significance

| Test | Result |
|------|--------|
| Diebold-Mariano Test | p-value < 0.05 ✅ |
| Bootstrap CI (2000 iter, 95%) | Excludes zero ✅ |
| Cohen's d Effect Size | Medium to Large ✅ |

---

## 🔬 Synthetic Data Generation

```
RAW DATA PROBLEM
════════════════
Real observations : 1,565  (Jan 3–7, 2020 only)
Missing rate      : 99.7%  when expanded to full hourly grid
Usable for DL?    : ❌ IMPOSSIBLE

GPR SOLUTION PIPELINE
══════════════════════
Step 1 → Complete hourly grid (8,784h × 199 stations = 1,748,016 rows)
Step 2 → Feature engineering on real observations
Step 3 → GPR fitting per station using RBF kernel (l=24h) + White noise
Step 4 → Synthetic generation: ŷ(t) = clip(μ(t) + ε(t), 0, 500)
Step 5 → AQI lag feature construction (1h, 6h, 24h)

RESULT
══════
Real Data    →  1,565 rows  |  AQI mean: 188.4  |  AQI std: 98.9
Synthetic    →  1,748,016   |  AQI mean: 117.6  |  AQI std: 102.1
Final Model  →  87,840      |  10 cities, 1 station each
```

---

## 🎛️ Feature Engineering

| Category | Features | Count |
|----------|----------|-------|
| **Pollutants** | PM2.5, PM10, NO2, SO2, CO, Ozone, NH3 | 7 |
| **AQI Lags** | AQI_lag_1h, AQI_lag_6h, AQI_lag_24h | 3 |
| **Cyclic Time** | hour_sin, hour_cos, month_sin, month_cos | 4 |
| **Seasonal Flags** | is_winter, is_summer, is_monsoon | 3 |
| **Target** | AQI | 1 |
| **Total** | | **18** |

---

## ⚙️ Training Configuration

```python
CONFIG = {
    # Architecture
    "hidden_dim"     : 128,
    "attention_heads": 4,
    "lstm_layers"    : 2,
    "dropout"        : 0.2,
    "lookback"       : 72,   # hours
    "horizon"        : 24,   # hours
    "input_features" : 18,

    # Training
    "optimizer"      : "AdamW",
    "learning_rate"  : 2e-3,
    "weight_decay"   : 5e-4,
    "lr_schedule"    : "Warmup(5 epochs) + Cosine Annealing",
    "batch_size"     : 256,
    "early_stopping" : 12,   # patience
    "grad_clip"      : 0.5,  # max norm
    "teacher_forcing": "0.5 → 0.0 by epoch 14",
    "precision"      : "Mixed FP16",
    "loss_function"  : "Quantile Loss Q10/Q50/Q90",
    "scaler"         : "RobustScaler",
}
```

---

## 📁 Repository Structure

```
TFT-AQI-Forecasting/
│
├── 📄 streamlit_white_UI_v2.py     ← Main Streamlit application
├── 📄 requirements.txt              ← Python dependencies
├── 📄 packages.txt                  ← System dependencies
├── 📄 README.md                     ← This file
│
├── 📂 models/
│   ├── best_tft_v3_model_20.1.pth  ← Trained TFT-v3 checkpoint
│   └── scalers_v3.pkl              ← Fitted RobustScaler objects
│
├── 📂 dataset/
│   └── val_tft_realistic_continuous.csv  ← Validation dataset
│
└── 📂 notebooks/                    ← Training and analysis notebooks
```

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/KaustubhSN12/TFT-AQI-Forecasting.git
cd TFT-AQI-Forecasting
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run streamlit_white_UI_v2.py
```

---

## 🌐 Deployment

The application is deployed on **Streamlit Community Cloud**.

**Features:**
- 🏙️ City selector — 10 Indian cities
- 📈 Historical AQI trend chart
- 🔮 24-hour probabilistic forecast
- 📊 Q10 / Q50 / Q90 quantile visualization
- 🌡️ AQI category color-coded labels
- 📋 Per-hour forecast table
- 🕐 Diurnal pattern analysis
- 🗺️ AQI intensity heatmap

---

## 🛠️ Tech Stack

```
┌─────────────────────────────────────────────────────┐
│                    TECH STACK                       │
├─────────────────┬───────────────────────────────────┤
│  Language       │  Python 3.9+                      │
│  Deep Learning  │  PyTorch                          │
│  ML             │  Scikit-learn, XGBoost            │
│  Data           │  Pandas, NumPy                    │
│  Visualization  │  Plotly, Matplotlib               │
│  Deployment     │  Streamlit                        │
│  Version Ctrl   │  Git, GitHub                      │
│  Hardware       │  NVIDIA GPU, CUDA, Mixed FP16     │
└─────────────────┴───────────────────────────────────┘
```

---

## 🔑 Key Lessons Learned

> **1. Preprocessing > Architecture**
> Replacing StandardScaler with RobustScaler reduced RMSE by 43.9% alone — more than any architectural change.

> **2. Loss Function Matters**
> MSE caused mean collapse — flat line predictions. Quantile Loss solved it and added uncertainty quantification for free.

> **3. Autoregressive Decoding Solves Horizon Degradation**
> RMSE grew from 26 at h+1 to 58 at h+24 with direct decoding. Autoregressive decoder flattened this completely.

> **4. Synthetic Data is Viable**
> GPR-generated data preserved AQI standard deviation (102.1 vs 98.9 real) while enabling full-year training.

---

## 📄 Research Paper

**Title:** Temporal Fusion Transformer Based Air Quality Index Forecasting Using Multivariate Time Series Data

**Author:** Kaustubh S. Narayankar

**Institution:** Department of Data Science, S.I.E.S College of Arts, Science and Commerce (Autonomous), Mumbai, Maharashtra, India

**Status:** Prepared for journal submission — 2026

**Target Journals:**
- Atmospheric Pollution Research
- Sustainable Cities and Society
- Environmental Modelling and Software

---

## 📚 References

```
[1] Lim et al. (2021) — Temporal Fusion Transformers for interpretable
    multi-horizon time series forecasting. Int. Journal of Forecasting.

[2] Diebold & Mariano (1995) — Comparing predictive accuracy.
    Journal of Business & Economic Statistics.

[3] Central Pollution Control Board (CPCB) — Real-Time AQI Data
    https://www.data.gov.in/catalog/real-time-air-quality-index
```

---

## 🏆 Citation

If you use this work in your research, please cite:

```bibtex
@misc{narayankar2026tft,
  author    = {Kaustubh S. Narayankar},
  title     = {Temporal Fusion Transformer Based Air Quality Index
               Forecasting Using Multivariate Time Series Data},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/KaustubhSN12/TFT-AQI-Forecasting}
}
```

---

## 👨‍💻 Author

**Kaustubh S. Narayankar**

Department of Data Science
S.I.E.S College of Arts, Science and Commerce (Autonomous)
Mumbai, Maharashtra, India

[![GitHub](https://img.shields.io/badge/GitHub-KaustubhSN12-181717?style=for-the-badge&logo=github)](https://github.com/KaustubhSN12)

---

## 📜 License

This project is licensed under the MIT License.

---

<div align="center">

**⭐ Star this repository if you found it useful ⭐**

*Built with ❤️ for cleaner air and smarter cities*

</div>
