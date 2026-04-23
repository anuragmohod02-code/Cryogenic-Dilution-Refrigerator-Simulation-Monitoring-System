# Cryogenic Dilution Refrigerator Simulation & Monitoring System

> **Physics-based 6-stage ODE thermal model + real-time Dash dashboard + 5-class ML fault classifier + PID feedback control + MXC parameter sweep + Bayesian fault posterior + cooldown predictor for a dilution refrigerator achieving 13 mK base temperature**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![MATLAB R2023+](https://img.shields.io/badge/MATLAB-R2023+-orange.svg)](https://mathworks.com)
[![Dash 4](https://img.shields.io/badge/Dash-4.1-green.svg)](https://dash.plotly.com)

---

## Key Results

| Stage | Simulated T | Target | Status |
|---|---|---|---|
| 300 K plate | 300 K | 300 K | ✓ |
| 50 K (PT1) | **43.4 K** | < 50 K | ✓ |
| 4 K (PT2) | **3.6 K** | < 4.5 K | ✓ |
| Still | **476 mK** | < 600 mK | ✓ |
| Cold plate | **33 mK** | < 50 mK | ✓ |
| MXC (mixing chamber) | **13.2 mK** | < 20 mK | ✓ |

Cool-down from 300 K to 13 mK simulated in **< 3 s** of compute time.

---

## Features

### Physics Model (`python/cryo_thermal.py`, `matlab/thermal_model.m`)
- 6-stage coupled ODE system (stiff solver: SciPy `Radau` / MATLAB `ode15s`)
- MXC cooling power: **P = ṅ₃ × L × T²** (Pobell, *Matter and Methods at Low
  Temperatures*, L = 84 J/mol·K²)
- ³He circulation rate slider (100–800 µmol/s) scales sub-K cooling power
  with first-principles formula
- **Warm-up simulation**: removes cooling power, drives all stages back to 300 K
  through boundary conduction
- Configurable heat loads: wiring conduction (24–96 lines), radiation emissivity,
  qubit dissipation (0–1 µW)

### Real-Time Dashboard (`python/dashboard.py`)
- **Plotly-native animated cool-down** — 40-frame client-side animation with
  built-in Play/Pause button and timeline scrubber (no server round-trips)
- 6 live temperature gauges with colour-coded status (green/amber/red)
- Sub-Kelvin detail plot (mK scale) with 15 mK target line
- Heat balance bar chart (cooling power vs heat load, µW, log scale)
- **Export CSV** button — downloads full simulation time series
- Anomaly score card updates after every run

### ML Anomaly Detection (`python/anomaly_detector.py`)
- **Isolation Forest** trained on 500 synthetic cool-down runs
  (200 normal + 60 × 5 fault modes)
- **15 features**: log(T_final[stages 1–5]), log(T_at_5h[1–5]),
  fractional cool-down rate per stage
- Five injected fault modes:
  1. PT1 stage failure (50 K conductance −80 %)
  2. PT2 stage failure (4 K conductance −80 %)
  3. Still gas leak (+0.2–0.5 W on 4 K stage)
  4. MXC overheating (qubit power × 200, vibration / RF leakage)
  5. ³He flow blockage (circulation −80 %)
- Model cached in `python/anomaly_model.pkl`; loaded in background
  thread so dashboard starts instantly

### 5-Class Fault Classifier (`python/ml_deep_dive.py`) *(v2)*
- **Random Forest** (200 estimators) trained on 1 000 synthetic runs
  (200 per class) with per-stage temperature features (15 features)
- **Five fault classes**: Normal, Still heater fault, ³He flow fault,
  4 K overload, MXC vibration coupling
- **Test accuracy: 91%** (300-sample hold-out, macro F1 = 0.91);
  fault magnitudes are physically realistic with deliberate class overlap
  (e.g. Still heater fault at 0.5–4 mW vs Normal baseline ≤50 µW)
- Per-class F1: Normal 0.84, Still heater 0.90, ³He flow 0.98,
  4K overload 0.84, MXC vibration 0.98 — hardest classes reflect true
  diagnostic ambiguity at early fault onset
- Thermometry noise (0.5–2% RMS per stage, multiplicative) applied to
  every simulation trace to model RuO₂ sensor uncertainty
- Feature importance: MXC T_final and ΔT_MXC/ΔT_Still ratio are the top
  two discriminating features
- t-SNE projection shows clear cluster structure with realistic boundary overlap
- Outputs: `outputs/ml_roc_curves.png`, `ml_confusion_matrix.png`,
  `ml_feature_importance.png`, `ml_tsne.png`

### Bayesian Posterior Fault Diagnosis (`python/bayesian_fault_classifier.py`) *(v3)*
- Extends RF `predict_proba` output as Bayesian posterior P(fault | sensor data)
  under a flat prior — physically motivated: operator needs posterior, not point label
- **5 representative case studies**: posterior bar charts with true class highlighted,
  entropy annotation per case
- **Calibration / reliability diagram**: per-class OvR calibration curves confirm
  that RF posterior probabilities match empirical frequencies
- **Shannon entropy H** used to flag ambiguous fault states
  (H close to ln 5 = 1.61 nats → manual inspection warranted)
- Key results:
  - Test accuracy: 91.0%
  - Mean H = 0.397 nats (max = 1.609); 63.3% of predictions have H < 0.5 (confident)
  - ³He flow fault and MXC vibration are highest-confidence classes (H = 0.16–0.15)
  - Normal and 4K overload show highest diagnostic uncertainty (H = 0.72–0.63)
- Output: `outputs/14_bayesian_posterior.png`

### Cooldown Time Predictor (`python/cooldown_predictor.py`) *(v3)*
- Uses `simulate_cooldown` ODE to predict time-to-base (T_MXC < 20 mK) from
  a 300 K warm start as a function of operating parameters
- **4-panel figure**:
  1. Full 6-stage cooldown trajectory (log-scale)
  2. MXC approach to base temperature (mK, annotated)
  3. Heatmap: time-to-base vs (ṅ₃, P_qubit) — reveals critical operating envelope
  4. Dual-axis sensitivity: T_MXC at t=10 h vs each parameter independently
- **Key findings** (nominal: ṅ₃ = 476 µmol/s, P_qubit = 5 µW):
  - T_MXC at t=12 h = **13.22 mK** (below 20 mK target) ✔
  - P_qubit ≥ 20 µW prevents reaching base T within 14 h at any flow rate
  - Reducing flow to 250 µmol/s raises steady-state T_MXC to 19.8 mK (borderline)
  - At 200 µW qubit dissipation: T_MXC stabilises at 70 mK (5.3× above target)
- Output: `outputs/15_cooldown_predictor.png`

### PID Feedback Controller (`python/pid_controller.py`) *(v2)*
- **Digital PID** with anti-windup clamping; ZOH at 1 Hz matching a
  realistic embedded controller
- Tuning: Kp = 50, Ki = 0.05, Kd = 200 (Still heater → MXC setpoint)
- **Setpoint step** 15 mK → 25 mK → 15 mK: settles within ~8 min, zero
  steady-state error
- **Disturbance rejection**: 3 mW heat pulse on MXC (300 s) — recovered
  in < 5 min with < 2 mK overshoot
- **Kp sweep** (10 / 50 / 150 / 400): overdamped → critically damped →
  underdamped response clearly visible
- Outputs: `outputs/pid_step_response.png`, `pid_disturbance.png`,
  `pid_kp_sweep.png`

### MXC Temperature Parameter Sweep (`python/mxc_temp_sweep.py`) *(v2)*
- Sweeps ³He circulation rate ṅ₃ over **50–700 µmol/s** (27 points),
  extracting steady-state temperature from the 6-stage ODE
- Overlays **Pobell T² analytical prediction**:
  $T_{\text{MXC}} = \sqrt{Q_{\text{load}} / (84 \cdot \dot{n}_3)}$
- Log-log plot confirms slope −½ scaling over full range
- Family-of-curves for Q_load = 0, 1, 5, 10, 20 µW heat loads
- At nominal 476 µmol/s: T_MXC,sim = 12.8 mK vs Pobell = 7.1 mK
  (factor ~1.8 offset from additional still/cold-plate heat leaks)
- Outputs: `outputs/mxc_temp_vs_n3flow.png`, `stage_temps_vs_n3flow.png`,
  `mxc_family_curves.png`

### MATLAB (`matlab/`)
- `thermal_model.m` — same ODE system implemented in MATLAB (`ode15s`)
- `cooling_power.m` — Pobell T² cooling power per stage
- `run_analysis.m`, `sensitivity_analysis.m`, `compare_stages.m`

### Jupyter Notebooks (`notebooks/`)
| Notebook | Content |
|---|---|
| `01_Physics_Derivation.ipynb` | ODE derivation, Pobell formula, heat-flow diagram |
| `02_Steady_State_Analysis.ipynb` | Parameter sweeps, sensitivity analysis |
| `03_Transient_Analysis.ipynb` | Time-domain cool-down, multi-stage dynamics |
| `04_ML_Anomaly_Detection.ipynb` | Isolation Forest training, feature importance, ROC |

---

## Quick Start

```bash
# 1. Install Python dependencies
pip install -r requirements_dash.txt

# 2. (First time only) train the anomaly model — ~2 min
python python/anomaly_detector.py

# 3. Run the pipeline (saves CSV + PNG)
python python/run_pipeline.py

# 4. Launch dashboard
python python/dashboard.py
# → http://localhost:8050
```

---

## Project Structure

```
Project3_CryoThermal/
├── python/
│   ├── cryo_thermal.py        # 6-stage ODE model (simulate_cooldown, simulate_warmup)
│   ├── anomaly_detector.py    # Isolation Forest fault detection
│   ├── anomaly_model.pkl      # Cached trained model (500 training samples)
│   ├── dashboard.py           # Plotly Dash dashboard v2
│   ├── ml_deep_dive.py        # 5-class RF fault classifier (v2)
│   ├── pid_controller.py      # Digital PID — Still heater → MXC setpoint (v2)
│   ├── mxc_temp_sweep.py      # MXC T vs ṅ₃ sweep + Pobell T² overlay (v2)   ├── bayesian_fault_classifier.py  # Bayesian posterior fault diagnosis (v3)
   ├── cooldown_predictor.py  # Time-to-base heatmap + sensitivity analysis (v3)│   └── run_pipeline.py        # CLI entry point
├── matlab/
│   ├── thermal_model.m        # MATLAB ODE implementation
│   ├── cooling_power.m        # Pobell T² cooling power
│   ├── sensitivity_analysis.m
│   └── compare_stages.m
├── notebooks/
│   ├── 01_Physics_Derivation.ipynb
│   ├── 02_Steady_State_Analysis.ipynb
│   ├── 03_Transient_Analysis.ipynb
│   └── 04_ML_Anomaly_Detection.ipynb
├── outputs/
│   ├── stage_temperatures.csv
│   ├── cooldown_curve.png
│   ├── 14_bayesian_posterior.png   # Bayesian fault posteriors + calibration (v3)
│   └── 15_cooldown_predictor.png  # Cooldown time heatmap + sensitivity (v3)
├── requirements_dash.txt
└── README.md
```

---

## References

1. Pobell F., *Matter and Methods at Low Temperatures*, 3rd ed., Springer (2007) — ch. 6 dilution refrigeration
2. Pedregosa et al., JMLR 12 (2011) — scikit-learn
3. Niculescu-Mizil & Caruana, *Predicting Good Probabilities with Supervised Learning*, ICML 2005 — classifier calibration
4. van der Maaten & Hinton, JMLR 9 (2008) — t-SNE

---

## Physics Reference

The mixing-chamber cooling power follows the Pobell relation:

$$P_{\text{MXC}} = \dot{n}_3 \cdot L \cdot T^2$$

where $L = 84\ \text{J/(mol·K}^2\text{)}$ and $\dot{n}_3$ is the ³He molar
circulation rate.  At the nominal 476 µmol/s:

$$A_{\text{MXC}} = 476 \times 10^{-6}\ \frac{\text{mol}}{\text{s}} \times 84\ \frac{\text{J}}{\text{mol·K}^2} = 0.040\ \frac{\text{W}}{\text{K}^2}$$

> Pobell, F. *Matter and Methods at Low Temperatures*, 3rd ed., Springer (2007).

---

## Tech Stack

`Python 3.12` · `SciPy (Radau ODE)` · `NumPy` · `Plotly / Dash 4` ·
`dash-bootstrap-components` · `scikit-learn (Isolation Forest + Random Forest)` ·
`MATLAB R2023+` · `Jupyter`
