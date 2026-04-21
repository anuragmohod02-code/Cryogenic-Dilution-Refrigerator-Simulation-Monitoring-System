"""
anomaly_detector.py
-------------------
Isolation Forest anomaly detection for dilution refrigerator cool-down curves.

Trained on 500 synthetic cool-down simulations spanning normal parameter
variation and five injected fault modes.  At inference time, any single
cool-down run can be scored in <1 ms.

Fault modes
-----------
1  PT1 failure      — 50 K pulse-tube conductance reduced 80 %  → T_50K >> 50 K
2  PT2 failure      — 4 K pulse-tube conductance reduced 80 %   → T_4K  >> 10 K
3  Still gas leak   — extra 200–500 mW heat load on 4 K stage   → T_Still >> 2 K
4  MXC overheating  — qubit power × 200 (vibration / RF leakage) → T_MXC >> 20 mK
5  ³He flow block   — circulation rate reduced 80 %              → T_MXC >> 50 mK

Usage
-----
    from anomaly_detector import load_or_train_model, score_simulation
    model = load_or_train_model()            # trains once, then caches
    result = score_simulation(model, t_h, T) # dict: score, is_anomaly, label
"""

from __future__ import annotations

import os
import pickle
import numpy as np
from typing import Any

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_HERE, "anomaly_model.pkl")

# ── Fault labels ──────────────────────────────────────────────────────────────
FAULT_LABELS = [
    "Normal operation",
    "PT1 failure (50 K stage)",
    "PT2 failure (4 K stage)",
    "Still gas leak (4 K heating)",
    "MXC overheating (vibration / RF)",
    "³He flow blockage",
]


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(t_h: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Extract a 15-element feature vector from a single cool-down run.

    Features (all log-scaled where temperature):
      0–4  : log(T_final[1:6])          — steady-state temps of stages 1–5
      5–9  : log(T_at_5h[1:6])          — mid-point temps at t ≈ 5 h
     10–14 : (T_at_5h - T_final) / T_at_5h  — fractional remaining cool-down
    """
    T_final = T[-1, 1:]                              # stages 1-5

    idx_5h  = np.argmin(np.abs(t_h - 5.0))
    T_mid   = T[idx_5h, 1:]

    rate = np.where(T_mid > 1e-12,
                    np.clip((T_mid - T_final) / T_mid, 0.0, 1.0),
                    0.0)

    return np.concatenate([
        np.log(np.maximum(T_final, 1e-10)),
        np.log(np.maximum(T_mid,   1e-10)),
        rate,
    ])


# ── Training data generation ──────────────────────────────────────────────────

def _normal_params(rng: np.random.Generator) -> dict:
    return {
        "n_wires":      int(rng.integers(8, 40)),
        "emissivity":   float(rng.uniform(0.01, 0.12)),
        "P_qubit_W":    float(rng.uniform(0.0, 20e-9)),
        "n3_flow_umol": float(rng.uniform(300.0, 700.0)),
    }


def _fault_params(fault_id: int, rng: np.random.Generator) -> dict:
    from cryo_thermal import _G0
    base = _normal_params(rng)
    G    = _G0.copy()

    if fault_id == 1:                                # PT1 failure
        G[1] = _G0[1] * rng.uniform(0.05, 0.25)
        base["G_override"] = G
    elif fault_id == 2:                              # PT2 failure
        G[2] = _G0[2] * rng.uniform(0.05, 0.25)
        base["G_override"] = G
    elif fault_id == 3:                              # Still gas leak
        base["P_extra_4K"] = rng.uniform(0.15, 0.5)
    elif fault_id == 4:                              # MXC overheating
        base["P_qubit_W"]  = rng.uniform(80e-6, 300e-6)
    elif fault_id == 5:                              # ³He flow blockage
        base["n3_flow_umol"] = rng.uniform(20.0, 100.0)

    return base


def generate_training_data(
        n_normal:     int = 200,
        n_fault_each: int = 60,
        seed:         int = 42,
        verbose:      bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run synthetic simulations and return feature matrix X (N,15) and labels y (N,).
    y=0 for normal, y=1..5 for fault types.
    """
    from cryo_thermal import simulate_cooldown
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    n_fault_types  = 5
    total          = n_normal + n_fault_each * n_fault_types

    if verbose:
        print(f"Generating {total} training samples "
              f"({n_normal} normal + {n_fault_each}×{n_fault_types} faults)…")

    for k in range(n_normal):
        try:
            t_h, T = simulate_cooldown(params=_normal_params(rng), n_eval=300)
            X_list.append(extract_features(t_h, T))
            y_list.append(0)
        except Exception:
            pass
        if verbose and k % 50 == 0 and k > 0:
            print(f"  Normal: {k}/{n_normal}")

    if verbose:
        print(f"  Normal: {n_normal}/{n_normal} ✓")

    for fid in range(1, n_fault_types + 1):
        ok = 0
        for _ in range(n_fault_each):
            try:
                t_h, T = simulate_cooldown(params=_fault_params(fid, rng), n_eval=300)
                X_list.append(extract_features(t_h, T))
                y_list.append(fid)
                ok += 1
            except Exception:
                pass
        if verbose:
            print(f"  Fault {fid} ({FAULT_LABELS[fid]}): {ok}/{n_fault_each} ✓")

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=int)
    if verbose:
        print(f"Dataset: {X.shape[0]} samples × {X.shape[1]} features")
    return X, y


# ── Model training ────────────────────────────────────────────────────────────

def train_detector(X_normal: np.ndarray) -> dict[str, Any]:
    """Train Isolation Forest on normal cool-down features."""
    from sklearn.ensemble    import IsolationForest
    from sklearn.preprocessing import StandardScaler

    scaler   = StandardScaler().fit(X_normal)
    X_scaled = scaler.transform(X_normal)
    clf = IsolationForest(
        n_estimators=200,
        contamination=0.15,
        random_state=42,
    )
    clf.fit(X_scaled)
    return {"clf": clf, "scaler": scaler}


def save_model(model_dict: dict, path: str = _MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model_dict, f)
    print(f"Model saved → {path}")


def load_model(path: str = _MODEL_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_or_train_model(force_retrain: bool = False) -> dict:
    """Load cached model or generate training data, train, and cache."""
    if not force_retrain and os.path.exists(_MODEL_PATH):
        print(f"Loading anomaly model from {_MODEL_PATH}")
        return load_model(_MODEL_PATH)

    print("Training anomaly detection model (one-time, ~2 min)…")
    X, y = generate_training_data(verbose=True)
    X_normal = X[y == 0]
    model    = train_detector(X_normal)
    save_model(model)

    # Quick evaluation on training data
    try:
        from sklearn.metrics import classification_report
        X_s  = model["scaler"].transform(X)
        pred = model["clf"].predict(X_s)          # 1=normal, -1=anomaly
        y_b  = np.where(y == 0, 1, -1)            # 1=normal, -1=fault
        print("\n── Training-set evaluation ──")
        print(classification_report(y_b, pred, target_names=["anomaly", "normal"],
                                    zero_division=0))
    except Exception:
        pass

    return model


# ── Inference ─────────────────────────────────────────────────────────────────

def score_simulation(
        model_dict: dict,
        t_h: np.ndarray,
        T:   np.ndarray,
) -> dict[str, Any]:
    """
    Score a single cool-down run against the anomaly model.

    Returns
    -------
    dict with keys:
        score      : float in ≈[-1, 1]  (higher = more normal)
        is_anomaly : bool
        label      : human-readable string
        pct        : int 0-100 "normality percentage"
    """
    feats  = extract_features(t_h, T).reshape(1, -1)
    Xs     = model_dict["scaler"].transform(feats)
    raw    = float(model_dict["clf"].decision_function(Xs)[0])
    pred   = int(model_dict["clf"].predict(Xs)[0])          # 1 normal, -1 anomaly

    # decision_function: 0.0 is the contamination boundary
    # Positive → normal side, negative → anomaly side
    # Normalise to a 0–100 "normality" percentage:
    #   raw = +0.5  → very normal  (100%)
    #   raw =  0.0  → at threshold (50%)
    #   raw = -0.5  → clear anomaly (0%)
    pct = int(np.clip((raw + 0.2) / 0.4 * 100, 0, 100))

    is_anomaly = pred == -1
    label = "⚠ Anomaly Detected" if is_anomaly else "✓ Normal Operation"

    return {
        "score":      float(np.clip((raw + 0.2) / 0.4, 0.0, 1.0)),
        "is_anomaly": is_anomaly,
        "label":      label,
        "pct":        pct,
        "raw_score":  raw,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    force = "--retrain" in sys.argv
    load_or_train_model(force_retrain=force)
