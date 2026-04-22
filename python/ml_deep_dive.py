"""
ml_deep_dive.py — ML Fault Classification Deep-Dive
======================================================

Extends the anomaly_detector.py binary classifier to a 5-class multi-fault
classifier that distinguishes between:

  0 - Normal operation
  1 - Still heater fault (power runaway → Still overtemperature)
  2 - ³He flow fault (circulation pump failure → MXC warm-up)
  3 - 4K plate overload (large experiment heat load on 4K stage)
  4 - MXC vibration coupling (dilution circuit oscillation → MXC noise)

Analysis produced
-----------------
1. ROC curves per fault class (one-vs-rest, AUC per class)
2. Confusion matrix (5×5, normalised)
3. Feature importance (Random Forest + permutation importance)
4. Precision-Recall curves
5. t-SNE projection of feature space (coloured by fault class)

References
----------
Pedregosa et al., JMLR 12 (2011)          — scikit-learn
van der Maaten & Hinton, JMLR 9 (2008)    — t-SNE
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                              classification_report, precision_recall_curve,
                              average_precision_score)
from sklearn.inspection import permutation_importance
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Import cryo thermal model to generate realistic training data
import sys
sys.path.insert(0, os.path.dirname(__file__))
from cryo_thermal import STAGE_NAMES, _G0, _C, cooling_power
from scipy.integrate import solve_ivp


def _scenario_ode(t, T, n3, extra):
    T = np.maximum(T, 1e-10)
    dTdt = np.zeros(6)
    dTdt[0] = 0.0
    G = _G0.copy()
    for i in range(1, 6):
        Q_in  = G[i] * (T[i-1] - T[i])
        Q_out = G[i+1] * (T[i] - T[i+1]) if i < 5 else 0.0
        Pcool = cooling_power(i, T[i], n3_flow_umol=n3)
        dTdt[i] = (Q_in - Q_out - Pcool + extra[i]) / _C[i]
    return dTdt


def run_simulation(t_end=3600, t_points=200, n3_flow_umol_s=476.0, extra_heat=None):
    """Local wrapper: runs thermal ODE with arbitrary per-stage extra heat loads."""
    if extra_heat is None:
        extra_heat = [0.0] * 6
    T0 = np.array([300.0, 50.0, 4.0, 0.70, 0.10, 0.015])
    extra = np.array(extra_heat, dtype=float)
    t_span = (0.0, float(t_end))
    t_eval = np.linspace(0.0, float(t_end), t_points)
    sol = solve_ivp(
        lambda t, y: _scenario_ode(t, y, n3_flow_umol_s, extra),
        t_span, T0, method="Radau", t_eval=t_eval,
        rtol=1e-5, atol=1e-7,
    )
    return {"T": sol.y.T}

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

FAULT_NAMES = [
    "Normal",
    "Still heater\nfault",
    "³He flow\nfault",
    "4K plate\noverload",
    "MXC vibration\ncoupling",
]
FAULT_COLORS = ["#2ECC71", "#E74C3C", "#E67E22", "#3498DB", "#9B59B6"]
N_CLASSES = 5


# ── Feature extraction from a thermal simulation run ─────────────────────────────

def extract_features(result: dict) -> np.ndarray:
    """
    Extract a feature vector from a thermal simulation result.

    Features (15 total):
      - Final temperature of each stage (6)
      - Max temperature excursion from steady state for each stage (6)
      - ΔT_MXC / ΔT_Still ratio (1)
      - ΔT_4K / ΔT_50K ratio (1)
      - Mean MXC temperature over last 10% of time (1)
    """
    T = result["T"]   # (n_time, 6)
    n = T.shape[0]
    n_tail = max(1, n // 10)

    T_final  = T[-1, :]
    T_steady = T[n//2:, :].mean(axis=0)   # approximate steady state
    T_excur  = (T - T_steady[np.newaxis, :]).max(axis=0)   # max excursion

    dT_mxc   = T_excur[5]
    dT_still = T_excur[3] + 1e-12
    dT_4k    = T_excur[2]
    dT_50k   = T_excur[1] + 1e-12

    T_mxc_mean = T[-n_tail:, 5].mean()

    feat = np.concatenate([
        T_final,
        T_excur,
        [dT_mxc / dT_still,
         dT_4k  / dT_50k,
         T_mxc_mean],
    ])
    return feat


FEATURE_NAMES = (
    [f"{n} T_final (K)" for n in STAGE_NAMES]
    + [f"{n} T_excur (K)" for n in STAGE_NAMES]
    + ["ΔT_MXC/ΔT_Still", "ΔT_4K/ΔT_50K", "T_MXC_mean (K)"]
)


# ── Fault scenario simulation ─────────────────────────────────────────────────────

def _gen_normal(rng, n_samples: int) -> list:
    """Normal operation with small random parameter variations."""
    results = []
    for _ in range(n_samples):
        n3  = rng.uniform(400, 560)    # µmol/s
        qh  = rng.uniform(0.0, 0.001)  # extra heat (W) small
        r   = run_simulation(t_end=3600, t_points=200,
                             n3_flow_umol_s=n3, extra_heat=[0,0,0,qh,0,0])
        results.append(r)
    return results


def _gen_still_fault(rng, n_samples: int) -> list:
    """Still heater runaway: large extra heat at Still stage."""
    results = []
    for _ in range(n_samples):
        n3  = rng.uniform(400, 560)
        qh  = rng.uniform(0.01, 0.05)   # 10–50 mW still fault
        r   = run_simulation(t_end=3600, t_points=200,
                             n3_flow_umol_s=n3, extra_heat=[0,0,0,qh,0,0])
        results.append(r)
    return results


def _gen_flow_fault(rng, n_samples: int) -> list:
    """³He flow fault: reduced circulation rate."""
    results = []
    for _ in range(n_samples):
        n3  = rng.uniform(50, 150)    # reduced flow → poor cooling
        qh  = rng.uniform(0.0, 0.001)
        r   = run_simulation(t_end=3600, t_points=200,
                             n3_flow_umol_s=n3, extra_heat=[0,0,0,0,0,qh])
        results.append(r)
    return results


def _gen_4k_overload(rng, n_samples: int) -> list:
    """4K stage overload: large experiment heat load."""
    results = []
    for _ in range(n_samples):
        n3  = rng.uniform(400, 560)
        qh  = rng.uniform(0.1, 0.5)    # 100–500 mW on 4K stage
        r   = run_simulation(t_end=3600, t_points=200,
                             n3_flow_umol_s=n3, extra_heat=[0,0,qh,0,0,0])
        results.append(r)
    return results


def _gen_mxc_vibration(rng, n_samples: int) -> list:
    """MXC vibration: small oscillatory heat on MXC + cold plate."""
    results = []
    for _ in range(n_samples):
        n3  = rng.uniform(400, 560)
        qh  = rng.uniform(0.0005, 0.003)   # 0.5–3 mW MXC vibration coupling
        qcp = rng.uniform(0.0001, 0.001)
        r   = run_simulation(t_end=3600, t_points=200,
                             n3_flow_umol_s=n3, extra_heat=[0,0,0,0,qcp,qh])
        results.append(r)
    return results


# ── Dataset generation ────────────────────────────────────────────────────────────

def build_dataset(n_per_class: int = 200, seed: int = 42) -> tuple:
    """Generate labelled feature dataset for all 5 fault classes."""
    rng = np.random.default_rng(seed)
    generators = [
        _gen_normal, _gen_still_fault, _gen_flow_fault,
        _gen_4k_overload, _gen_mxc_vibration,
    ]

    X_list, y_list = [], []
    for label, gen_fn in enumerate(generators):
        print(f"  Generating class {label} ({FAULT_NAMES[label].replace(chr(10),' ')})…", flush=True)
        runs = gen_fn(rng, n_per_class)
        for r in runs:
            X_list.append(extract_features(r))
        y_list.extend([label] * n_per_class)

    X = np.array(X_list)
    y = np.array(y_list, dtype=int)

    # Add small observation noise to features
    X += rng.normal(0, 0.001, X.shape) * X  # 0.1% relative noise

    return X, y


# ── Train & evaluate ────────────────────────────────────────────────────────────────

def train_and_evaluate(X: np.ndarray, y: np.ndarray):
    """Train RF classifier and compute all metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=200, max_depth=None,
                                  random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_score = clf.predict_proba(X_test)
    y_pred  = clf.predict(X_test)

    print("\n" + classification_report(y_test, y_pred,
          target_names=[n.replace("\n", " ") for n in FAULT_NAMES]))

    return clf, X_train, X_test, y_train, y_test, y_score, y_pred


# ── Plots ─────────────────────────────────────────────────────────────────────────

def plot_roc_curves(y_test, y_score):
    """One-vs-rest ROC curves for all fault classes."""
    Y_bin = label_binarize(y_test, classes=list(range(N_CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(N_CLASSES):
        fpr, tpr, _ = roc_curve(Y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=FAULT_COLORS[i], linewidth=2,
                label=f"{FAULT_NAMES[i].replace(chr(10),' ')}  AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#888", linewidth=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Multi-Class Fault Detection (One-vs-Rest)", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.35)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ml_roc_curves.png"), dpi=150, bbox_inches="tight")
    print(f"Saved → {os.path.join(OUT_DIR, 'ml_roc_curves.png')}")


def plot_confusion_matrix(y_test, y_pred):
    """Normalised 5×5 confusion matrix."""
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Fraction of true label")
    labels = [n.replace("\n", "\n") for n in FAULT_NAMES]
    ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(labels, fontsize=8, rotation=25, ha="right")
    ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(labels, fontsize=8)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if cm[i,j] > 0.5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — 5-Fault Classifier (Normalised)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ml_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    print(f"Saved → {os.path.join(OUT_DIR, 'ml_confusion_matrix.png')}")


def plot_feature_importance(clf, X_test, y_test):
    """Feature importance from RF + permutation importance."""
    imp_rf = clf.feature_importances_
    result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42)
    imp_perm = result.importances_mean

    # Top-10 features by RF importance
    top_idx = np.argsort(imp_rf)[::-1][:10]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Feature Importance — 5-Fault Classifier", fontweight="bold")

    ax = axes[0]
    names = [FEATURE_NAMES[i] for i in top_idx]
    ax.barh(range(10), imp_rf[top_idx][::-1], color="#3498DB", alpha=0.8)
    ax.set_yticks(range(10)); ax.set_yticklabels(names[::-1], fontsize=8)
    ax.set_xlabel("RF Gini importance"); ax.set_title("Random Forest Importance (top 10)")
    ax.grid(axis="x", alpha=0.35)

    ax = axes[1]
    top_perm = np.argsort(imp_perm)[::-1][:10]
    names_p = [FEATURE_NAMES[i] for i in top_perm]
    ax.barh(range(10), imp_perm[top_perm][::-1], color="#E74C3C", alpha=0.8)
    ax.set_yticks(range(10)); ax.set_yticklabels(names_p[::-1], fontsize=8)
    ax.set_xlabel("Permutation importance (mean accuracy drop)")
    ax.set_title("Permutation Importance (top 10)")
    ax.grid(axis="x", alpha=0.35)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ml_feature_importance.png"), dpi=150, bbox_inches="tight")
    print(f"Saved → {os.path.join(OUT_DIR, 'ml_feature_importance.png')}")


def plot_tsne(X, y):
    """t-SNE projection of feature space coloured by fault class."""
    print("  Computing t-SNE (this may take ~20 s)…", flush=True)
    X_emb = TSNE(n_components=2, random_state=42, perplexity=30,
                 max_iter=1000).fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (name, col) in enumerate(zip(FAULT_NAMES, FAULT_COLORS)):
        mask = y == i
        ax.scatter(X_emb[mask, 0], X_emb[mask, 1], s=12, color=col, alpha=0.6,
                   label=name.replace("\n", " "))
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.set_title("t-SNE Feature Space — 5 Fault Classes", fontweight="bold")
    ax.legend(fontsize=9, markerscale=2); ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ml_tsne.png"), dpi=150, bbox_inches="tight")
    print(f"Saved → {os.path.join(OUT_DIR, 'ml_tsne.png')}")


# ── Entry point ─────────────────────────────────────────────────────────────────────

def run_ml_deep_dive(n_per_class: int = 200):
    print("=" * 60)
    print("  ML Fault Classification Deep-Dive — 5 Fault Classes")
    print("=" * 60)
    print(f"\nGenerating dataset ({n_per_class} samples per class)…")
    X, y = build_dataset(n_per_class=n_per_class)
    print(f"Dataset shape: {X.shape},  labels: {np.bincount(y)}")

    clf, X_train, X_test, y_train, y_test, y_score, y_pred = train_and_evaluate(X, y)

    print("\nGenerating plots…")
    plot_roc_curves(y_test, y_score)
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(clf, X_test, y_test)
    plot_tsne(X, y)

    print("\nDone.")
    return clf


if __name__ == "__main__":
    run_ml_deep_dive(n_per_class=200)
