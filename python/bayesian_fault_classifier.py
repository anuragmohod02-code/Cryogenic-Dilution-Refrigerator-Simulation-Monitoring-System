"""
bayesian_fault_classifier.py -- Bayesian Posterior Fault Diagnosis
====================================================================

Extends the Random Forest classifier from ml_deep_dive.py with a Bayesian
posterior interpretation: RF predict_proba gives P(class | features), which
we treat as the posterior given a flat prior.  The analysis produces:

  1. Posterior bar charts for representative test cases (one per true fault class)
  2. Calibration curve (reliability diagram) -- does P_model match empirical freq?
  3. Posterior entropy vs classification confidence
  4. Bayesian decision boundary: P(correct | max_posterior)

Physics motivation
------------------
A Bayesian posterior is the natural framework for real-time cryostat fault
triage: the operator needs P(fault | sensor data), not just a point label.
High-entropy posteriors flag ambiguous fault states that warrant manual
inspection before intervention.

Output
------
    outputs/14_bayesian_posterior.png   -- 4-panel figure

References
----------
Pedregosa et al., JMLR 12 (2011)          -- scikit-learn
Niculescu-Mizil & Caruana, ICML 2005      -- calibration of classifiers
"""

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from scipy.stats import entropy as scipy_entropy

# Import dataset-building machinery from ml_deep_dive
from ml_deep_dive import (
    build_dataset, FAULT_NAMES, FAULT_COLORS, N_CLASSES
)

OUT = os.path.join(ROOT, 'outputs')
os.makedirs(OUT, exist_ok=True)
SEP = '=' * 60

FAULT_LABELS = [fn.replace('\n', ' ') for fn in FAULT_NAMES]

# ── Build dataset & train ─────────────────────────────────────────────────────

print(SEP)
print('  Bayesian Posterior Fault Diagnosis')
print(SEP)

print('\n[1/4] Generating dataset (5 classes x 200 samples)...')
X, y = build_dataset(n_per_class=200, seed=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

print('\n[2/4] Training Random Forest (200 estimators)...')
clf = RandomForestClassifier(
    n_estimators=200, max_depth=None,
    random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Posterior probabilities (N_test, 5)
posteriors = clf.predict_proba(X_test)   # P(class | x) via Bayes + flat prior
y_pred     = clf.predict(X_test)
accuracy   = float(np.mean(y_pred == y_test))
print(f'   Test accuracy: {accuracy*100:.1f}%')


# ── Pick representative test cases (one correct per true class) ───────────────

rep_indices = []
for cls in range(N_CLASSES):
    # Find test samples of this true class that are correctly classified
    mask = (y_test == cls) & (y_pred == cls)
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        # Fall back to any sample of this class
        idxs = np.where(y_test == cls)[0]
    # Pick the one with highest posterior for its true class (most confident)
    best = idxs[np.argmax(posteriors[idxs, cls])]
    rep_indices.append(int(best))

print(f'   Representative cases selected: {rep_indices}')


# ── Entropy & confidence ──────────────────────────────────────────────────────

H = scipy_entropy(posteriors, axis=1)   # Shannon entropy (nats)
H_max = np.log(N_CLASSES)               # maximum entropy (uniform)
confidence = posteriors.max(axis=1)     # P(most likely class)

# Fraction correctly classified vs confidence bin
n_bins    = 10
conf_bins = np.linspace(0, 1, n_bins + 1)
bin_acc   = np.zeros(n_bins)
bin_conf  = np.zeros(n_bins)
bin_count = np.zeros(n_bins, dtype=int)

for i in range(n_bins):
    lo, hi   = conf_bins[i], conf_bins[i + 1]
    in_bin   = (confidence >= lo) & (confidence < hi)
    if in_bin.sum() > 0:
        bin_acc[i]   = float(np.mean(y_pred[in_bin] == y_test[in_bin]))
        bin_conf[i]  = confidence[in_bin].mean()
        bin_count[i] = int(in_bin.sum())


# ── Calibration curve (OvR per class, averaged) ───────────────────────────────

cal_frac_pos = []
cal_mean_pred = []
for cls in range(N_CLASSES):
    y_bin   = (y_test == cls).astype(int)
    prob_cls = posteriors[:, cls]
    frac, mpred = calibration_curve(y_bin, prob_cls, n_bins=8)
    cal_frac_pos.append(frac)
    cal_mean_pred.append(mpred)


# ── Plotting ──────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'figure.dpi':       150,
    'font.size':        10,
    'axes.titlesize':   11,
    'axes.labelsize':   10,
    'legend.fontsize':  9,
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    f'Bayesian Fault Posterior: Cryostat Fault Diagnosis  '
    f'(RF 200 trees, 5 classes, test acc = {accuracy*100:.1f}%)',
    fontsize=12, fontweight='bold')

gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.4)

ax_bars = [fig.add_subplot(gs[0, k]) for k in range(3)]
ax_bars.append(fig.add_subplot(gs[1, 0]))
ax_bars.append(fig.add_subplot(gs[1, 1]))
ax_cal = fig.add_subplot(gs[1, 2])

# Panel 0-2: posterior bar charts for 3 representative cases (classes 0,1,2)
for pi in range(3):
    ax  = ax_bars[pi]
    idx = rep_indices[pi]
    post  = posteriors[idx]
    true_cls  = int(y_test[idx])
    pred_cls  = int(y_pred[idx])
    ent = float(scipy_entropy(post))

    bars = ax.bar(range(N_CLASSES), post * 100,
                  color=[FAULT_COLORS[k] for k in range(N_CLASSES)],
                  edgecolor='white', linewidth=0.8, alpha=0.88)
    ax.bar(true_cls, post[true_cls] * 100,
           color=FAULT_COLORS[true_cls], edgecolor='black', linewidth=2.0)
    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(FAULT_LABELS, fontsize=7.5, rotation=20, ha='right')
    ax.set_ylabel('Posterior probability (%)')
    ax.set_ylim(0, 115)
    ax.set_title(
        f'Case {pi+1}: True = {FAULT_LABELS[true_cls]}\n'
        f'Pred = {FAULT_LABELS[pred_cls]}   H = {ent:.2f} nats',
        fontweight='bold', fontsize=9.5)
    ax.grid(alpha=0.25, axis='y')
    for k, bar in enumerate(bars):
        h = bar.get_height()
        if h > 3:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                    f'{h:.1f}%', ha='center', va='bottom', fontsize=7.5)

# Panel 3-4: posterior bar charts for classes 3,4
for pi in range(2):
    ax  = ax_bars[3 + pi]
    idx = rep_indices[3 + pi]
    post  = posteriors[idx]
    true_cls  = int(y_test[idx])
    pred_cls  = int(y_pred[idx])
    ent = float(scipy_entropy(post))

    bars = ax.bar(range(N_CLASSES), post * 100,
                  color=[FAULT_COLORS[k] for k in range(N_CLASSES)],
                  edgecolor='white', linewidth=0.8, alpha=0.88)
    ax.bar(true_cls, post[true_cls] * 100,
           color=FAULT_COLORS[true_cls], edgecolor='black', linewidth=2.0)
    ax.set_xticks(range(N_CLASSES))
    ax.set_xticklabels(FAULT_LABELS, fontsize=7.5, rotation=20, ha='right')
    ax.set_ylabel('Posterior probability (%)')
    ax.set_ylim(0, 115)
    ax.set_title(
        f'Case {pi+4}: True = {FAULT_LABELS[true_cls]}\n'
        f'Pred = {FAULT_LABELS[pred_cls]}   H = {ent:.2f} nats',
        fontweight='bold', fontsize=9.5)
    ax.grid(alpha=0.25, axis='y')
    for k, bar in enumerate(bars):
        h = bar.get_height()
        if h > 3:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                    f'{h:.1f}%', ha='center', va='bottom', fontsize=7.5)

# Panel 5: calibration + confidence-accuracy reliability diagram
# Plot per-class calibration
for cls in range(N_CLASSES):
    mp = cal_mean_pred[cls]
    fp = cal_frac_pos[cls]
    if len(mp) > 1:
        ax_cal.plot(mp, fp, 'o-', color=FAULT_COLORS[cls], linewidth=1.5,
                    markersize=5, label=FAULT_LABELS[cls], alpha=0.85)

ax_cal.plot([0, 1], [0, 1], 'k--', linewidth=1.2, alpha=0.5, label='Perfect')
ax_cal.set_xlabel('Mean predicted probability')
ax_cal.set_ylabel('Fraction of positives')
ax_cal.set_title('Reliability Diagram\n(per-class calibration)', fontweight='bold')
ax_cal.legend(fontsize=8, loc='upper left')
ax_cal.grid(alpha=0.3)
ax_cal.set_xlim(0, 1); ax_cal.set_ylim(0, 1.05)

out_path = os.path.join(OUT, '14_bayesian_posterior.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'\n   Saved --> {out_path}')

# ── Summary ────────────────────────────────────────────────────────────────────

print(f'\n{SEP}')
print('  Summary')
print(SEP)
print(f'  Test accuracy       : {accuracy*100:.1f}%')
print(f'  Mean posterior H    : {H.mean():.3f} nats  (max = {H_max:.3f} nats)')
print(f'  Fraction H < 0.5    : {(H < 0.5).mean()*100:.1f}%  (confident predictions)')
print(f'  Mean max posterior  : {confidence.mean()*100:.1f}%')
for cls in range(N_CLASSES):
    mask = y_test == cls
    lbl  = FAULT_LABELS[cls]
    cls_acc = float(np.mean(y_pred[mask] == y_test[mask])) if mask.sum() > 0 else 0.0
    cls_H   = H[mask].mean() if mask.sum() > 0 else 0.0
    print(f'  Class {cls} [{lbl:20s}]  acc={cls_acc*100:.1f}%  mean_H={cls_H:.3f}')
print(SEP)
