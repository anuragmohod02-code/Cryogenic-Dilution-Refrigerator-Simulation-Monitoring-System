"""
cooldown_predictor.py -- Dilution Refrigerator Cooldown Time Prediction
========================================================================

Uses the 6-stage thermal ODE (simulate_cooldown) to predict how long it
takes to reach base temperature (T_MXC < 20 mK) from a warm start, and
how that time depends on two key operating parameters:

  * n3_flow  -- ³He circulation rate [µmol/s]  (controls sub-K cooling power)
  * P_qubit  -- qubit chip dissipation [µW]    (extra MXC heat load)

Analysis produced
-----------------
  Panel 1: Nominal cooldown trajectory (all 6 stages vs time)
  Panel 2: T_MXC zoom -- last stage approach to base temperature
  Panel 3: Heatmap -- time-to-base vs (n3_flow, P_qubit)
  Panel 4: Sensitivity curves -- T_MXC at t=10 h vs each parameter separately

Physics
-------
MXC cooling power: P_cool = n3_flow * 84e-6 * T_MXC^2  [W]
  (latent heat of ³He mixing, Pobell ch. 6; 84 J/(mol·K²))

Steady-state T_MXC follows P_cool = P_qubit + Q_conductance,
so T_MXC,ss = sqrt((P_qubit + Q_from_above) / (n3 * 84e-6))

Output
------
    outputs/15_cooldown_predictor.png  -- 4-panel figure

References
----------
Pobell F., Matter and Methods at Low Temperatures (3rd ed., 2007) -- ch. 6
Wiedemann-Franz law for wiring heat loads -- Standard Reference
"""

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cryo_thermal import simulate_cooldown, STAGE_NAMES

OUT = os.path.join(ROOT, 'outputs')
os.makedirs(OUT, exist_ok=True)
SEP = '=' * 60

# Stage colours consistent with other P3 plots
STAGE_COLORS = ['#E74C3C', '#E67E22', '#F1C40F', '#2ECC71', '#3498DB', '#9B59B6']
T_BASE_MK = 20.0      # mK -- base temperature threshold
T_BASE_K  = T_BASE_MK * 1e-3

print(SEP)
print('  Dilution Refrigerator Cooldown Predictor')
print(SEP)


# ── Helper: time to reach T_MXC < T_BASE_K ───────────────────────────────────

def time_to_base(n3_flow_umol, P_qubit_uW, t_max_h=14.0, n_eval=800):
    """
    Simulate cooldown and return time (hours) when T_MXC first drops below
    T_BASE_K = 20 mK.  Returns t_max_h if never reached within simulation.
    """
    params = {
        'n3_flow_umol': float(n3_flow_umol),
        'P_qubit_W':    float(P_qubit_uW) * 1e-6,
    }
    t, T = simulate_cooldown(t_hours=t_max_h, params=params, n_eval=n_eval)
    T_mxc = T[:, 5]
    idx = np.where(T_mxc <= T_BASE_K)[0]
    return float(t[idx[0]]) if len(idx) > 0 else float(t_max_h)


# ── Panel 1 + 2: nominal cooldown trajectory ──────────────────────────────────

print('\n[1/3] Nominal cooldown trajectory (n3=476 umol/s, P_qubit=5 uW)...')
t_nom, T_nom = simulate_cooldown(
    t_hours=12.0,
    params={'n3_flow_umol': 476.0, 'P_qubit_W': 5e-6},
    n_eval=2000)

T_mxc_nom = T_nom[:, 5] * 1e3        # K -> mK
idx_base = np.where(T_nom[:, 5] <= T_BASE_K)[0]
t_base_nom = float(t_nom[idx_base[0]]) if len(idx_base) > 0 else None
print(f'   Time to {T_BASE_MK:.0f} mK: {t_base_nom:.2f} h' if t_base_nom else '   Never reached base T')


# ── Panel 3: heatmap -- time to base vs (n3_flow, P_qubit) ───────────────────

print('\n[2/3] Heatmap: time-to-base vs n3_flow and P_qubit...')
n3_vals   = np.array([300, 350, 400, 450, 476, 520, 560, 600])   # umol/s
Pq_vals   = np.array([0, 1, 2, 5, 10, 20, 50, 100])              # uW

ttb_map = np.zeros((len(Pq_vals), len(n3_vals)))
for i, Pq in enumerate(Pq_vals):
    for j, n3 in enumerate(n3_vals):
        ttb_map[i, j] = time_to_base(n3, Pq)
    print(f'   P_qubit={Pq:4.0f} uW  -> t_base (h): '
          + '  '.join(f'{ttb_map[i, j]:.2f}' for j in range(len(n3_vals))),
          flush=True)


# ── Panel 4: sensitivity curves ──────────────────────────────────────────────

print('\n[3/3] Sensitivity: T_MXC at t=10 h vs n3_flow and P_qubit...')
n3_sweep  = np.linspace(250, 650, 20)
Pq_sweep  = np.logspace(np.log10(0.1), np.log10(200), 20)    # 0.1 -- 200 uW

T_mxc_vs_n3 = np.zeros(len(n3_sweep))
T_mxc_vs_Pq = np.zeros(len(Pq_sweep))

for i, n3 in enumerate(n3_sweep):
    _, T = simulate_cooldown(t_hours=10.0, params={'n3_flow_umol': n3, 'P_qubit_W': 5e-6}, n_eval=600)
    T_mxc_vs_n3[i] = T[-1, 5] * 1e3   # mK

for i, Pq in enumerate(Pq_sweep):
    _, T = simulate_cooldown(t_hours=10.0, params={'n3_flow_umol': 476.0, 'P_qubit_W': Pq * 1e-6}, n_eval=600)
    T_mxc_vs_Pq[i] = T[-1, 5] * 1e3   # mK

print('   Sensitivity sweeps complete.')


# ── Plotting ──────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'figure.dpi':        150,
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'legend.fontsize':   9,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'lines.linewidth':   2.0,
})

fig, axes = plt.subplots(2, 2, figsize=(13, 9),
                          gridspec_kw={'hspace': 0.42, 'wspace': 0.35})
fig.suptitle(
    'Dilution Refrigerator Cooldown Predictor\n'
    '(6-stage Lumped-RC ODE, Oxford Triton 400 parameters)',
    fontsize=12, fontweight='bold')

ax_traj, ax_mxc = axes[0]
ax_heat, ax_sens = axes[1]

# Panel 1: full cooldown trajectory
for s in range(6):
    T_plot = T_nom[:, s]
    # Use log scale; clip to avoid log(0)
    label = STAGE_NAMES[s]
    ax_traj.semilogy(t_nom, np.maximum(T_plot, 1e-4),
                     color=STAGE_COLORS[s], linewidth=2, label=label)
ax_traj.set_xlabel('Time (hours)')
ax_traj.set_ylabel('Temperature (K, log scale)')
ax_traj.set_title('Full Cooldown Trajectory\n(300 K warm start)', fontweight='bold')
ax_traj.legend(fontsize=8.5, loc='upper right')
ax_traj.grid(alpha=0.3, which='both')
ax_traj.axhline(T_BASE_K, color='black', linestyle=':', linewidth=1.5, alpha=0.6,
                label='20 mK target')
if t_base_nom:
    ax_traj.axvline(t_base_nom, color='black', linestyle='--', linewidth=1.2, alpha=0.5)
    ax_traj.text(t_base_nom + 0.1, 0.1,
                 f't_base = {t_base_nom:.1f} h', fontsize=8.5, color='black')

# Panel 2: MXC zoom (mK)
ax_mxc.plot(t_nom, T_mxc_nom, color=STAGE_COLORS[5], linewidth=2.5)
ax_mxc.axhline(T_BASE_MK, color='#E74C3C', linestyle='--', linewidth=1.8,
               label=f'{T_BASE_MK:.0f} mK base')
if t_base_nom:
    ax_mxc.axvline(t_base_nom, color='#E74C3C', linestyle=':', linewidth=1.5)
    ax_mxc.annotate(f't = {t_base_nom:.1f} h',
                    xy=(t_base_nom, T_BASE_MK),
                    xytext=(t_base_nom + 0.4, T_BASE_MK * 4),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                    fontsize=9)
ax_mxc.set_xlabel('Time (hours)')
ax_mxc.set_ylabel('T_MXC (mK)')
ax_mxc.set_title('MXC Stage: Approach to Base Temperature', fontweight='bold')
ax_mxc.set_yscale('log')
ax_mxc.legend(fontsize=9); ax_mxc.grid(alpha=0.3, which='both')
ax_mxc.set_ylim([1, None])

# Panel 3: heatmap
img = ax_heat.pcolormesh(
    n3_vals, Pq_vals, ttb_map,
    cmap='viridis_r', shading='auto')
cbar = fig.colorbar(img, ax=ax_heat, label='Time to base (h)')
ax_heat.set_xlabel('³He flow rate (µmol/s)')
ax_heat.set_ylabel('Qubit dissipation P_qubit (µW)')
ax_heat.set_title('Time to 20 mK\nvs Flow Rate & Qubit Load', fontweight='bold')
ax_heat.set_yscale('log')
# Mark nominal operating point
ax_heat.axvline(476, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
ax_heat.axhline(5,   color='white', linestyle='--', linewidth=1.5, alpha=0.8)
ax_heat.text(478, 6, 'nominal', color='white', fontsize=8.5, va='bottom')

# Contour overlay
try:
    cs = ax_heat.contour(n3_vals, Pq_vals, ttb_map,
                          levels=[6, 8, 10, 12],
                          colors='white', linewidths=0.8, alpha=0.6)
    ax_heat.clabel(cs, fmt='%.0f h', fontsize=7.5)
except Exception:
    pass

# Panel 4: sensitivity curves
ax_s1 = ax_sens
ax_s2 = ax_sens.twinx()
l1, = ax_s1.plot(n3_sweep, T_mxc_vs_n3, color='#2980B9', linewidth=2.5,
                 label='T_MXC vs n3 (P_q=5 µW)')
ax_s1.axhline(T_BASE_MK, color='#2980B9', linestyle=':', linewidth=1.2, alpha=0.5)
ax_s1.set_xlabel('³He flow rate (µmol/s)')
ax_s1.set_ylabel('T_MXC at t=10 h (mK)', color='#2980B9')
ax_s1.tick_params(axis='y', labelcolor='#2980B9')
ax_s1.set_ylim(bottom=0)

l2, = ax_s2.semilogx(Pq_sweep, T_mxc_vs_Pq, color='#E74C3C', linewidth=2.5,
                      label='T_MXC vs P_qubit (n3=476)')
ax_s2.axhline(T_BASE_MK, color='#E74C3C', linestyle=':', linewidth=1.2, alpha=0.5)
ax_s2.set_ylabel('T_MXC at t=10 h (mK) [P_qubit axis]', color='#E74C3C')
ax_s2.tick_params(axis='y', labelcolor='#E74C3C')
ax_s2.set_ylim(bottom=0)

ax_sens.set_title(f'Sensitivity: T_MXC vs Operating Parameters\n'
                  f'(steady-state at t=10 h)', fontweight='bold')
lines = [l1, l2]
labels = [l.get_label() for l in lines]
ax_sens.legend(lines, labels, fontsize=8.5, loc='upper left')
ax_sens.grid(alpha=0.3)

out_path = os.path.join(OUT, '15_cooldown_predictor.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'\n   Saved --> {out_path}')

print(f'\n{SEP}')
print('  Summary')
print(SEP)
print(f'  Nominal cooldown (n3=476 umol/s, P_qubit=5 uW):')
if t_base_nom:
    print(f'    Time to {T_BASE_MK:.0f} mK  = {t_base_nom:.2f} h')
T_ss = T_nom[-1, 5] * 1e3
print(f'    T_MXC at t=12 h      = {T_ss:.2f} mK')
print(f'  Heatmap range:')
print(f'    Fastest t_base       = {ttb_map.min():.2f} h '
      f'(n3={n3_vals[np.unravel_index(ttb_map.argmin(), ttb_map.shape)[1]]:.0f} umol/s, '
      f'P_q={Pq_vals[np.unravel_index(ttb_map.argmin(), ttb_map.shape)[0]]:.0f} uW)')
print(f'    Slowest t_base       = {ttb_map.max():.2f} h '
      f'(limiting parameter: high P_qubit or low n3 flow)')
print(f'  Sensitivity:')
print(f'    T_MXC @ n3=250 umol/s  = {T_mxc_vs_n3[0]:.2f} mK')
print(f'    T_MXC @ n3=650 umol/s  = {T_mxc_vs_n3[-1]:.2f} mK')
print(f'    T_MXC @ P_q=0.1 uW     = {T_mxc_vs_Pq[0]:.2f} mK')
print(f'    T_MXC @ P_q=200 uW     = {T_mxc_vs_Pq[-1]:.2f} mK')
print(SEP)
