"""
mxc_temp_sweep.py — MXC Temperature vs ³He Flow Rate Parameter Sweep
=====================================================================

Sweeps the ³He circulation rate ṅ₃ (µmol/s) and computes the steady-state
MXC temperature for each flow rate.

Physics
-------
In a dilution refrigerator (Pobell 2007, Ch. 6), the MXC cooling power is:

    Q̇_MXC ≈ 84 × ṅ₃ × T²_MXC  [µW, µmol/s, K²]

At steady state Q̇_MXC = Q̇_load, giving:
    T_MXC,ss = sqrt(Q̇_load / (84 × ṅ₃))

This is the "Pobell T² scaling" — linear on a log(T) vs log(ṅ₃) plot with
slope −0.5.

What this script does
---------------------
1. Sweeps ṅ₃ over 50–700 µmol/s (typical operating range)
2. For each ṅ₃, runs the 6-stage thermal ODE to steady state (1-hour simulation)
3. Extracts T_MXC,ss and T_Still,ss
4. Overlays the Pobell T² analytical prediction

Outputs
-------
outputs/mxc_temp_vs_n3flow.png  — Main T_MXC vs ṅ₃ plot with Pobell overlay
outputs/stage_temps_vs_n3flow.png — All 6 stage temperatures vs ṅ₃

References
----------
Pobell F., Matter and Methods at Low Temperatures, Springer 2007 (Sec. 6.3)
Uhlig K., J. Low Temp. Phys. 133, 215 (2003) — dry dilution refrigerators
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from cryo_thermal import simulate_cooldown, STAGE_NAMES, _G0, _C, cooling_power

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Pobell analytical fit constants (valid for dry dilution refrigerators)
POBELL_A = 84.0    # µW / (µmol/s · K²) — Pobell eq. 6.12
Q_LOAD   = 2e-6    # 2 µW base heat load on MXC (W) — typical wiring + radiation


def pobell_t_mxc(n3: np.ndarray, q_load: float = Q_LOAD) -> np.ndarray:
    """
    Analytical MXC temperature from Pobell T² scaling:
        T_MXC = sqrt(Q_load / (84 * n3_umol_s * 1e-6))
    All inputs in SI; returns T in K.
    """
    return np.sqrt(q_load / (POBELL_A * n3 * 1e-6))


def run_sweep(
    n3_range: np.ndarray | None = None,
    t_end:     float = 3600.0,
    t_points:  int   = 300,
    q_extra_mxc: float = 0.0,   # extra heat on MXC (W)
) -> dict:
    """
    Sweep ṅ₃ and extract steady-state temperatures.

    Returns dict with:
        n3, T_mxc, T_still, T_4k, T_50k, T_cp, T_all
    """
    if n3_range is None:
        n3_range = np.linspace(50, 700, 33)

    T_mxc_ss  = np.zeros(len(n3_range))
    T_still_ss = np.zeros(len(n3_range))
    T_4k_ss    = np.zeros(len(n3_range))
    T_50k_ss   = np.zeros(len(n3_range))
    T_cp_ss    = np.zeros(len(n3_range))
    T_all_ss   = np.zeros((len(n3_range), 6))

    t_hours_sim = t_end / 3600.0

    for ki, n3 in enumerate(n3_range):
        params = {
            "n3_flow_umol": float(n3),
            "P_qubit_W":    q_extra_mxc,
        }
        _, T = simulate_cooldown(t_hours=t_hours_sim, params=params, n_eval=t_points)
        # Steady state = mean of last 20% of simulation
        n_tail = max(1, len(T) // 5)
        T_ss = T[-n_tail:, :].mean(axis=0)
        T_mxc_ss[ki]   = T_ss[5]
        T_still_ss[ki] = T_ss[3]
        T_4k_ss[ki]    = T_ss[2]
        T_50k_ss[ki]   = T_ss[1]
        T_cp_ss[ki]    = T_ss[4]
        T_all_ss[ki]   = T_ss

        if (ki + 1) % 5 == 0 or ki == 0:
            print(f"  ṅ₃={n3:5.0f} µmol/s → T_MXC={T_ss[5]*1e3:.1f} mK  "
                  f"T_Still={T_ss[3]*1e3:.0f} mK", flush=True)

    return {
        "n3":        n3_range,
        "T_mxc":     T_mxc_ss,
        "T_still":   T_still_ss,
        "T_4k":      T_4k_ss,
        "T_50k":     T_50k_ss,
        "T_cp":      T_cp_ss,
        "T_all":     T_all_ss,
    }


def plot_mxc_pobell(sweep: dict) -> None:
    """MXC temperature vs ṅ₃ with Pobell T² overlay."""
    n3   = sweep["n3"]
    t_mxc = sweep["T_mxc"] * 1e3   # mK
    t_pobell = pobell_t_mxc(n3) * 1e3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("MXC Steady-State Temperature vs ³He Circulation Rate",
                 fontsize=12, fontweight="bold")

    # Linear scale
    ax1.plot(n3, t_mxc, "o-", color="#E74C3C", linewidth=2, markersize=5,
             label="ODE simulation (6-stage model)")
    ax1.plot(n3, t_pobell, "--", color="#3498DB", linewidth=2,
             label=f"Pobell T² scaling\n(Q_load={Q_LOAD*1e6:.0f} µW)")
    ax1.set_xlabel("³He flow rate ṅ₃ (µmol/s)")
    ax1.set_ylabel("T_MXC steady-state (mK)")
    ax1.set_title("Linear scale")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.35)

    # Log-log scale (slope -0.5 line)
    ax2.loglog(n3, t_mxc, "o-", color="#E74C3C", linewidth=2, markersize=5,
               label="ODE simulation")
    ax2.loglog(n3, t_pobell, "--", color="#3498DB", linewidth=2,
               label="Pobell T² (slope −0.5)")
    # Annotate slope
    x0, x1 = 100, 600
    y0 = t_pobell[np.searchsorted(n3, x0)]
    y1 = t_pobell[np.searchsorted(n3, x1)]
    ax2.annotate("", xy=(x1, y1), xytext=(x0, y0),
                 arrowprops=dict(arrowstyle="-|>", color="#888"))
    ax2.text(250, (y0+y1)/2 * 1.1, "slope −½", fontsize=9, color="#555")
    ax2.set_xlabel("³He flow rate ṅ₃ (µmol/s)")
    ax2.set_ylabel("T_MXC (mK)")
    ax2.set_title("Log-log scale (Pobell T² regime)")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.35, which="both")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "mxc_temp_vs_n3flow.png"), dpi=150, bbox_inches="tight")
    print(f"Saved → {os.path.join(OUT_DIR, 'mxc_temp_vs_n3flow.png')}")


def plot_all_stages(sweep: dict) -> None:
    """All 6 stage temperatures vs ṅ₃."""
    n3 = sweep["n3"]
    T_all = sweep["T_all"]

    colors = ["#E74C3C", "#E67E22", "#3498DB", "#2ECC71", "#9B59B6", "#F39C12"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, col) in enumerate(zip(STAGE_NAMES, colors)):
        T_i = T_all[:, i]
        if i == 0:
            continue   # 300K fixed
        ax.semilogy(n3, T_i * 1000, "o-", color=col, linewidth=2,
                    markersize=4, label=name)

    ax.set_xlabel("³He flow rate ṅ₃ (µmol/s)")
    ax.set_ylabel("Steady-state temperature (mK)")
    ax.set_title("All Stage Temperatures vs ³He Flow Rate", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.35, which="both")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "stage_temps_vs_n3flow.png"), dpi=150, bbox_inches="tight")
    print(f"Saved → {os.path.join(OUT_DIR, 'stage_temps_vs_n3flow.png')}")


def plot_heat_load_family(n3_range: np.ndarray | None = None) -> None:
    """Family of curves: T_MXC vs ṅ₃ for different heat loads."""
    if n3_range is None:
        n3_range = np.linspace(50, 700, 25)

    q_loads = [0.0, 1e-6, 5e-6, 10e-6, 20e-6]   # 0, 1, 5, 10, 20 µW
    q_colors = ["#2ECC71", "#3498DB", "#E67E22", "#E74C3C", "#9B59B6"]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("MXC Temperature vs ṅ₃ — Heat Load Family of Curves",
                 fontsize=11, fontweight="bold")

    for q, col in zip(q_loads, q_colors):
        t_p = pobell_t_mxc(n3_range, q_load=max(q, 1e-10)) * 1e3
        ax.plot(n3_range, t_p, color=col, linewidth=2,
                label=f"Q_load = {q*1e6:.0f} µW")

    ax.axvline(476, color="#888", linestyle="--", linewidth=1.2, label="ṅ₃=476 µmol/s (nominal)")
    ax.set_xlabel("³He flow rate ṅ₃ (µmol/s)")
    ax.set_ylabel("T_MXC (mK) — Pobell T² scaling")
    ax.legend(fontsize=9); ax.grid(alpha=0.35)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "mxc_family_curves.png"), dpi=150, bbox_inches="tight")
    print(f"Saved → {os.path.join(OUT_DIR, 'mxc_family_curves.png')}")


def run_mxc_sweep():
    print("=" * 60)
    print("  MXC Temperature vs ³He Flow Rate — Parameter Sweep")
    print("=" * 60)
    print("\nRunning ṅ₃ sweep…")
    n3_range = np.linspace(50, 700, 27)
    sweep = run_sweep(n3_range=n3_range)

    print(f"\nResults at nominal flow (476 µmol/s):")
    idx = np.argmin(np.abs(sweep["n3"] - 476))
    print(f"  T_MXC  = {sweep['T_mxc'][idx]*1e3:.2f} mK")
    print(f"  T_Still= {sweep['T_still'][idx]*1e3:.0f} mK")
    print(f"  Pobell prediction: {pobell_t_mxc(np.array([476.0]))[0]*1e3:.2f} mK")

    plot_mxc_pobell(sweep)
    plot_all_stages(sweep)
    plot_heat_load_family()
    print("\nDone.")


if __name__ == "__main__":
    run_mxc_sweep()
