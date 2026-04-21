"""
run_pipeline.py
---------------
Standalone runner for the cryogenic DR thermal simulation.
Generates outputs/stage_temperatures.csv and outputs/cooldown_curve.png
without needing MATLAB.

Usage:
    python run_pipeline.py
"""

import sys
import os

# Make sure the project python/ directory is importable
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cryo_thermal import (
    simulate_cooldown, heat_balance,
    STAGE_NAMES, STAGE_TARGETS_K,
)

OUT_DIR = os.path.join(os.path.dirname(script_dir), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("  Dilution Refrigerator Thermal Simulation")
    print("=" * 60)

    # ── Run simulation ────────────────────────────────────────────────
    print("\nRunning ODE solver (10-hour cool-down)…")
    t_h, T = simulate_cooldown(t_hours=10.0, n_eval=3000)
    print(f"  Done — {len(t_h)} time points, "
          f"t = 0 → {t_h[-1]:.2f} h")

    # ── Steady-state report ───────────────────────────────────────────
    T_final = T[-1]
    print("\n--- Steady-State Temperatures ---")
    for i, (name, target) in enumerate(zip(STAGE_NAMES, STAGE_TARGETS_K)):
        T_v = T_final[i]
        if T_v >= 1.0:
            val_str = f"{T_v:.4g} K"
        else:
            val_str = f"{T_v*1e3:.2f} mK"
        marker = "✓" if T_v < target * 2 else "⚠"
        print(f"  {marker} {name:<18} : {val_str}")

    # ── Heat balance ──────────────────────────────────────────────────
    print("\n--- Heat Balance at Steady State ---")
    print(f"  {'Stage':<18}  {'P_cool':>10}  {'Q_in':>10}  {'Balance':>10}")
    for row in heat_balance(T_final):
        def fmt(W):
            if abs(W) < 1e-6:
                return f"{W*1e9:.2f} nW"
            elif abs(W) < 1e-3:
                return f"{W*1e6:.2f} µW"
            elif abs(W) < 1.0:
                return f"{W*1e3:.2f} mW"
            else:
                return f"{W:.3f}  W"
        print(f"  {row['stage']:<18}  {fmt(row['P_cool_W']):>10}  "
              f"{fmt(row['Q_in_W']):>10}  {fmt(row['balance_W']):>10}")

    # ── Save CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "stage_temperatures.csv")
    header = "time_h,T_300K,T_50K,T_4K,T_still,T_cold_plate,T_mxc"
    data = np.column_stack([t_h, T])
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
    print(f"\nCSV saved  → {csv_path}")

    # ── Plot cool-down curves ─────────────────────────────────────────
    colors = plt.cm.tab10(np.linspace(0, 0.9, 6))

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    fig.patch.set_facecolor("#1A1A2E")

    for ax in axes:
        ax.set_facecolor("#16213E")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#4A5568")

    # All 6 stages
    ax0 = axes[0]
    for i in range(6):
        ax0.semilogy(t_h, T[:, i], lw=2, color=colors[i], label=STAGE_NAMES[i])
    ax0.set_xlabel("Time (hours)")
    ax0.set_ylabel("Temperature (K)  [log]")
    ax0.set_title("Dilution Refrigerator Cool-down — All Stages")
    ax0.set_ylim(1e-3, 400)
    ax0.legend(loc="upper right", fontsize=9,
               facecolor="#16213E", labelcolor="white", framealpha=0.8)
    ax0.grid(True, alpha=0.3, color="#4A5568")

    # Sub-K zoom
    ax1 = axes[1]
    sub_k_idx = [3, 4, 5]
    for i in sub_k_idx:
        ax1.semilogy(t_h, T[:, i] * 1e3, lw=2, color=colors[i],
                     label=STAGE_NAMES[i])
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Temperature (mK)  [log]")
    ax1.set_title("Sub-Kelvin Stages Zoom")
    ax1.legend(loc="upper right", fontsize=10,
               facecolor="#16213E", labelcolor="white", framealpha=0.8)
    ax1.grid(True, alpha=0.3, color="#4A5568")

    fig.tight_layout(pad=2.0)
    png_path = os.path.join(OUT_DIR, "cooldown_curve.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Plot saved → {png_path}")

    # ── Summary stats ─────────────────────────────────────────────────
    mxc_final_mK = T_final[5] * 1e3
    cold_plate_mK = T_final[4] * 1e3
    print("\n=== Summary ===")
    print(f"  MXC final temperature  : {mxc_final_mK:.2f} mK "
          f"({'✓ < 20 mK' if mxc_final_mK < 20 else '⚠ above target'})")
    print(f"  Cold plate temperature : {cold_plate_mK:.2f} mK")
    print(f"  Cool-down time 4K→mK  : ~{t_h[-1]:.1f} h")
    print("=" * 60)


if __name__ == "__main__":
    main()
