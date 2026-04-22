"""
pid_controller.py — PID Feedback Simulation for Dilution Refrigerator
=======================================================================

Simulates a digital PID feedback loop that regulates the Still heater power
to maintain a target MXC setpoint temperature.

Control loop structure:
  error(t) = T_MXC_target − T_MXC(t)
  u(t)     = Kp·e(t) + Ki·∫e(t)dt + Kd·de/dt
  P_still  = P_nominal + clamp(u(t), 0, P_max_still)

The PID is discretised with a zero-order hold (ZOH) at 1 Hz sampling,
matching a realistic digital controller.

Scenarios simulated
-------------------
1. Setpoint step: T_MXC_target 15 mK → 25 mK → back to 15 mK
2. Disturbance rejection: external heat pulse at t=2000 s (10 mW for 300 s)
3. PID parameter sweep: Kp sweep, overdamped vs critically damped vs underdamped

Output
------
outputs/pid_step_response.png
outputs/pid_disturbance.png
outputs/pid_kp_sweep.png

References
----------
Pobell F., Matter and Methods at Low Temperatures, Springer 2007 (Ch. 4)
Ang K. et al., ISA Trans. 44, 223 (2005) — PID autotuning
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from cryo_thermal import _G0, _C, cooling_power, STAGE_NAMES

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# ── PID controller ─────────────────────────────────────────────────────────────────

class DigitalPID:
    """
    Discrete-time PID controller with anti-windup and output clamping.

    u[k] = Kp*e[k] + Ki*dt*sum(e) + Kd*(e[k]-e[k-1])/dt
    """
    def __init__(self, Kp: float, Ki: float, Kd: float,
                 u_min: float = 0.0, u_max: float = 0.05,
                 dt: float = 1.0):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.u_min, self.u_max = u_min, u_max
        self.dt  = dt
        self._integral = 0.0
        self._prev_err = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_err = 0.0

    def step(self, setpoint: float, measurement: float) -> float:
        e    = setpoint - measurement
        self._integral += e * self.dt
        deriv = (e - self._prev_err) / self.dt
        u = self.Kp * e + self.Ki * self._integral + self.Kd * deriv
        # Anti-windup: clamp integral if output is saturated
        u_sat = float(np.clip(u, self.u_min, self.u_max))
        if u != u_sat:
            self._integral -= e * self.dt  # undo windup
        self._prev_err = e
        return u_sat


# ── Coupled thermal simulation with PID ─────────────────────────────────────────────

def run_pid_simulation(
    pid:           DigitalPID,
    t_end:         float = 7200.0,       # simulation time (s)
    dt_ctrl:       float = 1.0,          # controller sample period (s)
    T_mxc_target:  float | list = 0.015,  # setpoint (K) or list of (t, T) tuples
    n3_flow_umol:  float = 476.0,
    disturbance:   tuple | None = None,   # (t_start, t_end, P_mxc_W)
) -> dict:
    """
    Simulate PID-controlled MXC temperature.

    Uses Euler integration of the thermal ODEs with the controller applying
    a heater correction to the Still stage.
    """

    n_steps = int(t_end / dt_ctrl)
    T       = np.zeros((n_steps + 1, 6))
    T[0, :] = [300.0, 50.0, 4.0, 0.70, 0.10, 0.015]   # initial conditions

    P_still_arr = np.zeros(n_steps + 1)
    t_arr       = np.linspace(0, t_end, n_steps + 1)

    pid.reset()

    def get_setpoint(t_now):
        if isinstance(T_mxc_target, (int, float)):
            return float(T_mxc_target)
        # List of (time_s, setpoint_K)
        sp = T_mxc_target[0][1]
        for (t_s, t_val) in T_mxc_target:
            if t_now >= t_s:
                sp = t_val
        return sp

    for k in range(n_steps):
        T_k    = T[k, :].copy()
        t_now  = t_arr[k]

        # PID output: extra still heater power
        sp = get_setpoint(t_now)
        P_ctrl = pid.step(sp, T_k[5])   # control MXC via Still heater
        P_still_arr[k] = P_ctrl

        # Build extra heat vector
        extra = np.zeros(6)
        extra[3] = P_ctrl   # Still heater

        # Optional disturbance on MXC
        if disturbance is not None:
            t_d0, t_d1, P_d = disturbance
            if t_d0 <= t_now <= t_d1:
                extra[5] += P_d

        # 300 K plate is fixed
        T_k[0] = 300.0

        # Euler step for stages 1-5
        for i in range(1, 6):
            Q_cool = cooling_power(i, T_k[i], n3_flow_umol)
            Q_in   = _G0[i] * (T_k[i-1] - T_k[i]) + extra[i]
            Q_out  = _G0[i] * (T_k[i] - T_k[i+1]) if i < 5 else 0.0
            dTdt   = (Q_in - Q_out - Q_cool) / _C[i]
            T[k+1, i] = T_k[i] + dTdt * dt_ctrl
            T[k+1, i] = max(T[k+1, i], 0.001)  # physical floor

        T[k+1, 0] = 300.0

    P_still_arr[-1] = P_still_arr[-2]

    return {"t": t_arr, "T": T, "P_still": P_still_arr}


# ── Demo plots ────────────────────────────────────────────────────────────────────

def demo_pid():
    print("=" * 60)
    print("  PID Feedback Simulation — MXC Temperature Control")
    print("=" * 60)

    # ── Scenario 1: Setpoint step response
    pid = DigitalPID(Kp=50.0, Ki=0.05, Kd=200.0, u_max=0.05)
    setpoints = [(0, 0.015), (1800, 0.025), (3600, 0.015)]   # step up then back
    result1 = run_pid_simulation(pid, t_end=5400, T_mxc_target=setpoints)

    t1 = result1["t"] / 60   # minutes
    T_mxc1 = result1["T"][:, 5] * 1e3   # mK
    P_ctrl1 = result1["P_still"] * 1e3  # mW

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    fig.suptitle("PID MXC Temperature Control — Setpoint Step Response",
                 fontsize=12, fontweight="bold")

    # Setpoint trace
    sp_trace = np.array([15.0 if t < 30 else (25.0 if t < 60 else 15.0) for t in t1])
    ax1.plot(t1, T_mxc1, color="#E74C3C", linewidth=2, label="T_MXC (simulated)")
    ax1.step(t1[::60], sp_trace[::60], where="post", color="#3498DB",
             linewidth=1.5, linestyle="--", label="Setpoint")
    ax1.set_ylabel("MXC Temperature (mK)"); ax1.legend(fontsize=9); ax1.grid(alpha=0.35)

    ax2.plot(t1, P_ctrl1, color="#2ECC71", linewidth=2)
    ax2.set_ylabel("Still heater P_ctrl (mW)")
    ax2.set_xlabel("Time (min)"); ax2.grid(alpha=0.35)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pid_step_response.png"), dpi=150, bbox_inches="tight")
    print(f"Saved → {os.path.join(OUT_DIR, 'pid_step_response.png')}")

    # ── Scenario 2: Disturbance rejection
    pid.reset()
    result2 = run_pid_simulation(
        pid, t_end=5400, T_mxc_target=0.015,
        disturbance=(2000, 2300, 0.003))   # 3 mW pulse for 300 s

    t2 = result2["t"] / 60
    T_mxc2 = result2["T"][:, 5] * 1e3
    P_ctrl2 = result2["P_still"] * 1e3

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    fig.suptitle("PID MXC Temperature Control — Disturbance Rejection\n"
                 "(3 mW heat pulse on MXC, t=2000–2300 s)",
                 fontsize=11, fontweight="bold")
    ax1.plot(t2, T_mxc2, color="#E74C3C", linewidth=2, label="T_MXC")
    ax1.axhline(15.0, color="#3498DB", linestyle="--", linewidth=1.5, label="Setpoint 15 mK")
    ax1.axvspan(2000/60, 2300/60, color="#F39C12", alpha=0.2, label="Disturbance")
    ax1.set_ylabel("MXC Temperature (mK)"); ax1.legend(fontsize=9); ax1.grid(alpha=0.35)
    ax2.plot(t2, P_ctrl2, color="#2ECC71", linewidth=2)
    ax2.axvspan(2000/60, 2300/60, color="#F39C12", alpha=0.2)
    ax2.set_ylabel("Still heater P_ctrl (mW)"); ax2.set_xlabel("Time (min)"); ax2.grid(alpha=0.35)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pid_disturbance.png"), dpi=150, bbox_inches="tight")
    print(f"Saved → {os.path.join(OUT_DIR, 'pid_disturbance.png')}")

    # ── Scenario 3: Kp sweep
    kp_values = [10, 50, 150, 400]
    kp_colors = ["#3498DB", "#2ECC71", "#E67E22", "#E74C3C"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.suptitle("PID Kp Sweep — MXC Step Response (15→25 mK setpoint)",
                 fontsize=11, fontweight="bold")
    for kp, col in zip(kp_values, kp_colors):
        p = DigitalPID(Kp=kp, Ki=0.05, Kd=kp*4, u_max=0.05)
        r = run_pid_simulation(p, t_end=3600,
                               T_mxc_target=[(0, 0.015), (600, 0.025)])
        t_m = r["t"] / 60
        ax.plot(t_m, r["T"][:, 5] * 1e3, color=col, linewidth=2, label=f"Kp={kp}")
    ax.axhline(15, color="#888", linestyle=":", linewidth=1)
    ax.axhline(25, color="#888", linestyle=":", linewidth=1)
    ax.set_ylabel("MXC Temperature (mK)"); ax.set_xlabel("Time (min)")
    ax.legend(fontsize=9); ax.grid(alpha=0.35)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pid_kp_sweep.png"), dpi=150, bbox_inches="tight")
    print(f"Saved → {os.path.join(OUT_DIR, 'pid_kp_sweep.png')}")


if __name__ == "__main__":
    demo_pid()
