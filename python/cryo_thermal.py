"""
cryo_thermal.py
---------------
Python implementation of the 6-stage dilution refrigerator thermal ODE model.
Mirrors the MATLAB thermal_model.m physics so the Dash dashboard and Jupyter
notebooks can run without MATLAB.

Stages (indices 0-5):
    0 - 300 K plate  (fixed boundary)
    1 - 50 K stage
    2 - 4 K stage
    3 - Still  (~700 mK)
    4 - Cold plate (~100 mK)
    5 - MXC (~15 mK)
"""

import numpy as np
from scipy.integrate import solve_ivp

# ── Stage metadata ───────────────────────────────────────────────────────────
STAGE_NAMES = ["300 K Plate", "50 K Stage", "4 K Stage",
               "Still", "Cold Plate", "MXC"]
STAGE_TARGETS_K = np.array([300.0, 50.0, 4.0, 0.70, 0.10, 0.015])

# ── Physical constants ────────────────────────────────────────────────────────
STEFAN_BOLTZMANN = 5.6704e-8   # W/(m^2·K^4)

# ── Effective inter-stage conductances [W/K] ──────────────────────────────────
# Derived from equilibrium balance: G_i * ΔT = P_cool_i at operating point.
# G_12: radiation + wires 300K→4K shield; calibrated so T_2,eq ≈ 44 K
# G_23: wires 50K→4K stage;  calibrated so T_3,eq ≈  3.7 K
# G_34: wires + He-gas path 4K→Still; calibrated so T_4,eq ≈  0.52 K
# G_45: still→cold-plate He-mixture flow ≈ 0.5 mW/K (Pobell Table 6.1)
# G_56: cold-plate→MXC ³He condensate  ≈ 0.1 mW/K
#  Index i = conductance between stage (i-1) and stage i
_G0 = np.array([0.0, 0.05, 0.012, 0.003, 5e-4, 1e-4])

# ── Thermal capacities [J/K] ─────────────────────────────────────────────────
# Sized so time constants τ_i = C_i/G_i fit within the 10-hour simulation.
# τ_50K  = 500/0.05  = 10 ks ≈ 2.8 h
# τ_4K   =  30/0.012 =  2.5 ks ≈ 41 min
# τ_Still =   3/0.003 =  1 ks ≈ 17 min
# τ_CP    = 0.3/5e-4  = 600 s ≈ 10 min
# τ_MXC   = 0.03/1e-4 = 300 s ≈  5 min
_C = np.array([1e5, 500.0, 30.0, 3.0, 0.3, 0.03])


# ── Cooling power curves ─────────────────────────────────────────────────────

def cooling_power(stage: int, T: float) -> float:
    """
    Cooling power [W] of stage `stage` at temperature `T` [K].
    Analytical fits to Oxford Instruments Triton 400 specs.

    The curves are designed so that P_cool(T) ≈ heat load at the nominal
    operating temperature, and P_cool ≈ 0 well above that temperature
    (the pulse tube / dilution unit only provides useful cooling below a
    certain threshold).

    Parameters
    ----------
    stage : int  — stage index (1=50K, 2=4K, 3=Still, 4=ColdPlate, 5=MXC)
    T     : float — temperature [K]
    """
    T = max(T, 1e-10)   # guard against T≤0

    if stage == 1:
        # Pulse tube 1st stage: P = P_max*(1 - T_base/T)  (heat pump law)
        # Oxford Triton: ~40 W at 50 K, cools from 300 K; base temp ~30 K
        T_base = 30.0
        return max(40.0 * (1.0 - T_base / T), 0.0)

    elif stage == 2:
        # Pulse tube 2nd stage: ~1.5 W at 4 K; base temp ~2.5 K
        T_base = 2.5
        return max(1.5 * (1.0 - T_base / T), 0.0)

    elif stage == 3:
        # Still (dilution unit): T² law.
        # A so P(0.7 K) = 20 mW → A = 20e-3/0.49 ≈ 40.8e-3
        # No upper cutoff: the T² cooling drives the stage from ~4K to 0.7K.
        return 40.8e-3 * T ** 2

    elif stage == 4:
        # Cold plate: T² law.
        # A so P(0.1 K) = 2 mW → A = 0.2
        # No upper cutoff: drives stage from ~0.7K to 0.1K.
        return 0.2 * T ** 2

    elif stage == 5:
        # MXC mixing chamber: T² law (osmotic-pressure enthalpy of ³He).
        # A so P(0.1 K) = 400 µW → A = 0.04  W/K²
        # No upper cutoff: drives stage from ~0.1K to 15 mK.
        return 0.04 * T ** 2

    else:
        return 0.0


# ── Heat load model (kept for heat_balance reporting) ────────────────────────

def heat_loads(T_hot: float, T_cold: float,
               n_wires: int = 24,
               wire_A_m2: float = 2e-8,
               wire_L_m: float = 0.3,
               rad_area_m2: float = 0.01,
               emissivity: float = 0.05,
               P_qubit_W: float = 0.0) -> tuple[float, float, float]:
    """
    Returns (P_wire, P_rad, P_qubit) — heat loads from warm to cold stage [W].
    One-directional (T_hot > T_cold assumed).
    """
    kappa_mean = 0.1   # W/(m·K)
    P_wire = n_wires * kappa_mean * wire_A_m2 / wire_L_m * max(T_hot - T_cold, 0)
    P_rad  = emissivity * STEFAN_BOLTZMANN * rad_area_m2 * max(T_hot**4 - T_cold**4, 0)
    return P_wire, P_rad, P_qubit_W


# ── Lumped-RC ODE ─────────────────────────────────────────────────────────────

def _dr_ode(t, T, params: dict) -> np.ndarray:
    """
    Right-hand side of the 6-stage ODE.

    dT_i/dt = (Q_from_above_i - Q_to_below_i - P_cool_i + P_internal_i) / C_i

    Q_from_above_i = G_i * (T[i-1] - T[i])   bidirectional — can be negative
    Q_to_below_i   = G_{i+1} * (T[i] - T[i+1]) bidirectional — can be negative

    Sign convention:
      - P_cool removes heat from stage (PT pump / dilution unit) → subtract
      - Q_from_above > 0 when stage above is hotter → adds heat → add
      - Q_to_below   > 0 when current stage is hotter → removes heat → subtract
    """
    # Clamp temperatures to physical minimum
    T = np.maximum(T, 1e-10)

    dTdt = np.zeros(6)
    dTdt[0] = 0.0   # 300 K plate — fixed boundary

    # Effective conductance scaling from slider parameters
    n_w_scale  = params.get("n_wires", 24) / 24.0
    # Scale PT-stage conductances with wire count; sub-K stages are dominated
    # by the He-mixture condensate flow (not rescalable by wire count)
    if "G_override" in params:
        G = np.asarray(params["G_override"], dtype=float)
    else:
        G = _G0.copy()
        G[1] *= n_w_scale
        G[2] *= n_w_scale

    # Optional extra static loads (used for sensitivity studies)
    P_extra_4K = params.get("P_extra_4K", 0.0)   # W — extra load on 4K stage

    for i in range(1, 6):
        Q_from_above = G[i] * (T[i-1] - T[i])
        Q_to_below   = G[i+1] * (T[i] - T[i+1]) if i < 5 else 0.0
        Pcool        = cooling_power(i, T[i])
        P_qubit      = params.get("P_qubit_W", 5e-6) if i == 5 else 0.0
        P_extra      = P_extra_4K if i == 2 else 0.0   # extra load on 4K stage only
        C_i          = _C[i]

        dTdt[i] = (Q_from_above - Q_to_below - Pcool + P_qubit + P_extra) / C_i

    return dTdt


def simulate_cooldown(t_hours: float = 10.0,
                      T0: np.ndarray | None = None,
                      params: dict | None = None,
                      n_eval: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate the DR cool-down transient.

    Parameters
    ----------
    t_hours : float — total simulation time [hours]
    T0      : (6,) array of initial temperatures [K]
    params  : dict of heat-load overrides (n_wires, emissivity, P_qubit_W, …)
    n_eval  : number of output time points

    Returns
    -------
    t   : (N,) array — time [hours]
    T   : (N, 6) array — temperatures [K]
    """
    if T0 is None:
        T0 = np.array([300.0, 295.0, 290.0, 285.0, 280.0, 275.0])
    if params is None:
        params = {}

    t_span_s = (0.0, t_hours * 3600.0)
    t_eval_s = np.linspace(*t_span_s, n_eval)

    sol = solve_ivp(
        fun=lambda t, y: _dr_ode(t, y, params),
        t_span=t_span_s,
        y0=T0,
        method="Radau",
        t_eval=t_eval_s,
        rtol=1e-6,
        atol=np.array([1e-2, 1e-2, 1e-4, 1e-5, 1e-6, 1e-7]),
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    t_h = sol.t / 3600.0
    T   = sol.y.T          # (N, 6)
    return t_h, T


# ── Heat balance at steady state ──────────────────────────────────────────────

def heat_balance(T_final: np.ndarray, params: dict | None = None) -> list[dict]:
    """
    Compute heat balance at each stage given final temperatures.

    Returns list of dicts with keys: stage, T_K, P_cool_W, Q_net_W, balance_W
    """
    if params is None:
        params = {}

    n_w_scale = params.get("n_wires", 24) / 24.0
    G = _G0.copy()
    G[1] *= n_w_scale
    G[2] *= n_w_scale

    rows = []
    for i in range(1, 6):
        Q_from_above = G[i]   * (T_final[i-1] - T_final[i])
        Q_to_below   = G[i+1] * (T_final[i]   - T_final[i+1]) if i < 5 else 0.0
        Q_net        = Q_from_above - Q_to_below  # net heat into stage i
        P_qubit      = params.get("P_qubit_W", 5e-6) if i == 5 else 0.0
        Pcool        = cooling_power(i, T_final[i])
        rows.append({
            "stage":     STAGE_NAMES[i],
            "T_K":       T_final[i],
            "P_cool_W":  Pcool,
            "Q_in_W":    Q_net + P_qubit,
            "balance_W": Pcool - (Q_net + P_qubit),
        })
    return rows
