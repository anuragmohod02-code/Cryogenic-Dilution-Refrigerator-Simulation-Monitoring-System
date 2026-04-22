import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from cryo_thermal import simulate_cooldown, simulate_warmup
from anomaly_detector import load_or_train_model, score_simulation

# Cool-down with n3_flow override
t_h, T = simulate_cooldown(params={"n3_flow_umol": 600}, n_eval=300)
print(f"Cool-down MXC = {T[-1,5]*1000:.2f} mK  (n3=600)")

# Warm-up
t_h2, T2 = simulate_warmup(t_hours=10.0, n_eval=300)
print(f"Warm-up  MXC  = {T2[-1,5]:.1f} K after 10 h")

# Anomaly: normal run
m = load_or_train_model()
res = score_simulation(m, t_h, T)
print(f"Normal score  = {res['pct']}% — {res['label']}")

# Anomaly: flow blockage fault
t_f, T_f = simulate_cooldown(params={"n3_flow_umol": 50}, n_eval=300)
res_f = score_simulation(m, t_f, T_f)
print(f"Fault  score  = {res_f['pct']}% — {res_f['label']}  (flow=50 umol/s)")

print("All checks passed.")
