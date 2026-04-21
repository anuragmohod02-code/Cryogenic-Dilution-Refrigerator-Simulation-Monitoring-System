function [t, T_all] = thermal_model(t_span, T0, params)
% THERMAL_MODEL  Lumped RC ODE model of a dilution refrigerator cool-down.
%
%   [t, T_all] = thermal_model(t_span, T0, params)
%
%   Six temperature stages (indices 1-6):
%     1 = 300 K plate (room temperature — fixed boundary)
%     2 = 50 K stage
%     3 = 4 K stage
%     4 = Still (~700 mK)
%     5 = Cold Plate (~100 mK)
%     6 = MXC (~15 mK)
%
%   Governing ODE:
%     C_i * dT_i/dt = P_cool_i(T_i) - Q_in_i(T_{i-1}, T_i)
%
%   Inputs
%   ------
%   t_span : [t_start, t_end] in seconds
%   T0     : 6×1 initial temperatures [K]
%   params : struct (optional overrides for heat-load params)
%
%   Returns
%   -------
%   t     : time vector [s]
%   T_all : N×6 temperature matrix [K]

if nargin < 3
    params = struct();
end

% Thermal capacities C [J/K] per stage
% Sized so time-constants tau_i = C_i/G_i fit within 10-hour simulation
C = [1e5,   ...  % 300K plate  (large fixed boundary mass)
     500,    ...  % 50K stage   tau = 500/0.05  = 2.8 h
     30,     ...  % 4K stage    tau = 30/0.012  = 41 min
     3,      ...  % Still       tau = 3/0.003   = 17 min
     0.3,    ...  % Cold plate  tau = 0.3/5e-4  = 10 min
     0.03];       % MXC         tau = 0.03/1e-4 =  5 min

stage_keys = {'50K','4K','still','cold_plate','mxc'};
G_eff = [0, 0.05, 0.012, 0.003, 5e-4, 1e-4];

opts = odeset('RelTol',1e-6,'AbsTol',1e-9,'MaxStep',60);
[t, T_all] = ode45(@(t,T) dr_ode(t, T, C, G_eff, stage_keys), t_span, T0(:), opts);

end % thermal_model

% =========================================================================
function dTdt = dr_ode(~, T, C, G_eff, stage_keys)
% Internal ODE right-hand side

dTdt = zeros(6,1);

% Stage 1: 300 K — fixed boundary (large thermal mass)
dTdt(1) = 0;

% Stages 2-6: bidirectional heat exchange  dT_i/dt = (Q_in - Q_out - P_cool + P_qubit)/C_i
for i = 2:6
    Q_from_above = G_eff(i) * (T(i-1) - T(i));
    if i < 6
        Q_to_below = G_eff(i+1) * (T(i) - T(i+1));
    else
        Q_to_below = 0;
    end
    Pcool = cooling_power(stage_keys{i-1}, max(T(i), 1e-10));
    P_qubit = 0;
    if i == 6
        P_qubit = 5e-6;   % 5 µW qubit chip + coax cables
    end
    dTdt(i) = (Q_from_above - Q_to_below - Pcool + P_qubit) / C(i);
end

end % dr_ode
