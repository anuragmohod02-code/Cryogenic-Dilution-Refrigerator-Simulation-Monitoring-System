function [P_wire, P_rad, P_qubit] = heat_loads(T_hot, T_cold, stage_params)
% HEAT_LOADS  Compute heat loads into a cryogenic stage.
%
%   [P_wire, P_rad, P_qubit] = heat_loads(T_hot, T_cold, stage_params)
%
%   Inputs
%   ------
%   T_hot         : temperature of the warmer stage [K]
%   T_cold        : temperature of this stage [K]
%   stage_params  : struct with fields:
%       .n_wires      number of instrumentation wires
%       .wire_A_m2    cross-sectional area per wire [m^2]  (default 2e-8)
%       .wire_L_m     wire length between stages [m]       (default 0.3)
%       .rad_area_m2  radiative surface area [m^2]         (default 0.01)
%       .emissivity   surface emissivity                   (default 0.05)
%       .P_qubit_W    qubit chip dissipation [W]           (default 0)
%
%   Returns
%   -------
%   P_wire  : conductive heat load through wires [W]
%   P_rad   : radiative heat load [W]
%   P_qubit : qubit chip dissipation [W]

% ---- defaults ----
if ~isfield(stage_params,'wire_A_m2'),  stage_params.wire_A_m2  = 2e-8; end
if ~isfield(stage_params,'wire_L_m'),   stage_params.wire_L_m   = 0.3;  end
if ~isfield(stage_params,'rad_area_m2'),stage_params.rad_area_m2= 0.01; end
if ~isfield(stage_params,'emissivity'), stage_params.emissivity  = 0.05; end
if ~isfield(stage_params,'P_qubit_W'),  stage_params.P_qubit_W  = 0;    end
if ~isfield(stage_params,'n_wires'),    stage_params.n_wires     = 24;   end

% ---- Wire conduction (Wiedemann-Franz integral approximation) ----
% kappa(T) for NbTi/stainless: use mean conductivity ~ 0.1 W/(m*K) at cryo temps
kappa_mean = 0.1;   % W/(m·K)
P_wire = stage_params.n_wires ...
       * kappa_mean ...
       * stage_params.wire_A_m2 / stage_params.wire_L_m ...
       * (T_hot - T_cold);
P_wire = max(P_wire, 0);

% ---- Radiation (Stefan-Boltzmann) ----
sigma = 5.6704e-8;   % W/(m^2·K^4)
P_rad = stage_params.emissivity ...
      * sigma ...
      * stage_params.rad_area_m2 ...
      * (T_hot^4 - T_cold^4);
P_rad = max(P_rad, 0);

% ---- Qubit dissipation ----
P_qubit = stage_params.P_qubit_W;
end
