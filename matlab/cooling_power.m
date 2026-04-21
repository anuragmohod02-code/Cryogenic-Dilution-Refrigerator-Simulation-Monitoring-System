function P_cool = cooling_power(stage, T)
% COOLING_POWER  Cooling power of each DR stage as a function of temperature.
%
%   P_cool = cooling_power(stage, T)
%
%   Inputs
%   ------
%   stage : string — one of '50K','4K','still','cold_plate','mxc'
%   T     : numeric array — temperature values [K]
%
%   Returns
%   -------
%   P_cool : cooling power [W]  (same size as T)
%
%   Based on Oxford Instruments Triton 400 dilution refrigerator specs.
%   Analytical fits to published cool-down curves.

switch lower(stage)
    case '50k'
        % Pulse tube 1st stage: heat-pump law  P = 40*(1 - 30/T)  W
        % Oxford Triton 400: ~40 W at 50 K, base temperature ~30 K
        T_base = 30.0;
        P_cool = max(40.0 .* (1.0 - T_base ./ max(T, 1e-10)), 0.0);

    case '4k'
        % Pulse tube 2nd stage: heat-pump law  P = 1.5*(1 - 2.5/T)  W
        % Oxford Triton 400: ~1.5 W at 4 K, base temperature ~2.5 K
        T_base = 2.5;
        P_cool = max(1.5 .* (1.0 - T_base ./ max(T, 1e-10)), 0.0);

    case 'still'
        % Still (dilution unit): T² law  P = 40.8e-3 * T²  W
        % A so P(0.7 K) = 20 mW.  No upper cutoff: drives stage from ~4K down.
        P_cool = 40.8e-3 .* max(T, 1e-10).^2;

    case 'cold_plate'
        % Cold plate: T² law  P = 0.2 * T²  W
        % A so P(0.1 K) = 2 mW.  No upper cutoff.
        P_cool = 0.2 .* max(T, 1e-10).^2;

    case 'mxc'
        % Mixing chamber: T² law  P = 0.04 * T²  W
        % A so P(0.1 K) = 400 µW (Oxford Triton 400).  No upper cutoff.
        P_cool = 0.04 .* max(T, 1e-10).^2;

    otherwise
        error('cooling_power: unknown stage "%s"', stage);
end
end
