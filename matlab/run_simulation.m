% RUN_SIMULATION  Master script — run DR cool-down simulation, export results.
%
%   Runs the full 6-stage thermal ODE model, generates plots, and exports
%   a CSV suitable for the Python Dash dashboard.
%
%   Usage:
%       run_simulation          % uses defaults
%
%   Outputs written to ../outputs/ :
%       cooldown_curve.png
%       stage_temperatures.csv

clear; clc;

%% ---- Simulation parameters ----
t_hours   = 10;                    % total cool-down time [hours]
t_span    = [0, t_hours * 3600];   % convert to seconds

% Initial temperatures: everything near room temperature
T0 = [300; 290; 285; 280; 275; 270];   % [K]

fprintf('Starting 6-stage DR thermal simulation (%.1f hours)...\n', t_hours);
tic;

%% ---- Run ODE solver ----
[t, T_all] = thermal_model(t_span, T0);

elapsed = toc;
fprintf('Simulation completed in %.2f s (%d time points).\n', elapsed, length(t));

%% ---- Steady-state report ----
stage_names = {'300 K plate', '50 K stage', '4 K stage', ...
               'Still', 'Cold Plate', 'MXC'};
T_final = T_all(end,:);

fprintf('\n--- Steady-State Temperatures ---\n');
for k = 1:6
    fprintf('  %-15s : %10.4g K\n', stage_names{k}, T_final(k));
end
fprintf('\n');

%% ---- Export CSV ----
out_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'outputs');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

t_hours_vec = t / 3600;
csv_data = [t_hours_vec, T_all];
header = 'time_h,T_300K,T_50K,T_4K,T_still,T_cold_plate,T_mxc';

csv_path = fullfile(out_dir, 'stage_temperatures.csv');
fid = fopen(csv_path, 'w');
fprintf(fid, '%s\n', header);
fclose(fid);
dlmwrite(csv_path, csv_data, '-append', 'precision', '%.6g');
fprintf('CSV written: %s\n', csv_path);

%% ---- Plot cool-down curves ----
colors = lines(6);
fig = figure('Visible','off','Position',[100 100 900 600]);

% Log-scale temperature plot
subplot(2,1,1);
hold on;
for k = 1:6
    semilogy(t_hours_vec, T_all(:,k), 'LineWidth', 1.8, 'Color', colors(k,:));
end
xlabel('Time (hours)');
ylabel('Temperature (K)  [log scale]');
title('Dilution Refrigerator Cool-down — All Stages');
legend(stage_names, 'Location','northeast');
grid on;
ylim([1e-3, 400]);
hold off;

% Sub-K stages zoom
subplot(2,1,2);
hold on;
sub_K = [4,5,6];
for k = sub_K
    semilogy(t_hours_vec, T_all(:,k)*1e3, 'LineWidth', 1.8, 'Color', colors(k,:));
end
xlabel('Time (hours)');
ylabel('Temperature (mK)  [log scale]');
title('Sub-Kelvin Stages Zoom');
legend(stage_names(sub_K), 'Location','northeast');
grid on;
hold off;

png_path = fullfile(out_dir, 'cooldown_curve.png');
exportgraphics(fig, png_path, 'Resolution', 150);
fprintf('Plot saved: %s\n', png_path);

%% ---- Heat balance table ----
fprintf('\n--- Heat Balance at Steady State ---\n');
fprintf('  %-15s  %10s  %10s  %10s\n', 'Stage','P_cool(W)','Q_in(W)','Balance(W)');
wp = struct('n_wires',24,'wire_A_m2',2e-8,'wire_L_m',0.3,'rad_area_m2',0.01,'emissivity',0.05,'P_qubit_W',0);
stage_keys = {'50K','4K','still','cold_plate','mxc'};
T_prev = T_final(1);
for k = 2:6
    Pc = cooling_power(stage_keys{k-1}, T_final(k));
    [Pw,Pr,Pq] = heat_loads(T_prev, T_final(k), wp);
    Qi = Pw + Pr + Pq;
    fprintf('  %-15s  %10.3g  %10.3g  %+10.3g\n', stage_names{k}, Pc, Qi, Pc-Qi);
    T_prev = T_final(k);
end
