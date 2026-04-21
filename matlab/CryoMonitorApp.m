classdef CryoMonitorApp < handle
%CRYOMONITORAPP  MATLAB GUI app for dilution refrigerator monitoring.
%
%   Provides three tabs:
%     Tab 1 — Live temperature gauges (color-coded labels per stage)
%     Tab 2 — Cool-down transient plot (log-scale)
%     Tab 3 — Heat load parameter sliders
%
%   Usage:
%       app = CryoMonitorApp();
%
%   Requires:  thermal_model.m, cooling_power.m, heat_loads.m in same folder.

    properties (Access = private)
        % Window
        UIFigure        matlab.ui.Figure
        GridLayout      matlab.ui.container.GridLayout

        % Top controls
        RunButton       matlab.ui.control.Button
        ExportButton    matlab.ui.control.Button
        StatusLabel     matlab.ui.control.Label

        % Tab group
        TabGroup        matlab.ui.container.TabGroup
        Tab1            matlab.ui.container.Tab
        Tab2            matlab.ui.container.Tab
        Tab3            matlab.ui.container.Tab

        % Tab 1 — Temperature gauges
        GaugeGrid       matlab.ui.container.GridLayout
        StageLabels     matlab.ui.control.Label    % 6×1
        TempLabels      matlab.ui.control.Label    % 6×1

        % Tab 2 — Plot
        CooldownAxes    matlab.ui.control.UIAxes

        % Tab 3 — Sliders
        SliderGrid      matlab.ui.container.GridLayout
        NWiresSlider    matlab.ui.control.Slider
        EmissSlider     matlab.ui.control.Slider
        QubitPowSlider  matlab.ui.control.Slider
        NWiresLabel     matlab.ui.control.Label
        EmissLabel      matlab.ui.control.Label
        QubitPowLabel   matlab.ui.control.Label

        % Simulation data
        t_sim           double
        T_sim           double   % N×6
    end

    properties (Constant, Access = private)
        STAGE_NAMES = {'300 K plate','50 K stage','4 K stage','Still','Cold Plate','MXC'}
        STAGE_TARGETS = [300, 50, 4, 0.7, 0.1, 0.015]   % K
        COLORS_GOOD = '#2ECC71'    % green
        COLORS_WARN = '#F39C12'    % amber
        COLORS_BAD  = '#E74C3C'    % red
    end

    % =====================================================================
    methods (Access = public)
        function app = CryoMonitorApp()
            buildComponents(app);
            if nargout == 0
                clear app
            end
        end

        function delete(app)
            if isvalid(app.UIFigure)
                delete(app.UIFigure);
            end
        end
    end

    % =====================================================================
    methods (Access = private)

        function buildComponents(app)
            %% Figure
            app.UIFigure = uifigure('Name','Cryo DR Monitor','Position',[200 100 900 650]);
            app.UIFigure.Color = '#1A1A2E';

            %% Outer grid: top bar + tabs
            app.GridLayout = uigridlayout(app.UIFigure, [2, 3]);
            app.GridLayout.RowHeight   = {50, '1x'};
            app.GridLayout.ColumnWidth = {'fit','fit','1x'};
            app.GridLayout.BackgroundColor = '#1A1A2E';
            app.GridLayout.Padding = [10 10 10 10];

            %% Top bar buttons
            app.RunButton = uibutton(app.GridLayout, 'Text','▶  Run Simulation', ...
                'FontSize',14,'FontWeight','bold', ...
                'BackgroundColor','#0F3460','FontColor','white', ...
                'ButtonPushedFcn', @(~,~) runSimulation(app));
            app.RunButton.Layout.Row = 1; app.RunButton.Layout.Column = 1;

            app.ExportButton = uibutton(app.GridLayout, 'Text','💾  Export CSV', ...
                'FontSize',14,'FontWeight','bold', ...
                'BackgroundColor','#16213E','FontColor','#A0AEC0', ...
                'ButtonPushedFcn', @(~,~) exportCSV(app));
            app.ExportButton.Layout.Row = 1; app.ExportButton.Layout.Column = 2;

            app.StatusLabel = uilabel(app.GridLayout, ...
                'Text','Ready — press Run Simulation', ...
                'FontSize',12,'FontColor','#A0AEC0','HorizontalAlignment','left');
            app.StatusLabel.Layout.Row = 1; app.StatusLabel.Layout.Column = 3;

            %% Tab group
            app.TabGroup = uitabgroup(app.GridLayout);
            app.TabGroup.Layout.Row = 2; app.TabGroup.Layout.Column = [1,3];

            app.Tab1 = uitab(app.TabGroup, 'Title','🌡  Temperature Gauges');
            app.Tab2 = uitab(app.TabGroup, 'Title','📈  Cool-down Curve');
            app.Tab3 = uitab(app.TabGroup, 'Title','⚙  Heat Load Params');

            buildTab1(app);
            buildTab2(app);
            buildTab3(app);
        end

        % ---- Tab 1: gauges ----
        function buildTab1(app)
            g = uigridlayout(app.Tab1, [7, 3]);
            g.RowHeight = {30, '1x','1x','1x','1x','1x','1x'};
            g.ColumnWidth = {'fit','fit','1x'};
            g.BackgroundColor = '#1A1A2E';
            g.Padding = [20 20 20 20];
            app.GaugeGrid = g;

            % Header
            hdr1 = uilabel(g,'Text','Stage','FontWeight','bold','FontColor','#A0AEC0','FontSize',13);
            hdr1.Layout.Row=1; hdr1.Layout.Column=1;
            hdr2 = uilabel(g,'Text','Temperature','FontWeight','bold','FontColor','#A0AEC0','FontSize',13);
            hdr2.Layout.Row=1; hdr2.Layout.Column=2;

            app.StageLabels = matlab.ui.control.Label.empty;
            app.TempLabels  = matlab.ui.control.Label.empty;

            for k = 1:6
                sl = uilabel(g,'Text',app.STAGE_NAMES{k}, ...
                    'FontSize',15,'FontWeight','bold','FontColor','white');
                sl.Layout.Row=k+1; sl.Layout.Column=1;

                tl = uilabel(g,'Text','--- K', ...
                    'FontSize',20,'FontWeight','bold','FontColor','#A0AEC0', ...
                    'BackgroundColor','#16213E','HorizontalAlignment','center');
                tl.Layout.Row=k+1; tl.Layout.Column=2;

                app.StageLabels(k) = sl;
                app.TempLabels(k)  = tl;
            end
        end

        % ---- Tab 2: axes ----
        function buildTab2(app)
            g = uigridlayout(app.Tab2,[1,1]);
            g.BackgroundColor = '#1A1A2E';
            app.CooldownAxes = uiaxes(g);
            app.CooldownAxes.Color = '#16213E';
            app.CooldownAxes.XColor = 'white';
            app.CooldownAxes.YColor = 'white';
            app.CooldownAxes.Title.String = 'DR Cool-down Transient';
            app.CooldownAxes.Title.Color  = 'white';
            app.CooldownAxes.XLabel.String = 'Time (hours)';
            app.CooldownAxes.YLabel.String = 'Temperature (K) [log]';
            app.CooldownAxes.XLabel.Color  = 'white';
            app.CooldownAxes.YLabel.Color  = 'white';
            grid(app.CooldownAxes,'on');
        end

        % ---- Tab 3: sliders ----
        function buildTab3(app)
            g = uigridlayout(app.Tab3, [4,3]);
            g.RowHeight    = {30,'fit','fit','fit'};
            g.ColumnWidth  = {'fit','1x','fit'};
            g.BackgroundColor = '#1A1A2E';
            g.Padding = [20 20 20 20];
            app.SliderGrid = g;

            hdr = uilabel(g,'Text','Adjust heat-load parameters and re-run:', ...
                'FontColor','#A0AEC0','FontSize',12);
            hdr.Layout.Row=1; hdr.Layout.Column=[1,3];

            % N wires
            lbl1 = uilabel(g,'Text','# Wires: 24','FontColor','white','FontSize',13);
            lbl1.Layout.Row=2; lbl1.Layout.Column=1;
            app.NWiresLabel = lbl1;
            s1 = uislider(g,'Limits',[4 96],'Value',24, ...
                'ValueChangedFcn',@(src,~) updateSliderLabel(app,'nwires',src.Value));
            s1.Layout.Row=2; s1.Layout.Column=2;
            app.NWiresSlider = s1;

            % Emissivity
            lbl2 = uilabel(g,'Text','Emissivity: 0.050','FontColor','white','FontSize',13);
            lbl2.Layout.Row=3; lbl2.Layout.Column=1;
            app.EmissLabel = lbl2;
            s2 = uislider(g,'Limits',[0.001 0.2],'Value',0.05, ...
                'ValueChangedFcn',@(src,~) updateSliderLabel(app,'emiss',src.Value));
            s2.Layout.Row=3; s2.Layout.Column=2;
            app.EmissSlider = s2;

            % Qubit power
            lbl3 = uilabel(g,'Text','Qubit power: 100 nW','FontColor','white','FontSize',13);
            lbl3.Layout.Row=4; lbl3.Layout.Column=1;
            app.QubitPowLabel = lbl3;
            s3 = uislider(g,'Limits',[0 1000],'Value',100, ...
                'ValueChangedFcn',@(src,~) updateSliderLabel(app,'qubit',src.Value));
            s3.Layout.Row=4; s3.Layout.Column=2;
            app.QubitPowSlider = s3;
        end

        % ---- Callbacks ----
        function updateSliderLabel(app, which, val)
            switch which
                case 'nwires'
                    app.NWiresLabel.Text = sprintf('# Wires: %d', round(val));
                case 'emiss'
                    app.EmissLabel.Text = sprintf('Emissivity: %.3f', val);
                case 'qubit'
                    app.QubitPowLabel.Text = sprintf('Qubit power: %.0f nW', val);
            end
        end

        function runSimulation(app)
            app.StatusLabel.Text = 'Running simulation...';
            app.StatusLabel.FontColor = '#F39C12';
            drawnow;

            T0 = [300; 290; 285; 280; 275; 270];
            params = struct();
            try
                [t, T] = thermal_model([0, 10*3600], T0, params);
                app.t_sim = t;
                app.T_sim = T;
                updateGauges(app);
                updatePlot(app);
                app.StatusLabel.Text = sprintf('Done — MXC = %.2f mK', T(end,6)*1e3);
                app.StatusLabel.FontColor = app.COLORS_GOOD;
            catch ME
                app.StatusLabel.Text = ['Error: ' ME.message];
                app.StatusLabel.FontColor = app.COLORS_BAD;
            end
        end

        function updateGauges(app)
            T_final = app.T_sim(end,:);
            for k = 1:6
                T = T_final(k);
                target = app.STAGE_TARGETS(k);
                if T < target * 1.5
                    c = app.COLORS_GOOD;
                elseif T < target * 5
                    c = app.COLORS_WARN;
                else
                    c = app.COLORS_BAD;
                end
                if T >= 1
                    app.TempLabels(k).Text = sprintf('%.2f K', T);
                else
                    app.TempLabels(k).Text = sprintf('%.1f mK', T*1e3);
                end
                app.TempLabels(k).FontColor = c;
            end
        end

        function updatePlot(app)
            ax = app.CooldownAxes;
            cla(ax);
            t_h = app.t_sim / 3600;
            colors = lines(6);
            hold(ax,'on');
            for k = 1:6
                semilogy(ax, t_h, app.T_sim(:,k), 'LineWidth', 2, 'Color', colors(k,:));
            end
            hold(ax,'off');
            legend(ax, app.STAGE_NAMES, 'Location','northeast', 'TextColor','white');
            ax.YScale = 'log';
            ax.YLim = [1e-3, 400];
        end

        function exportCSV(app)
            if isempty(app.t_sim)
                uialert(app.UIFigure, 'Run simulation first.', 'No data');
                return;
            end
            [file, path] = uiputfile('*.csv','Save CSV','stage_temperatures.csv');
            if isequal(file,0), return; end
            csv_path = fullfile(path, file);
            headers = 'time_h,T_300K,T_50K,T_4K,T_still,T_cold_plate,T_mxc';
            fid = fopen(csv_path,'w');
            fprintf(fid,'%s\n',headers);
            fclose(fid);
            dlmwrite(csv_path, [app.t_sim/3600, app.T_sim], '-append','precision','%.6g');
            uialert(app.UIFigure, sprintf('Saved to:\n%s',csv_path),'Export Done','Icon','success');
        end
    end
end
