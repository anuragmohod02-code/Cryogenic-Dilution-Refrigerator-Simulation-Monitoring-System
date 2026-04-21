"""
dashboard.py
------------
Plotly Dash web dashboard for the cryogenic dilution refrigerator simulation.
Reads outputs/stage_temperatures.csv or runs the simulation live.

New in v2:
  • Plotly-native animated cool-down playback with Play/Pause
  • ³He circulation rate slider (scales sub-K cooling power)
  • Warm-up mode simulation toggle
  • Export CSV download button
  • ML anomaly-detection score card (Isolation Forest)

Usage:
    python dashboard.py
    → Open http://localhost:8050 in a browser
"""

import os
import sys
import threading

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc

from cryo_thermal import (
    simulate_cooldown, simulate_warmup, heat_balance,
    STAGE_NAMES, STAGE_TARGETS_K,
)

# ── File path ─────────────────────────────────────────────────────────────────
OUT_DIR  = os.path.join(os.path.dirname(script_dir), "outputs")
CSV_PATH = os.path.join(OUT_DIR, "stage_temperatures.csv")

# ── Colour palette ────────────────────────────────────────────────────────────
STAGE_COLORS = [
    "#E74C3C",   # 300K  red
    "#E67E22",   # 50K   orange
    "#F1C40F",   # 4K    yellow
    "#2ECC71",   # Still green
    "#3498DB",   # CP    blue
    "#9B59B6",   # MXC   purple
]
BG_DARK  = "#1A1A2E"
BG_CARD  = "#16213E"
TEXT_CLR = "#E2E8F0"

# ── Anomaly detector (loaded in background thread) ────────────────────────────
_anomaly_model: dict | None = None
_anomaly_ready = threading.Event()

def _load_anomaly_model_bg():
    global _anomaly_model
    try:
        from anomaly_detector import load_or_train_model
        _anomaly_model = load_or_train_model()
        print("Anomaly detection model ready.")
    except Exception as e:
        print(f"Anomaly detection unavailable: {e}")
    finally:
        _anomaly_ready.set()

threading.Thread(target=_load_anomaly_model_bg, daemon=True).start()

# ── Load or generate data ─────────────────────────────────────────────────────

def load_or_simulate() -> pd.DataFrame:
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        t_h, T = simulate_cooldown()
        df = pd.DataFrame(
            np.column_stack([t_h, T]),
            columns=["time_h","T_300K","T_50K","T_4K","T_still","T_cold_plate","T_mxc"],
        )
    return df


df_global = load_or_simulate()

# ── Gauge builder ─────────────────────────────────────────────────────────────

def make_gauge(title: str, value_K: float, target_K: float, color: str) -> go.Figure:
    ratio = value_K / target_K
    if ratio < 1.5:
        bar_color = "#2ECC71"
    elif ratio < 5:
        bar_color = "#F39C12"
    else:
        bar_color = "#E74C3C"

    if value_K >= 1:
        display = f"{value_K:.2f} K"
    else:
        display = f"{value_K*1e3:.1f} mK"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value_K * 1e3 if value_K < 1 else value_K,
        number={"suffix": " mK" if value_K < 1 else " K",
                "font": {"color": TEXT_CLR, "size": 18}},
        title={"text": title, "font": {"color": TEXT_CLR, "size": 13}},
        gauge={
            "axis": {
                "range": [0, target_K * 10 * 1e3 if target_K < 1 else target_K * 10],
                "tickfont": {"color": TEXT_CLR},
            },
            "bar":  {"color": bar_color},
            "bgcolor": BG_CARD,
            "bordercolor": "#4A5568",
            "threshold": {
                "line": {"color": "#FFD700", "width": 2},
                "thickness": 0.75,
                "value": target_K * 1e3 if target_K < 1 else target_K,
            },
        },
        delta={
            "reference": target_K * 1e3 if target_K < 1 else target_K,
            "increasing": {"color": "#E74C3C"},
            "decreasing": {"color": "#2ECC71"},
        },
    ))
    fig.update_layout(
        paper_bgcolor=BG_CARD,
        font_color=TEXT_CLR,
        height=200,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


def make_cooldown_plot(df: pd.DataFrame) -> go.Figure:
    t = df["time_h"].values
    T_cols = ["T_300K","T_50K","T_4K","T_still","T_cold_plate","T_mxc"]
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("All Stages", "Sub-Kelvin Zoom"),
                        vertical_spacing=0.15)
    for i, col in enumerate(T_cols):
        y = df[col].values
        fig.add_trace(
            go.Scatter(x=t, y=y, name=STAGE_NAMES[i],
                       line=dict(color=STAGE_COLORS[i], width=2.5),
                       mode="lines"),
            row=1, col=1,
        )
    for i in [3, 4, 5]:
        col = T_cols[i]
        fig.add_trace(
            go.Scatter(x=t, y=df[col].values * 1e3,
                       name=STAGE_NAMES[i] + " (mK)",
                       line=dict(color=STAGE_COLORS[i], width=2.5),
                       mode="lines", showlegend=False),
            row=2, col=1,
        )
    fig.update_yaxes(type="log", title_text="Temperature (K)", row=1, col=1)
    fig.update_yaxes(type="log", title_text="Temperature (mK)", row=2, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
    fig.update_layout(
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_CARD,
        font_color=TEXT_CLR,
        legend=dict(bgcolor=BG_CARD, bordercolor="#4A5568", borderwidth=1),
        height=600,
        margin=dict(l=50, r=20, t=60, b=40),
    )
    fig.update_xaxes(gridcolor="#2D3748", zerolinecolor="#4A5568")
    fig.update_yaxes(gridcolor="#2D3748", zerolinecolor="#4A5568")
    return fig


def make_heat_balance_bar(T_final: np.ndarray) -> go.Figure:
    rows = heat_balance(T_final)
    stages  = [r["stage"] for r in rows]
    p_cool  = [r["P_cool_W"] * 1e6 for r in rows]   # µW
    q_in    = [r["Q_in_W"]   * 1e6 for r in rows]
    fig = go.Figure()
    fig.add_bar(name="Cooling Power", x=stages, y=p_cool,
                marker_color="#2ECC71")
    fig.add_bar(name="Heat Load In", x=stages, y=q_in,
                marker_color="#E74C3C")
    fig.update_layout(
        barmode="group",
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_CARD,
        font_color=TEXT_CLR,
        xaxis=dict(gridcolor="#2D3748"),
        yaxis=dict(gridcolor="#2D3748", title="Power (µW) [log]", type="log"),
        legend=dict(bgcolor=BG_CARD),
        height=350,
        margin=dict(l=50, r=20, t=40, b=40),
        title="Heat Balance per Stage at Steady State",
    )
    return fig


def make_animated_cooldown(df: pd.DataFrame, n_frames: int = 40) -> go.Figure:
    """
    Plotly-native animated cool-down figure with Play/Pause and time slider.
    Animation is fully client-side — no server round-trips per frame.
    """
    t      = df["time_h"].values
    T_cols = ["T_300K","T_50K","T_4K","T_still","T_cold_plate","T_mxc"]
    N      = len(t)
    ends   = np.linspace(1, N - 1, n_frames, dtype=int)

    # Initial state: show data up to first frame end
    init = [
        go.Scatter(x=t[:ends[0]], y=df[col].values[:ends[0]],
                   name=STAGE_NAMES[i],
                   line=dict(color=STAGE_COLORS[i], width=2.5), mode="lines")
        for i, col in enumerate(T_cols)
    ]
    fig = go.Figure(data=init)

    # Animation frames (each replaces all 6 traces)
    frames, slider_steps = [], []
    for fi, idx in enumerate(ends):
        frame_data = [
            go.Scatter(x=t[:idx+1], y=df[col].values[:idx+1])
            for col in T_cols
        ]
        frames.append(go.Frame(data=frame_data, name=str(fi)))
        slider_steps.append({
            "args":   [[str(fi)], {"frame":      {"duration": 80, "redraw": True},
                                   "mode":       "immediate",
                                   "transition": {"duration": 0}}],
            "label":  f"{t[idx]:.1f} h",
            "method": "animate",
        })

    fig.frames = frames

    fig.update_layout(
        paper_bgcolor=BG_DARK, plot_bgcolor=BG_CARD, font_color=TEXT_CLR,
        yaxis=dict(type="log", title="Temperature (K)",
                   range=[-2.5, 2.5], gridcolor="#2D3748"),
        xaxis=dict(title="Time (hours)", gridcolor="#2D3748"),
        height=480,
        margin=dict(l=60, r=20, t=70, b=90),
        legend=dict(bgcolor=BG_CARD, bordercolor="#4A5568", borderwidth=1),
        title=dict(text="Cool-down Transient — Animated (▶ Play below)",
                   font=dict(color=TEXT_CLR, size=14)),
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 80, "redraw": True},
                                 "fromcurrent": True, "transition": {"duration": 0}}],
                 "label": "▶  Play", "method": "animate"},
                {"args": [[None], {"mode": "immediate", "transition": {"duration": 0},
                                   "frame": {"duration": 0, "redraw": True}}],
                 "label": "⏸  Pause", "method": "animate"},
            ],
            "direction": "left", "pad": {"r": 10, "t": 8},
            "showactive": True, "type": "buttons",
            "x": 0.0, "xanchor": "left", "y": 1.14, "yanchor": "top",
            "bgcolor": "#0F3460", "font": {"color": TEXT_CLR, "size": 13},
            "bordercolor": "#4A5568",
        }],
        sliders=[{
            "active": n_frames - 1,
            "yanchor": "top", "xanchor": "left",
            "currentvalue": {"font": {"size": 12, "color": TEXT_CLR},
                             "prefix": "t = ", "visible": True, "xanchor": "right"},
            "transition": {"duration": 0},
            "pad": {"b": 10, "t": 55}, "len": 0.9, "x": 0.1, "y": 0,
            "steps": slider_steps,
            "bgcolor": "#0F3460", "bordercolor": "#4A5568",
            "font": {"color": TEXT_CLR},
        }],
    )
    return fig


def make_subkelvin_plot(df: pd.DataFrame) -> go.Figure:
    """Sub-Kelvin stages detail on mK scale."""
    t = df["time_h"].values
    fig = go.Figure()
    for i, (col, name, color) in enumerate(zip(
            ["T_still", "T_cold_plate", "T_mxc"],
            [STAGE_NAMES[3], STAGE_NAMES[4], STAGE_NAMES[5]],
            [STAGE_COLORS[3], STAGE_COLORS[4], STAGE_COLORS[5]])):
        fig.add_trace(go.Scatter(x=t, y=df[col].values * 1e3,
                                 name=name, line=dict(color=color, width=2.5),
                                 mode="lines"))
    fig.add_hline(y=15, line=dict(color="#FFD700", dash="dash", width=1.5),
                  annotation_text="MXC target 15 mK",
                  annotation_font_color="#FFD700")
    fig.update_layout(
        paper_bgcolor=BG_DARK, plot_bgcolor=BG_CARD, font_color=TEXT_CLR,
        yaxis=dict(type="log", title="Temperature (mK)", gridcolor="#2D3748"),
        xaxis=dict(title="Time (hours)", gridcolor="#2D3748"),
        legend=dict(bgcolor=BG_CARD, bordercolor="#4A5568", borderwidth=1),
        height=340, margin=dict(l=60, r=20, t=50, b=40),
        title=dict(text="Sub-Kelvin Stage Detail", font=dict(color=TEXT_CLR)),
    )
    return fig


# ── App layout ────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="Cryo DR Monitor v2",
)

def build_layout():
    df = df_global
    T_final = df.iloc[-1][["T_300K","T_50K","T_4K","T_still","T_cold_plate","T_mxc"]].values

    gauge_cards = []
    for i in range(6):
        gauge_cards.append(
            dbc.Col(
                dbc.Card([
                    dbc.CardBody(
                        dcc.Graph(
                            id=f"gauge-{i}",
                            figure=make_gauge(
                                STAGE_NAMES[i], T_final[i],
                                STAGE_TARGETS_K[i], STAGE_COLORS[i],
                            ),
                            config={"displayModeBar": False},
                        )
                    )
                ], style={"backgroundColor": BG_CARD, "border": "1px solid #2D3748"}),
                width=2, style={"padding": "4px"},
            )
        )

    card_style = {"backgroundColor": BG_CARD, "border": "1px solid #2D3748",
                  "margin": "10px 0"}

    layout = dbc.Container([
        # ── Header ────────────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(html.H2("🧊 Cryogenic Dilution Refrigerator Monitor",
                            style={"color": "#63B3ED", "marginTop": "16px"})),
            dbc.Col([
                dbc.Button("▶ Run Simulation", id="run-btn",
                           color="primary", size="sm",
                           style={"marginTop": "20px", "marginRight": "8px"}),
                dbc.Button("⬇ Export CSV", id="export-btn",
                           color="secondary", size="sm",
                           style={"marginTop": "20px"}),
                dcc.Download(id="download-csv"),
            ], width=4, style={"textAlign": "right"}),
        ], style={"backgroundColor": BG_DARK, "padding": "0 20px"}),

        html.Hr(style={"borderColor": "#2D3748"}),

        # ── Slider controls ────────────────────────────────────────────────────
        dbc.Card([
            dbc.CardHeader("Simulation Parameters",
                           style={"color": TEXT_CLR, "backgroundColor": "#0F3460"}),
            dbc.CardBody([
                # Row 1: heat loads
                dbc.Row([
                    dbc.Col([
                        html.Label("# Wires (thermal conduction)",
                                   style={"color": TEXT_CLR, "fontSize": "0.85rem"}),
                        dcc.Slider(id="sl-nwires", min=4, max=96, step=4,
                                   value=24,
                                   marks={4:"4", 24:"24", 48:"48", 96:"96"},
                                   tooltip={"placement":"bottom"}),
                    ], width=4),
                    dbc.Col([
                        html.Label("Shield emissivity",
                                   style={"color": TEXT_CLR, "fontSize": "0.85rem"}),
                        dcc.Slider(id="sl-emiss", min=0.001, max=0.2, step=0.005,
                                   value=0.05,
                                   marks={0.001:"0.001", 0.05:"0.05", 0.2:"0.2"},
                                   tooltip={"placement":"bottom"}),
                    ], width=4),
                    dbc.Col([
                        html.Label("Qubit dissipation (nW)",
                                   style={"color": TEXT_CLR, "fontSize": "0.85rem"}),
                        dcc.Slider(id="sl-qubit", min=0, max=1000, step=10,
                                   value=100,
                                   marks={0:"0", 100:"100", 500:"500", 1000:"1000"},
                                   tooltip={"placement":"bottom"}),
                    ], width=4),
                ], style={"marginBottom": "10px"}),
                # Row 2: ³He flow + warm-up toggle
                dbc.Row([
                    dbc.Col([
                        html.Label("³He circulation rate (µmol/s)",
                                   style={"color": TEXT_CLR, "fontSize": "0.85rem"}),
                        dcc.Slider(id="sl-n3flow", min=100, max=800, step=20,
                                   value=476,
                                   marks={100:"100", 476:"476 ★", 800:"800"},
                                   tooltip={"placement":"bottom"}),
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.Label("Warm-up simulation",
                                       style={"color": TEXT_CLR, "fontSize": "0.85rem",
                                              "marginRight": "10px"}),
                            dbc.Switch(id="sw-warmup", value=False,
                                       label="",
                                       style={"display": "inline-block"}),
                        ], style={"marginTop": "22px"}),
                    ], width=3),
                    dbc.Col([
                        html.Div(id="anomaly-card",
                                 children="Anomaly detection: loading…",
                                 style={"color": "#F39C12", "fontFamily": "monospace",
                                        "fontSize": "0.85rem", "marginTop": "20px",
                                        "padding": "6px 10px",
                                        "border": "1px solid #4A5568",
                                        "borderRadius": "6px",
                                        "backgroundColor": BG_DARK}),
                    ], width=3),
                ]),
            ], style={"backgroundColor": BG_CARD}),
        ], style={"margin": "10px 0"}),

        # ── Status bar ─────────────────────────────────────────────────────────
        html.Div(id="status-bar",
                 children=f"Loaded data — MXC = {T_final[5]*1e3:.2f} mK",
                 style={"color": "#2ECC71", "padding": "6px 0",
                        "fontFamily": "monospace"}),

        # ── Temperature gauges ─────────────────────────────────────────────────
        dbc.Row(gauge_cards, style={"margin": "8px 0"}),

        # ── Animated cool-down plot ────────────────────────────────────────────
        dbc.Card([
            dbc.CardBody(dcc.Graph(id="cooldown-plot",
                                   figure=make_animated_cooldown(df),
                                   config={"displayModeBar": True}))
        ], style=card_style),

        # ── Sub-K zoom + heat balance (side by side) ───────────────────────────
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody(dcc.Graph(id="subkelvin-plot",
                                           figure=make_subkelvin_plot(df),
                                           config={"displayModeBar": False}))
                ], style=card_style),
                width=6,
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardBody(dcc.Graph(id="heat-balance-plot",
                                           figure=make_heat_balance_bar(T_final),
                                           config={"displayModeBar": False}))
                ], style=card_style),
                width=6,
            ),
        ]),

        # ── Hidden stores ──────────────────────────────────────────────────────
        dcc.Store(id="sim-data-store"),

    ], fluid=True, style={"backgroundColor": BG_DARK, "minHeight": "100vh",
                           "padding": "0 20px 40px"})
    return layout


app.layout = build_layout()


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("cooldown-plot",    "figure"),
    Output("subkelvin-plot",   "figure"),
    Output("heat-balance-plot","figure"),
    Output("status-bar",       "children"),
    Output("status-bar",       "style"),
    Output("anomaly-card",     "children"),
    Output("anomaly-card",     "style"),
    Output("sim-data-store",   "data"),
    *[Output(f"gauge-{i}", "figure") for i in range(6)],
    Input("run-btn",     "n_clicks"),
    State("sl-nwires",   "value"),
    State("sl-emiss",    "value"),
    State("sl-qubit",    "value"),
    State("sl-n3flow",   "value"),
    State("sw-warmup",   "value"),
    prevent_initial_call=True,
)
def run_simulation(n_clicks, n_wires, emissivity, qubit_nW, n3_flow, warmup_on):
    base_style   = {"fontFamily": "monospace", "padding": "6px 10px",
                    "border": "1px solid #4A5568", "borderRadius": "6px",
                    "fontSize": "0.85rem", "backgroundColor": BG_DARK}

    params = {
        "n_wires":      int(n_wires),
        "emissivity":   float(emissivity),
        "P_qubit_W":    float(qubit_nW) * 1e-9,
        "n3_flow_umol": float(n3_flow),
    }
    try:
        if warmup_on:
            t_h, T = simulate_warmup(t_hours=20.0, params=params, n_eval=1500)
            mode_label = "Warm-up"
        else:
            t_h, T = simulate_cooldown(params=params, n_eval=2000)
            mode_label = "Cool-down"
    except Exception as e:
        err_style = {"color": "#E74C3C", "padding": "6px 0",
                     "fontFamily": "monospace"}
        empty = go.Figure()
        empty.update_layout(paper_bgcolor=BG_DARK, plot_bgcolor=BG_CARD,
                            font_color=TEXT_CLR)
        return (empty, empty, empty, f"Error: {e}", err_style,
                "Anomaly: n/a", {**base_style, "color": "#888888"},
                None, *([empty] * 6))

    T_cols = ["T_300K","T_50K","T_4K","T_still","T_cold_plate","T_mxc"]
    df      = pd.DataFrame(np.column_stack([t_h, T]),
                           columns=["time_h"] + T_cols)
    T_final = T[-1]
    mxc_mK  = T_final[5] * 1e3

    # ── Status
    ok_col  = "#2ECC71" if mxc_mK < 20 else "#F39C12"
    ok_style = {"color": ok_col, "padding": "6px 0", "fontFamily": "monospace"}
    status   = (f"{mode_label} complete — MXC = {mxc_mK:.2f} mK | "
                f"Wires={n_wires}, ε={emissivity:.3f}, "
                f"P_q={qubit_nW:.0f} nW, ṅ₃={n3_flow:.0f} µmol/s")

    # ── Anomaly score
    if not warmup_on and _anomaly_ready.is_set() and _anomaly_model is not None:
        from anomaly_detector import score_simulation as _score
        res = _score(_anomaly_model, t_h, T)
        a_text  = f"{res['label']}  (score {res['pct']}%)"
        a_color = "#E74C3C" if res["is_anomaly"] else "#2ECC71"
    elif not _anomaly_ready.is_set():
        a_text, a_color = "Anomaly model: training…", "#F39C12"
    else:
        a_text, a_color = "Anomaly: n/a (warm-up mode)", "#888888"

    anomaly_card_style = {**base_style, "color": a_color}

    # ── Gauges
    gauges = [
        make_gauge(STAGE_NAMES[i], T_final[i], STAGE_TARGETS_K[i], STAGE_COLORS[i])
        for i in range(6)
    ]

    return (
        make_animated_cooldown(df),
        make_subkelvin_plot(df),
        make_heat_balance_bar(T_final),
        status, ok_style,
        a_text, anomaly_card_style,
        df.to_dict("records"),
        *gauges,
    )


@app.callback(
    Output("download-csv", "data"),
    Input("export-btn",    "n_clicks"),
    State("sim-data-store","data"),
    prevent_initial_call=True,
)
def export_csv(n_clicks, store_data):
    if not store_data:
        return no_update
    df = pd.DataFrame(store_data)
    return dcc.send_data_frame(df.to_csv, "dr_cooldown.csv", index=False)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Cryo DR Monitor dashboard → http://localhost:8050")
    app.run(debug=False, host="127.0.0.1", port=8050)
