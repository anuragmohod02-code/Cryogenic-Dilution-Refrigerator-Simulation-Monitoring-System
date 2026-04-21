"""
dashboard.py
------------
Plotly Dash web dashboard for the cryogenic dilution refrigerator simulation.
Reads outputs/stage_temperatures.csv or runs the simulation live.

Usage:
    python dashboard.py
    → Open http://localhost:8050 in a browser
"""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

from cryo_thermal import (
    simulate_cooldown, heat_balance,
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


# ── App layout ────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="Cryo DR Monitor",
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

    layout = dbc.Container([
        # Header
        dbc.Row([
            dbc.Col(html.H2("🧊 Cryogenic Dilution Refrigerator Monitor",
                            style={"color": "#63B3ED", "marginTop": "16px"})),
            dbc.Col(
                dbc.Button("▶ Re-run Simulation", id="run-btn",
                           color="primary", size="sm",
                           style={"marginTop": "20px", "float": "right"}),
                width=3,
            ),
        ], style={"backgroundColor": BG_DARK, "padding": "0 20px"}),

        html.Hr(style={"borderColor": "#2D3748"}),

        # Slider controls
        dbc.Card([
            dbc.CardHeader("Heat Load Parameters",
                           style={"color": TEXT_CLR, "backgroundColor": "#0F3460"}),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("# Wires", style={"color": TEXT_CLR}),
                        dcc.Slider(id="sl-nwires", min=4, max=96, step=4,
                                   value=24, marks={4:"4",24:"24",48:"48",96:"96"},
                                   tooltip={"placement":"bottom"}),
                    ], width=4),
                    dbc.Col([
                        html.Label("Emissivity", style={"color": TEXT_CLR}),
                        dcc.Slider(id="sl-emiss", min=0.001, max=0.2, step=0.005,
                                   value=0.05, marks={0.001:"0.001",0.05:"0.05",0.2:"0.2"},
                                   tooltip={"placement":"bottom"}),
                    ], width=4),
                    dbc.Col([
                        html.Label("Qubit Dissipation (nW)", style={"color": TEXT_CLR}),
                        dcc.Slider(id="sl-qubit", min=0, max=1000, step=10,
                                   value=100, marks={0:"0",100:"100",500:"500",1000:"1000"},
                                   tooltip={"placement":"bottom"}),
                    ], width=4),
                ]),
            ], style={"backgroundColor": BG_CARD}),
        ], style={"margin": "10px 0"}),

        # Status bar
        html.Div(id="status-bar",
                 children=f"Loaded data — MXC = {T_final[5]*1e3:.2f} mK",
                 style={"color": "#2ECC71", "padding": "6px 0",
                        "fontFamily": "monospace"}),

        # Temperature gauges
        dbc.Row(gauge_cards, style={"margin": "8px 0"}),

        # Cool-down plot
        dbc.Card([
            dbc.CardBody(dcc.Graph(id="cooldown-plot",
                                   figure=make_cooldown_plot(df),
                                   config={"displayModeBar": True}))
        ], style={"backgroundColor": BG_CARD, "border": "1px solid #2D3748",
                  "margin": "10px 0"}),

        # Heat balance bar
        dbc.Card([
            dbc.CardBody(dcc.Graph(id="heat-balance-plot",
                                   figure=make_heat_balance_bar(T_final),
                                   config={"displayModeBar": False}))
        ], style={"backgroundColor": BG_CARD, "border": "1px solid #2D3748",
                  "margin": "10px 0"}),

        # Store for simulation data
        dcc.Store(id="sim-data-store"),

    ], fluid=True, style={"backgroundColor": BG_DARK, "minHeight": "100vh",
                           "padding": "0 20px 40px"})
    return layout


app.layout = build_layout()


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("cooldown-plot",    "figure"),
    Output("heat-balance-plot","figure"),
    Output("status-bar",       "children"),
    Output("status-bar",       "style"),
    *[Output(f"gauge-{i}", "figure") for i in range(6)],
    Input("run-btn",   "n_clicks"),
    State("sl-nwires", "value"),
    State("sl-emiss",  "value"),
    State("sl-qubit",  "value"),
    prevent_initial_call=True,
)
def run_simulation(n_clicks, n_wires, emissivity, qubit_nW):
    params = {
        "n_wires":    int(n_wires),
        "emissivity": float(emissivity),
        "P_qubit_W":  float(qubit_nW) * 1e-9,
    }
    try:
        t_h, T = simulate_cooldown(params=params, n_eval=2000)
    except Exception as e:
        err_style = {"color": "#E74C3C", "padding": "6px 0",
                     "fontFamily": "monospace"}
        empty = go.Figure()
        empty.update_layout(paper_bgcolor=BG_DARK, plot_bgcolor=BG_CARD,
                            font_color=TEXT_CLR)
        gauges = [empty] * 6
        return empty, empty, f"Error: {e}", err_style, *gauges

    T_cols = ["T_300K","T_50K","T_4K","T_still","T_cold_plate","T_mxc"]
    df = pd.DataFrame(np.column_stack([t_h, T]), columns=["time_h"] + T_cols)
    T_final = T[-1]
    mxc_mK  = T_final[5] * 1e3

    ok_style = {"color": "#2ECC71" if mxc_mK < 20 else "#F39C12",
                "padding": "6px 0", "fontFamily": "monospace"}
    status = (f"Simulation complete — MXC = {mxc_mK:.2f} mK | "
              f"Wires={n_wires}, ε={emissivity:.3f}, P_qubit={qubit_nW:.0f} nW")

    gauges = [
        make_gauge(STAGE_NAMES[i], T_final[i], STAGE_TARGETS_K[i], STAGE_COLORS[i])
        for i in range(6)
    ]
    return (make_cooldown_plot(df), make_heat_balance_bar(T_final),
            status, ok_style, *gauges)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Cryo DR Monitor dashboard → http://localhost:8050")
    app.run(debug=False, host="127.0.0.1", port=8050)
