"""Visual identity for VoxCustomer.

Centralizes the color palette, the global Streamlit CSS overrides
and a custom Plotly template so every page looks like one product.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ── Brand palette ────────────────────────────────────────────────────────────
PALETTE = {
    "bg":          "#0B0E1A",
    "surface":     "#141A2E",
    "surface_alt": "#1B2238",
    "border":      "#252C46",
    "text":        "#E6E8F2",
    "text_muted":  "#8A93B0",
    "primary":     "#7C5CFF",
    "primary_dim": "#5B40D6",
    "accent":      "#00E0C7",
    "positive":    "#22C55E",
    "neutral":     "#94A3B8",
    "negative":    "#EF4444",
    "warning":     "#F59E0B",
}

SENTIMENT_COLORS = {
    "Positive": PALETTE["positive"],
    "Neutral":  PALETTE["neutral"],
    "Negative": PALETTE["negative"],
}

PRIORITY_COLORS = {
    "Low":      PALETTE["accent"],
    "Medium":   PALETTE["primary"],
    "High":     PALETTE["warning"],
    "Critical": PALETTE["negative"],
}


# ── Plotly template ──────────────────────────────────────────────────────────
def _register_plotly_template() -> None:
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family="Inter, system-ui, sans-serif", size=13, color=PALETTE["text"]),
        title=dict(font=dict(size=18, color=PALETTE["text"]), x=0.01, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[
            PALETTE["primary"], PALETTE["accent"], PALETTE["warning"],
            PALETTE["positive"], PALETTE["negative"], "#F472B6", "#60A5FA",
        ],
        xaxis=dict(
            gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"],
            linecolor=PALETTE["border"], tickfont=dict(color=PALETTE["text_muted"]),
            title=dict(font=dict(color=PALETTE["text_muted"], size=12)),
        ),
        yaxis=dict(
            gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"],
            linecolor=PALETTE["border"], tickfont=dict(color=PALETTE["text_muted"]),
            title=dict(font=dict(color=PALETTE["text_muted"], size=12)),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor=PALETTE["border"], borderwidth=0,
            font=dict(color=PALETTE["text"], size=12),
        ),
        margin=dict(l=40, r=20, t=60, b=40),
        hoverlabel=dict(
            bgcolor=PALETTE["surface_alt"], bordercolor=PALETTE["primary"],
            font=dict(color=PALETTE["text"], size=12),
        ),
    )
    pio.templates["voxcustomer"] = template
    pio.templates.default = "voxcustomer"


# ── Global CSS ───────────────────────────────────────────────────────────────
_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: 'Inter', system-ui, sans-serif !important;
        color: {PALETTE['text']};
    }}

    .stApp {{
        background:
            radial-gradient(1200px 600px at 10% -10%, rgba(124,92,255,0.15), transparent 60%),
            radial-gradient(900px 500px at 110% 10%, rgba(0,224,199,0.08), transparent 60%),
            {PALETTE['bg']};
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: {PALETTE['surface']};
        border-right: 1px solid {PALETTE['border']};
    }}
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span {{
        color: {PALETTE['text']} !important;
    }}

    /* Block container — override Streamlit's narrow default */
    .main .block-container,
    section.main > div.block-container,
    div[data-testid="stMainBlockContainer"] {{
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 100% !important;
    }}

    /* Brand header */
    .vox-brand {{
        display: flex; align-items: center; gap: 14px;
        padding: 8px 0 20px 0;
        border-bottom: 1px solid {PALETTE['border']};
        margin-bottom: 22px;
    }}
    .vox-logo {{
        width: 44px; height: 44px; border-radius: 12px;
        background: linear-gradient(135deg, {PALETTE['primary']}, {PALETTE['accent']});
        display: grid; place-items: center;
        font-weight: 800; color: white; font-size: 18px;
        box-shadow: 0 8px 24px rgba(124,92,255,0.35);
    }}
    .vox-title {{ font-size: 22px; font-weight: 700; letter-spacing: -0.01em; }}
    .vox-subtitle {{ color: {PALETTE['text_muted']}; font-size: 13px; margin-top: 2px; }}

    /* Page header */
    .vox-page-title {{
        font-size: 28px; font-weight: 700; letter-spacing: -0.02em;
        margin: 0 0 4px 0;
    }}
    .vox-page-subtitle {{
        color: {PALETTE['text_muted']}; font-size: 14px;
        margin: 0 0 24px 0;
    }}

    /* KPI cards */
    .vox-kpi {{
        background: linear-gradient(180deg, {PALETTE['surface']} 0%, {PALETTE['surface_alt']} 100%);
        border: 1px solid {PALETTE['border']};
        border-radius: 14px;
        padding: 14px 16px;
        height: 100%;
        min-width: 0; overflow: hidden;
        transition: transform .15s ease, border-color .15s ease;
    }}
    .vox-kpi:hover {{
        transform: translateY(-2px);
        border-color: {PALETTE['primary']};
    }}
    .vox-kpi-label {{
        color: {PALETTE['text_muted']} !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        line-height: 1.25 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }}
    .vox-kpi-value {{
        font-size: 26px !important;
        font-weight: 700 !important;
        margin-top: 4px !important;
        line-height: 1.1 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }}
    .vox-kpi-delta {{
        font-size: 11.5px !important;
        margin-top: 6px !important;
        color: {PALETTE['text_muted']};
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }}
    .vox-kpi-delta.pos {{ color: {PALETTE['positive']}; }}
    .vox-kpi-delta.neg {{ color: {PALETTE['negative']}; }}

    /* Section card */
    .vox-card {{
        background: {PALETTE['surface']};
        border: 1px solid {PALETTE['border']};
        border-radius: 16px;
        padding: 18px 20px 8px 20px;
        margin-bottom: 18px;
    }}
    .vox-card h3 {{
        font-size: 15px; font-weight: 600; margin: 0 0 4px 0;
        color: {PALETTE['text']};
    }}
    .vox-card .vox-card-sub {{
        font-size: 12px; color: {PALETTE['text_muted']}; margin-bottom: 12px;
    }}

    /* Insight bubble */
    .vox-insight {{
        background: linear-gradient(135deg, rgba(124,92,255,0.12), rgba(0,224,199,0.06));
        border: 1px solid rgba(124,92,255,0.35);
        border-left: 3px solid {PALETTE['primary']};
        border-radius: 10px;
        padding: 12px 14px;
        font-size: 13px;
        color: {PALETTE['text']};
        margin: 6px 0 14px 0;
    }}
    .vox-insight b {{ color: {PALETTE['accent']}; }}

    /* Badges */
    .vox-badge {{
        display: inline-block; padding: 3px 10px; border-radius: 999px;
        font-size: 11px; font-weight: 600; letter-spacing: 0.04em;
        text-transform: uppercase;
    }}
    .vox-badge.pos {{ background: rgba(34,197,94,0.15);  color: {PALETTE['positive']}; }}
    .vox-badge.neu {{ background: rgba(148,163,184,0.18); color: {PALETTE['neutral']}; }}
    .vox-badge.neg {{ background: rgba(239,68,68,0.15);  color: {PALETTE['negative']}; }}

    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {PALETTE['primary']}, {PALETTE['primary_dim']});
        color: white; border: 0; border-radius: 10px;
        font-weight: 600; padding: 8px 18px;
        box-shadow: 0 6px 16px rgba(124,92,255,0.35);
    }}
    .stButton > button:hover {{
        filter: brightness(1.08); transform: translateY(-1px);
    }}

    /* Text input */
    .stTextArea textarea, .stTextInput input {{
        background: {PALETTE['surface']} !important;
        border: 1px solid {PALETTE['border']} !important;
        color: {PALETTE['text']} !important;
        border-radius: 10px !important;
    }}
    .stTextArea textarea:focus, .stTextInput input:focus {{
        border-color: {PALETTE['primary']} !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px; border-bottom: 1px solid {PALETTE['border']};
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent; color: {PALETTE['text_muted']};
        padding: 10px 16px; border-radius: 10px 10px 0 0;
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        color: {PALETTE['text']} !important;
        background: {PALETTE['surface']} !important;
        border-bottom: 2px solid {PALETTE['primary']} !important;
    }}

    /* Dataframes */
    [data-testid="stDataFrame"] {{
        border: 1px solid {PALETTE['border']};
        border-radius: 12px; overflow: hidden;
    }}

    /* Hide Streamlit chrome */
    #MainMenu, footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{ background: transparent; }}
</style>
"""


def apply_theme() -> None:
    """Install CSS + Plotly template. Safe to call once per page."""
    _register_plotly_template()
    st.markdown(_CSS, unsafe_allow_html=True)
