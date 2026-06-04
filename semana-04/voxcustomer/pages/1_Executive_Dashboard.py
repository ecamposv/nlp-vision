"""Page 1 — Executive Dashboard.

Treats sentiment as a production feature. Hydrates the support ticket
dataset, runs the TF-IDF baseline on every description and surfaces how
the resulting signal correlates with product, channel and priority.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.components import (
    brand_header, insight, kpi_card, kpi_row, page_header, section,
)
from src.data_loader import dataset_summary, load_tickets
from src.models import get_traditional_model
from src.theme import PALETTE, PRIORITY_COLORS, SENTIMENT_COLORS, apply_theme

st.set_page_config(
    page_title="VoxCustomer · Executive Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()


# ── Cached enrichment ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Scoring sentiment for every ticket…")
def score_tickets(df: pd.DataFrame) -> pd.DataFrame:
    model = get_traditional_model()
    preds = model.predict_batch(df["Ticket Description"].tolist())
    out = df.copy()
    out["Predicted Sentiment"] = [p.label for p in preds]
    out["Sentiment Confidence"] = [p.score for p in preds]
    out["Negativity Score"] = [p.neg_score for p in preds]
    return out


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    brand_header()
    st.markdown("### Filters")
    raw_df = load_tickets()

    channels = sorted(raw_df["Ticket Channel"].dropna().unique().tolist())
    sel_channels = st.multiselect("Channel", channels, default=channels)

    statuses = sorted(raw_df["Ticket Status"].dropna().unique().tolist())
    sel_statuses = st.multiselect("Status", statuses, default=statuses)

    priorities_all = ["Low", "Medium", "High", "Critical"]
    priorities_present = [p for p in priorities_all if p in raw_df["Ticket Priority"].unique()]
    sel_priorities = st.multiselect("Priority", priorities_present, default=priorities_present)

    st.divider()
    st.caption(
        "Model: **TF-IDF + Logistic Regression**\n\n"
        "Trained on a curated, support-domain corpus bundled with VoxCustomer. "
        "Inference runs in-process on every ticket."
    )


# ── Score + filter ───────────────────────────────────────────────────────────
scored = score_tickets(raw_df)
df = scored[
    scored["Ticket Channel"].isin(sel_channels)
    & scored["Ticket Status"].isin(sel_statuses)
    & scored["Ticket Priority"].isin(sel_priorities)
].copy()

# ── Header ───────────────────────────────────────────────────────────────────
page_header(
    "Executive Dashboard",
    "Sentiment as a first-class operational metric, derived from raw ticket text.",
)

if df.empty:
    st.warning("No tickets match the current filters.")
    st.stop()


# ── KPI bar ──────────────────────────────────────────────────────────────────
total = len(df)
neg_count = int((df["Predicted Sentiment"] == "Negative").sum())
pos_count = int((df["Predicted Sentiment"] == "Positive").sum())
neg_share = neg_count / total
pos_share = pos_count / total
health_score = round(100 * (1 - df["Negativity Score"].mean()), 1)
critical_share = (df["Ticket Priority"] == "Critical").mean()
csat = df["Customer Satisfaction Rating"].dropna()
avg_csat = float(csat.mean()) if len(csat) else float("nan")
summary = dataset_summary(df)

kpi_row([
    kpi_card("Customer health index", f"{health_score:.1f}",
             delta="out of 100 · higher is better", delta_kind="neutral"),
    kpi_card("Tickets in view", f"{total:,}",
             delta=f"{summary['open_tickets']:,} still open", delta_kind="neg"),
    kpi_card("Avg. CSAT", f"{avg_csat:.2f}" if avg_csat == avg_csat else "n/a",
             delta=f"{len(csat):,} responses",
             delta_kind="pos" if avg_csat == avg_csat and avg_csat >= 3.5 else "neg"),
])
kpi_row([
    kpi_card("Negative voice", f"{neg_share:.1%}",
             delta=f"{neg_count:,} tickets flagged",
             delta_kind="neg" if neg_share >= 0.30 else "neutral"),
    kpi_card("Positive voice", f"{pos_share:.1%}",
             delta=f"{pos_count:,} tickets",
             delta_kind="pos"),
    kpi_card("Critical priority", f"{critical_share:.1%}",
             delta="of all tickets in view", delta_kind="neutral"),
])

insight(
    "The <b>Customer Health Index</b> is <b>1 − average negativity probability</b> across "
    "all tickets in view. It moves in real time as filters change, giving leadership a "
    "single, sentiment-aware number that traditional ticket counters cannot produce."
)

st.markdown(" ")

# ── Row: sentiment mix + product × sentiment ─────────────────────────────────
left, right = st.columns([1, 2], gap="large")

with left:
    with section("Voice of customer", "Predicted sentiment mix across the current view."):
        mix = df["Predicted Sentiment"].value_counts().reindex(
            ["Positive", "Neutral", "Negative"], fill_value=0
        )
        donut = go.Figure(go.Pie(
            labels=mix.index.tolist(), values=mix.values.tolist(),
            hole=0.62, sort=False,
            marker=dict(colors=[SENTIMENT_COLORS[lbl] for lbl in mix.index],
                        line=dict(color=PALETTE["bg"], width=2)),
            textposition="outside",
            texttemplate="%{label}<br>%{percent:.1%}",
            hovertemplate="<b>%{label}</b><br>%{value:,} tickets (%{percent})<extra></extra>",
        ))
        donut.update_layout(
            height=360, showlegend=False, margin=dict(t=20, b=20, l=10, r=10),
            annotations=[dict(
                text=f"<b>{total:,}</b><br><span style='color:{PALETTE['text_muted']}'>tickets</span>",
                showarrow=False, font=dict(size=18, color=PALETTE["text"]),
            )],
        )
        st.plotly_chart(donut, width="stretch")

with right:
    with section(
        "Product × Predicted Sentiment",
        "Top 12 products ranked by negative-voice share. The brands that need "
        "intervention surface immediately.",
    ):
        top_products = df["Product Purchased"].value_counts().head(12).index.tolist()
        sub = df[df["Product Purchased"].isin(top_products)]
        ct = (
            sub.groupby(["Product Purchased", "Predicted Sentiment"])
               .size().unstack(fill_value=0)
        )
        for col in ["Negative", "Neutral", "Positive"]:
            if col not in ct.columns:
                ct[col] = 0
        ct = ct[["Negative", "Neutral", "Positive"]]
        ct["__total__"] = ct.sum(axis=1)
        ct["__neg_share__"] = ct["Negative"] / ct["__total__"].clip(lower=1)
        ct = ct.sort_values("__neg_share__", ascending=True)

        fig = go.Figure()
        for lbl in ["Negative", "Neutral", "Positive"]:
            fig.add_bar(
                y=ct.index, x=ct[lbl], name=lbl, orientation="h",
                marker=dict(color=SENTIMENT_COLORS[lbl]),
                hovertemplate=f"<b>%{{y}}</b><br>{lbl}: %{{x:,}}<extra></extra>",
            )
        fig.update_layout(
            barmode="stack", height=420,
            xaxis_title="Tickets", yaxis_title=None,
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, width="stretch")

# ── Row: priority vs negativity + sentiment vs CSAT ──────────────────────────
left2, right2 = st.columns([3, 2], gap="large")

with left2:
    with section(
        "Ticket Priority vs. Text Negativity",
        "Distribution of the per-ticket negativity probability for each priority "
        "bucket. A healthy queue would show the negativity mass shifting right as "
        "priority climbs.",
    ):
        order = [p for p in ["Low", "Medium", "High", "Critical"]
                 if p in df["Ticket Priority"].unique()]
        fig = go.Figure()
        for prio in order:
            sub = df[df["Ticket Priority"] == prio]["Negativity Score"]
            fig.add_trace(go.Violin(
                x=[prio] * len(sub), y=sub, name=prio, box_visible=True,
                meanline_visible=True, line_color=PRIORITY_COLORS.get(prio, PALETTE["primary"]),
                fillcolor=PRIORITY_COLORS.get(prio, PALETTE["primary"]), opacity=0.55,
                spanmode="hard", points=False,
            ))
        fig.update_layout(
            height=420, showlegend=False,
            yaxis=dict(title="Negativity probability", range=[0, 1]),
            xaxis=dict(title=None, categoryorder="array", categoryarray=order),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, width="stretch")

with right2:
    with section(
        "Predicted Sentiment vs. CSAT",
        "Ground-truth check: does the inferred sentiment line up with the rating "
        "customers actually leave?",
    ):
        sub = df.dropna(subset=["Customer Satisfaction Rating"])
        if sub.empty:
            st.info("No CSAT responses in the current view.")
        else:
            grouped = (
                sub.groupby("Predicted Sentiment")["Customer Satisfaction Rating"]
                   .agg(["mean", "count"])
                   .reindex(["Negative", "Neutral", "Positive"])
                   .dropna()
            )
            fig = go.Figure()
            fig.add_bar(
                x=grouped.index, y=grouped["mean"],
                marker=dict(color=[SENTIMENT_COLORS[i] for i in grouped.index]),
                text=[f"{v:.2f}" for v in grouped["mean"]],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Mean CSAT: %{y:.2f}<extra></extra>",
            )
            fig.update_layout(
                height=420,
                yaxis=dict(title="Mean CSAT (1–5)", range=[0, 5]),
                xaxis=dict(title=None),
                showlegend=False,
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "Counts per bucket: "
                + " · ".join(f"{lbl}: {int(grouped.loc[lbl, 'count'])}"
                             for lbl in grouped.index)
            )

# ── Row: channel sentiment heat + raw drill-in ───────────────────────────────
with section(
    "Channel × Sentiment heat",
    "Where in the contact mix is negative voice concentrated? "
    "Hot cells point straight at staffing decisions.",
):
    ct = (
        df.groupby(["Ticket Channel", "Predicted Sentiment"])
          .size().unstack(fill_value=0)
          .reindex(columns=["Negative", "Neutral", "Positive"], fill_value=0)
    )
    ct_pct = ct.div(ct.sum(axis=1).replace(0, 1), axis=0)
    fig = go.Figure(go.Heatmap(
        z=ct_pct.values * 100,
        x=ct_pct.columns.tolist(),
        y=ct_pct.index.tolist(),
        colorscale=[[0, PALETTE["surface_alt"]], [0.5, PALETTE["primary"]], [1, PALETTE["negative"]]],
        zmin=0, zmax=100,
        colorbar=dict(title="% of channel", tickfont=dict(color=PALETTE["text_muted"])),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>",
        text=[[f"{v:.0f}%" for v in row] for row in ct_pct.values * 100],
        texttemplate="%{text}", textfont=dict(color="white", size=12),
    ))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width="stretch")


with section(
    "Most negative tickets in view",
    "Sorted by the model's confidence that the description is negative. "
    "Useful for QA, escalation drilldowns or sampling for fine-tuning.",
):
    top_neg = df.sort_values("Negativity Score", ascending=False).head(15)
    show = top_neg[[
        "Ticket ID", "Product Purchased", "Ticket Priority",
        "Ticket Channel", "Predicted Sentiment", "Negativity Score",
        "Ticket Description",
    ]].rename(columns={
        "Negativity Score": "P(neg)",
        "Ticket Description": "Description",
    })
    st.dataframe(
        show, width="stretch", hide_index=True,
        column_config={
            "P(neg)": st.column_config.ProgressColumn(
                "P(neg)", min_value=0.0, max_value=1.0, format="%.2f",
            ),
            "Description": st.column_config.TextColumn(width="large"),
        },
    )
