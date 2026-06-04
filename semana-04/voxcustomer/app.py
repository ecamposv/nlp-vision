"""VoxCustomer — landing page.

Streamlit auto-loads the multipage app from `pages/`. This file is the
default entry point and acts as the product home screen.
"""
from __future__ import annotations

import streamlit as st

from src.components import brand_header, insight, kpi_card, kpi_row, page_header
from src.data_loader import dataset_summary, load_tickets
from src.theme import apply_theme

st.set_page_config(
    page_title="VoxCustomer · Home",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

with st.sidebar:
    brand_header()
    st.markdown(
        "Use the navigation above to explore the two views:\n\n"
        "**1 · Executive Dashboard** — sentiment as a production feature, "
        "applied to the full ticket stream.\n\n"
        "**2 · Model Evaluation** — a head-to-head between a TF-IDF baseline "
        "and a transformer model, with an interactive playground."
    )

page_header(
    "Welcome to VoxCustomer",
    "Turn unstructured customer feedback into an operational, measurable signal.",
)

# ── Hero KPIs from the live dataset ──────────────────────────────────────────
df = load_tickets()
summary = dataset_summary(df)

kpi_row([
    kpi_card("Tickets in scope", f"{summary['total_tickets']:,}"),
    kpi_card("Unique products", f"{summary['unique_products']:,}"),
    kpi_card("Open tickets", f"{summary['open_tickets']:,}",
             delta=f"{summary['open_tickets']/max(summary['total_tickets'],1):.0%} of total",
             delta_kind="neg"),
    kpi_card("Avg. CSAT", f"{summary['avg_csat']:.2f}" if summary['avg_csat'] == summary['avg_csat'] else "n/a",
             delta=f"{summary['csat_responses']:,} responses", delta_kind="neutral"),
])

insight(
    "VoxCustomer ingests raw <b>ticket descriptions</b> — the kind of free-text noise "
    "most BI tools quietly ignore — and converts them into a sentiment signal you can "
    "slice by product, channel, and priority. Open a page from the sidebar to see it in action."
)

st.markdown(" ")

# ── Page cards ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        """
        <div class="vox-card" style="height: 100%;">
          <h3>📊 Executive Dashboard</h3>
          <div class="vox-card-sub">Sentiment as a production feature</div>
          <p style="color:#8A93B0; font-size:13px; line-height:1.6;">
            A live operational view: customer-health KPIs, product vs.
            predicted sentiment, and a distribution of textual negativity
            broken down by ticket priority.
          </p>
          <p style="color:#E6E8F2; font-size:13px;">
            <b>Master's lesson:</b> text is not a string — it is an analytical
            dimension that drives business decisions and resource allocation.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="vox-card" style="height: 100%;">
          <h3>🧪 Model Evaluation</h3>
          <div class="vox-card-sub">TF-IDF vs. Transformer, head-to-head</div>
          <p style="color:#8A93B0; font-size:13px; line-height:1.6;">
            An engineer's playground: type a tricky ticket — sarcasm,
            negation, mixed sentiment — and watch both models react.
            Includes confusion matrices and per-document latency.
          </p>
          <p style="color:#E6E8F2; font-size:13px;">
            <b>Master's lesson:</b> production AI is a constant trade-off
            between accuracy, compute cost, and latency.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div style="margin-top: 24px; color:#8A93B0; font-size:12px; text-align:center;">
        VoxCustomer · built with Streamlit, Plotly, scikit-learn and Hugging Face Transformers
    </div>
    """,
    unsafe_allow_html=True,
)
