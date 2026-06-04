"""Reusable visual components for the VoxCustomer pages."""
from __future__ import annotations

from typing import Iterable

import streamlit as st


def brand_header() -> None:
    """Render the product brand block at the top of the sidebar."""
    st.markdown(
        """
        <div class="vox-brand">
          <div class="vox-logo">Vx</div>
          <div>
            <div class="vox-title">VoxCustomer</div>
            <div class="vox-subtitle">The voice of the customer, decoded.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div>
          <div class="vox-page-title">{title}</div>
          <div class="vox-page-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, delta: str | None = None,
             delta_kind: str = "neutral") -> str:
    """Return KPI HTML. Use inside an `st.markdown(..., unsafe_allow_html=True)`.

    `delta_kind`: 'pos' | 'neg' | 'neutral'.
    """
    delta_cls = {"pos": "pos", "neg": "neg"}.get(delta_kind, "")
    delta_html = (
        f'<div class="vox-kpi-delta {delta_cls}">{delta}</div>' if delta else ""
    )
    return (
        f'<div class="vox-kpi">'
        f'  <div class="vox-kpi-label">{label}</div>'
        f'  <div class="vox-kpi-value">{value}</div>'
        f'  {delta_html}'
        f'</div>'
    )


def kpi_row(cards: Iterable[str]) -> None:
    cards = list(cards)
    cols = st.columns(len(cards), gap="small")
    for col, card in zip(cols, cards):
        with col:
            st.markdown(card, unsafe_allow_html=True)
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)


def insight(text: str) -> None:
    st.markdown(f'<div class="vox-insight">{text}</div>', unsafe_allow_html=True)


def section(title: str, subtitle: str | None = None):
    """Context-manager-friendly section card opener.

    Usage:
        with section("Title", "subtitle"):
            st.plotly_chart(fig, width="stretch")
    """
    return _Section(title, subtitle)


class _Section:
    def __init__(self, title: str, subtitle: str | None):
        self.title = title
        self.subtitle = subtitle

    def __enter__(self):
        sub = (
            f'<div class="vox-card-sub">{self.subtitle}</div>'
            if self.subtitle else ""
        )
        st.markdown(
            f'<div class="vox-card"><h3>{self.title}</h3>{sub}',
            unsafe_allow_html=True,
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        st.markdown("</div>", unsafe_allow_html=True)
        return False


def sentiment_badge(label: str) -> str:
    cls = {"Positive": "pos", "Neutral": "neu", "Negative": "neg"}.get(label, "neu")
    return f'<span class="vox-badge {cls}">{label}</span>'
