"""Dataset loading and cleaning for the Customer Support Ticket dataset.

The CSV ships with a placeholder `{product_purchased}` token in every
ticket description. We hydrate that token before any text inference so
that products surface as a real analytical dimension.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "customer_support_tickets.csv"

# Columns we actively use downstream — keep this list explicit.
USED_COLUMNS: tuple[str, ...] = (
    "Ticket ID",
    "Customer Age",
    "Customer Gender",
    "Product Purchased",
    "Ticket Type",
    "Ticket Subject",
    "Ticket Description",
    "Ticket Status",
    "Ticket Priority",
    "Ticket Channel",
    "Customer Satisfaction Rating",
    "Date of Purchase",
)


def _hydrate_description(text: str, product: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.replace("{product_purchased}", str(product) if isinstance(product, str) else "")


@st.cache_data(show_spinner="Loading customer support tickets…")
def load_tickets(sample_size: int | None = None, seed: int = 7) -> pd.DataFrame:
    """Load and clean the support tickets CSV.

    Args:
        sample_size: If provided, return a deterministic random sample of this size.
        seed: RNG seed for sampling.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Download "
            "'customer_support_tickets.csv' from Kaggle and place it there."
        )

    df = pd.read_csv(DATA_PATH)
    missing = [c for c in USED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.copy()
    df["Ticket Description"] = [
        _hydrate_description(t, p)
        for t, p in zip(df["Ticket Description"], df["Product Purchased"])
    ]
    df["Date of Purchase"] = pd.to_datetime(df["Date of Purchase"], errors="coerce")
    df["Customer Satisfaction Rating"] = pd.to_numeric(
        df["Customer Satisfaction Rating"], errors="coerce"
    )

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    """Lightweight aggregate stats used by KPI cards."""
    total = len(df)
    resolved = (df["Ticket Status"] == "Closed").sum()
    open_t = total - int(resolved)
    csat = df["Customer Satisfaction Rating"].dropna()
    return {
        "total_tickets": total,
        "open_tickets": int(open_t),
        "resolved_tickets": int(resolved),
        "unique_products": int(df["Product Purchased"].nunique()),
        "avg_csat": float(csat.mean()) if len(csat) else float("nan"),
        "csat_responses": int(len(csat)),
    }
