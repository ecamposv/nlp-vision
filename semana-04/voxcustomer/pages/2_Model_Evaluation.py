"""Page 2 — Deep-Dive Model Evaluation.

Head-to-head between a TF-IDF + Logistic Regression baseline and a
RoBERTa transformer fine-tuned for sentiment. Includes an interactive
playground, a benchmark on a curated tricky test set, and a short
theory section on latency vs. accuracy trade-offs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.components import (
    brand_header, insight, kpi_card, kpi_row, page_header, section,
    sentiment_badge,
)
from src.corpus import LABELS
from src.models import (
    Prediction, classification_report, confusion_matrix,
    evaluation_set, get_traditional_model, get_transformer_model,
)
from src.theme import PALETTE, SENTIMENT_COLORS, apply_theme

st.set_page_config(
    page_title="VoxCustomer · Model Evaluation",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()


# ── Suggested adversarial prompts ────────────────────────────────────────────
SUGGESTED_PROMPTS: list[tuple[str, str]] = [
    ("Sarcasm",   "Oh fantastic, only three weeks to get a reply. Truly world-class support."),
    ("Negation",  "The product is not bad at all, it actually works really well."),
    ("Mixed",     "Great hardware, but the software is buggy and the support is awful."),
    ("Subtle +",  "After the latest update I am finally able to use this product the way I wanted."),
    ("Subtle −",  "After the third broken unit, I have given up on this brand."),
]


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    brand_header()
    st.markdown("### Models in this comparison")
    st.markdown(
        "**A · TF-IDF + LogReg**\n"
        "Classic bag-of-words pipeline. Bundled, trains in ~1 s.\n\n"
        "**B · RoBERTa transformer**\n"
        "`cardiffnlp/twitter-roberta-base-sentiment-latest` — contextual, "
        "downloads once (~500 MB), then runs locally."
    )
    st.divider()
    st.caption(
        "The transformer is loaded **on demand**. The first action that needs "
        "it will trigger the download and cache it for the rest of the session."
    )


# ── Header ───────────────────────────────────────────────────────────────────
page_header(
    "Deep-Dive Model Evaluation",
    "TF-IDF vs. transformer: where each model wins, where each breaks, and what it costs.",
)


# ── Model status cards ───────────────────────────────────────────────────────
traditional = get_traditional_model()
transformer_state = st.session_state.setdefault("transformer_state", "idle")  # idle | ready

col_a, col_b = st.columns(2, gap="large")
with col_a:
    st.markdown(
        f"""
        <div class="vox-card" style="height:100%;">
          <h3>A · {traditional.name}</h3>
          <div class="vox-card-sub">{traditional.family} · ready</div>
          <p style="color:#8A93B0; font-size:13px;">
            Word + character n-grams fed into a multinomial logistic regression.
            Trains in seconds on the bundled support-domain corpus, predicts in
            microseconds per document, ships in any environment.
          </p>
          <span class="vox-badge pos">Ready</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_b:
    if transformer_state == "ready":
        transformer = get_transformer_model()
        if transformer.available:
            badge = '<span class="vox-badge pos">Ready</span>'
            note = (
                "Pretrained sentiment-classification head over RoBERTa. Captures "
                "context, negation, sarcasm and idioms — at the cost of model "
                "weights and per-token compute."
            )
        else:
            badge = '<span class="vox-badge neg">Unavailable</span>'
            note = transformer.unavailable_reason or "Transformer could not be loaded."
        st.markdown(
            f"""
            <div class="vox-card" style="height:100%;">
              <h3>B · RoBERTa transformer</h3>
              <div class="vox-card-sub">Deep Learning / Transformer</div>
              <p style="color:#8A93B0; font-size:13px;">{note}</p>
              {badge}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="vox-card" style="height:100%;">
              <h3>B · RoBERTa transformer</h3>
              <div class="vox-card-sub">Deep Learning / Transformer · idle</div>
              <p style="color:#8A93B0; font-size:13px;">
                Click <b>Load transformer</b> to fetch the model weights and
                enable side-by-side comparison. The download happens once per
                machine; subsequent runs are instant.
              </p>
              <span class="vox-badge neu">Not loaded</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Load transformer", key="load_transformer"):
            with st.spinner("Loading transformer weights…"):
                get_transformer_model()
            st.session_state["transformer_state"] = "ready"
            st.rerun()


# ── Helpers ──────────────────────────────────────────────────────────────────
def _confidence_bar(p: Prediction, classes: list[str], probs: dict[str, float]) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=[probs[c] * 100 for c in classes], y=classes, orientation="h",
        marker=dict(color=[SENTIMENT_COLORS[c] for c in classes]),
        text=[f"{probs[c]*100:.1f}%" for c in classes],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=170, margin=dict(l=10, r=20, t=10, b=10),
        xaxis=dict(range=[0, 110], title=None, showticklabels=False),
        yaxis=dict(title=None, categoryorder="array",
                   categoryarray=list(reversed(classes))),
        showlegend=False,
    )
    return fig


def _predict_with_probs(model, text: str) -> tuple[Prediction, dict[str, float]]:
    """Return prediction plus full per-class probability dict.

    Mirrors what the underlying models compute internally so we can render
    a confidence bar without recomputing probabilities.
    """
    probs = model._proba([text])[0]  # noqa: SLF001 — internal but stable
    pred = model.predict_one(text)
    return pred, {c: float(probs[i]) for i, c in enumerate(model.classes_)}


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_play, tab_bench, tab_theory = st.tabs(
    ["🎮 Playground", "📐 Benchmark", "📚 Theory & cost"]
)


# ── Playground ───────────────────────────────────────────────────────────────
with tab_play:
    with section(
        "Try a tricky ticket",
        "Type a customer message or pick a preset. Both models score it "
        "side-by-side. Watch how the baseline reacts to sarcasm and negation.",
    ):
        if "playground_text" not in st.session_state:
            st.session_state["playground_text"] = SUGGESTED_PROMPTS[0][1]

        suggest_cols = st.columns(len(SUGGESTED_PROMPTS))
        for i, (label, text) in enumerate(SUGGESTED_PROMPTS):
            if suggest_cols[i].button(label, key=f"suggest_{i}", width="stretch"):
                st.session_state["playground_text"] = text
                st.rerun()

        user_text = st.text_area(
            "Customer message",
            key="playground_text",
            height=110,
            placeholder="Paste a ticket description here…",
        )

        run_clicked = st.button("Run analysis", type="primary")

    if run_clicked and user_text.strip():
        col_l, col_r = st.columns(2, gap="large")

        # Traditional
        with col_l:
            with section(f"A · {traditional.name}", traditional.family):
                pred_a, probs_a = _predict_with_probs(traditional, user_text)
                st.markdown(
                    f"<div style='font-size:22px; font-weight:700;'>"
                    f"{sentiment_badge(pred_a.label)} &nbsp; "
                    f"<span style='color:{PALETTE['text_muted']}; font-size:13px; font-weight:500;'>"
                    f"confidence {pred_a.score:.0%} · {pred_a.latency_ms:.2f} ms</span></div>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    _confidence_bar(pred_a, list(LABELS), probs_a),
                    width="stretch",
                )

        # Transformer
        with col_r:
            if transformer_state != "ready":
                with section("B · RoBERTa transformer", "Not loaded yet"):
                    st.info(
                        "Click **Load transformer** at the top of the page to "
                        "enable side-by-side comparison."
                    )
            else:
                transformer = get_transformer_model()
                if not transformer.available:
                    with section("B · RoBERTa transformer", "Unavailable"):
                        st.error(transformer.unavailable_reason)
                else:
                    with section(f"B · {transformer.name}", transformer.family):
                        pred_b, probs_b = _predict_with_probs(transformer, user_text)
                        st.markdown(
                            f"<div style='font-size:22px; font-weight:700;'>"
                            f"{sentiment_badge(pred_b.label)} &nbsp; "
                            f"<span style='color:{PALETTE['text_muted']}; font-size:13px; font-weight:500;'>"
                            f"confidence {pred_b.score:.0%} · {pred_b.latency_ms:.1f} ms</span></div>",
                            unsafe_allow_html=True,
                        )
                        st.plotly_chart(
                            _confidence_bar(pred_b, list(LABELS), probs_b),
                            width="stretch",
                        )

                        if pred_a.label != pred_b.label:
                            insight(
                                f"The two models <b>disagree</b>: the baseline says "
                                f"<b>{pred_a.label}</b> while the transformer says "
                                f"<b>{pred_b.label}</b>. This is exactly the kind of "
                                f"input — sarcasm, negation, mixed clauses — where "
                                f"contextual representations earn their compute cost."
                            )
                        else:
                            insight(
                                f"Both models <b>agree</b> on <b>{pred_a.label}</b>. "
                                f"For clear-signal text, the baseline often matches the "
                                f"transformer at a fraction of the latency."
                            )


# ── Benchmark ────────────────────────────────────────────────────────────────
with tab_bench:
    with section(
        "Curated benchmark",
        "A hand-labeled, 50-example test set with a deliberate dose of "
        "sarcasm, negation and mixed sentiment. Same test set is fed to both "
        "models so the comparison is apples-to-apples.",
    ):
        texts, y_true = evaluation_set()

        # Traditional
        preds_a = traditional.predict_batch(texts)
        y_pred_a = [p.label for p in preds_a]
        report_a = classification_report(y_true, y_pred_a)
        cm_a = confusion_matrix(y_true, y_pred_a)
        lat_a = float(np.mean([p.latency_ms for p in preds_a]))

        # Transformer
        if transformer_state == "ready":
            transformer = get_transformer_model()
            if transformer.available:
                with st.spinner("Scoring benchmark with the transformer…"):
                    preds_b = transformer.predict_batch(texts)
                y_pred_b = [p.label for p in preds_b]
                report_b = classification_report(y_true, y_pred_b)
                cm_b = confusion_matrix(y_true, y_pred_b)
                lat_b = float(np.mean([p.latency_ms for p in preds_b]))
                transformer_ready = True
            else:
                transformer_ready = False
        else:
            transformer_ready = False

        # KPI row
        if transformer_ready:
            speedup = lat_b / max(lat_a, 1e-6)
            kpi_row([
                kpi_card("A · Accuracy", f"{report_a['accuracy']:.1%}",
                         delta=f"macro F1 {report_a['macro_f1']:.2f}", delta_kind="neutral"),
                kpi_card("B · Accuracy", f"{report_b['accuracy']:.1%}",
                         delta=f"macro F1 {report_b['macro_f1']:.2f}",
                         delta_kind="pos" if report_b["accuracy"] > report_a["accuracy"] else "neg"),
                kpi_card("A · Latency", f"{lat_a:.2f} ms",
                         delta="per document", delta_kind="pos"),
                kpi_card("B · Latency", f"{lat_b:.1f} ms",
                         delta=f"{speedup:.0f}× slower than A", delta_kind="neg"),
            ])
        else:
            kpi_row([
                kpi_card("A · Accuracy", f"{report_a['accuracy']:.1%}",
                         delta=f"macro F1 {report_a['macro_f1']:.2f}", delta_kind="neutral"),
                kpi_card("A · Macro F1", f"{report_a['macro_f1']:.2f}", delta_kind="neutral"),
                kpi_card("A · Avg latency", f"{lat_a:.2f} ms",
                         delta="per document", delta_kind="pos"),
                kpi_card("B · status", "Idle",
                         delta="Load transformer to compare", delta_kind="neutral"),
            ])

    # Confusion matrices
    def _confusion_fig(cm: np.ndarray, title: str) -> go.Figure:
        fig = go.Figure(go.Heatmap(
            z=cm, x=list(LABELS), y=list(LABELS),
            colorscale=[[0, PALETTE["surface_alt"]], [1, PALETTE["primary"]]],
            zmin=0, zmax=max(cm.max(), 1),
            colorbar=dict(title="count", tickfont=dict(color=PALETTE["text_muted"])),
            text=cm, texttemplate="%{text}",
            textfont=dict(color="white", size=14),
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        ))
        fig.update_layout(
            title=title, height=360,
            xaxis=dict(title="Predicted", side="bottom"),
            yaxis=dict(title="Actual", autorange="reversed"),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        return fig

    cm_cols = st.columns(2, gap="large")
    with cm_cols[0]:
        with section("Confusion matrix · A", "TF-IDF + LogReg"):
            st.plotly_chart(_confusion_fig(cm_a, ""), width="stretch")
    with cm_cols[1]:
        with section("Confusion matrix · B", "Transformer"):
            if transformer_ready:
                st.plotly_chart(_confusion_fig(cm_b, ""), width="stretch")
            else:
                st.info("Load the transformer to populate this matrix.")

    # Per-class report
    def _report_df(report: dict) -> pd.DataFrame:
        rows = []
        for lbl in LABELS:
            r = report["per_class"][lbl]
            rows.append({
                "Class": lbl,
                "Precision": r["precision"], "Recall": r["recall"],
                "F1": r["f1"], "Support": r["support"],
            })
        return pd.DataFrame(rows)

    with section("Per-class precision / recall / F1", "Both models on the same examples."):
        col_l, col_r = st.columns(2, gap="large")
        with col_l:
            st.caption("A · TF-IDF + LogReg")
            st.dataframe(
                _report_df(report_a), width="stretch", hide_index=True,
                column_config={
                    "Precision": st.column_config.ProgressColumn("Precision", min_value=0.0, max_value=1.0, format="%.2f"),
                    "Recall":    st.column_config.ProgressColumn("Recall",    min_value=0.0, max_value=1.0, format="%.2f"),
                    "F1":        st.column_config.ProgressColumn("F1",        min_value=0.0, max_value=1.0, format="%.2f"),
                },
            )
        with col_r:
            st.caption("B · Transformer")
            if transformer_ready:
                st.dataframe(
                    _report_df(report_b), width="stretch", hide_index=True,
                    column_config={
                        "Precision": st.column_config.ProgressColumn("Precision", min_value=0.0, max_value=1.0, format="%.2f"),
                        "Recall":    st.column_config.ProgressColumn("Recall",    min_value=0.0, max_value=1.0, format="%.2f"),
                        "F1":        st.column_config.ProgressColumn("F1",        min_value=0.0, max_value=1.0, format="%.2f"),
                    },
                )
            else:
                st.info("Load the transformer to populate this table.")

    # Latency chart
    with section(
        "Inference latency per document",
        "Wall-clock time per ticket, averaged over the benchmark. Log scale.",
    ):
        names = [traditional.name]
        latencies = [lat_a]
        colors = [PALETTE["accent"]]
        if transformer_ready:
            names.append("RoBERTa transformer")
            latencies.append(lat_b)
            colors.append(PALETTE["primary"])
        fig = go.Figure(go.Bar(
            x=names, y=latencies, marker=dict(color=colors),
            text=[f"{v:.2f} ms" for v in latencies], textposition="outside",
            hovertemplate="<b>%{x}</b><br>%{y:.3f} ms / doc<extra></extra>",
        ))
        fig.update_layout(
            height=320, yaxis=dict(title="ms / doc", type="log"),
            margin=dict(l=10, r=10, t=20, b=10), showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

    # Disagreement table
    if transformer_ready:
        with section(
            "Where the models disagree",
            "Examples where A and B produced different labels. Hover-worthy for class discussion.",
        ):
            diffs = []
            for txt, true, pa, pb in zip(texts, y_true, y_pred_a, y_pred_b):
                if pa != pb:
                    diffs.append({
                        "Text": txt,
                        "True": true,
                        "A · TF-IDF": pa,
                        "B · Transformer": pb,
                        "Winner": "B" if pb == true else ("A" if pa == true else "—"),
                    })
            if diffs:
                st.dataframe(pd.DataFrame(diffs), width="stretch", hide_index=True)
            else:
                st.success("Both models agreed on every benchmark example.")


# ── Theory ───────────────────────────────────────────────────────────────────
with tab_theory:
    with section(
        "Why is the transformer so much slower?",
        "A quick refresher on the asymptotic cost of each model.",
    ):
        st.markdown(
            """
            **TF-IDF + Logistic Regression** scans a document once to compute
            n-gram counts and then takes a dot product with a fixed weight
            vector:

            $$
            \\text{cost}_{\\text{tfidf}} \\;=\\; O(N) \\,+\\, O(V_{\\text{active}})
            $$

            where $N$ is the number of tokens in the document and
            $V_{\\text{active}}$ is the number of vocabulary terms that
            actually fire. Both terms are small and linear, so inference is
            measured in microseconds.

            **Self-attention** in a transformer compares every token with
            every other token in the input window. Per layer, the cost is:

            $$
            \\text{cost}_{\\text{attn}} \\;=\\; O\\!\\left(N^{2} \\cdot d\\right)
            $$

            with hidden size $d$, repeated across $L$ layers. That is the
            same model that lets it understand sarcasm and negation — and
            also the reason a single CPU inference can cost dozens of
            milliseconds per document.
            """,
        )

    with section(
        "When to pick which",
        "A pragmatic rubric for shipping decisions.",
    ):
        st.markdown(
            """
            | Scenario | Recommended model | Why |
            |---|---|---|
            | Real-time scoring on every ticket as it lands | **TF-IDF baseline** | Microsecond latency, no GPU |
            | Daily executive dashboards on millions of rows | **TF-IDF baseline** | Trivially parallel, cheap to recompute |
            | Edge cases: sarcasm, negation, multilingual nuance | **Transformer** | Context-aware embeddings |
            | Quality assurance on a small, high-stakes sample | **Transformer** | Highest accuracy on the long tail |
            | Cold-start product with no labels at all | **Transformer (zero-shot)** | No training data required |

            The master's takeaway is **not** "always pick the bigger model".
            It is to know exactly what each option costs — in dollars, in
            milliseconds, and in mistakes — and to compose a system that
            uses each tool where it shines.
            """,
        )
