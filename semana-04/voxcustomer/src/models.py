"""Sentiment models used across VoxCustomer.

Two backends with the same public interface:

* ``TraditionalModel`` — TF-IDF (word + char n-grams) + LogisticRegression.
  Trains in ~1s on the bundled :mod:`src.corpus` training set. Pure
  scikit-learn, zero-network dependency, microseconds per prediction.

* ``TransformerModel`` — ``cardiffnlp/twitter-roberta-base-sentiment-latest``
  via Hugging Face Transformers. Loaded lazily, downloaded on first use,
  cached on disk. Heavyweight but contextual.

Both expose ``predict_one(text)`` and ``predict_batch(texts)`` returning
``(label, score, latency_ms)`` or a list of such tuples.

The module is safe to import even when ``transformers`` or ``torch`` are
not installed: the transformer class will simply report unavailability
when instantiated.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .corpus import EVAL_CORPUS, LABELS, TRAINING_CORPUS


# ── Shared types ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Prediction:
    label: str
    score: float          # confidence in [0, 1] for the predicted class
    neg_score: float      # probability mass assigned to Negative (0..1)
    latency_ms: float


# ── Traditional baseline ─────────────────────────────────────────────────────
class TraditionalModel:
    """TF-IDF (word + char) + LogisticRegression. Bag-of-words world view."""

    name = "TF-IDF + Logistic Regression"
    family = "Traditional / Statistical"

    def __init__(self) -> None:
        self.classes_ = list(LABELS)
        self._pipe: Pipeline = self._build_and_fit()

    def _build_and_fit(self) -> Pipeline:
        texts = [t for t, _ in TRAINING_CORPUS]
        labels = [y for _, y in TRAINING_CORPUS]
        features = FeatureUnion([
            ("word", TfidfVectorizer(
                lowercase=True, ngram_range=(1, 2),
                min_df=1, max_df=0.95, sublinear_tf=True,
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5),
                min_df=1, sublinear_tf=True,
            )),
        ])
        clf = LogisticRegression(
            max_iter=1000, C=2.0, class_weight="balanced", n_jobs=None,
        )
        pipe = Pipeline([("features", features), ("clf", clf)])
        pipe.fit(texts, labels)
        return pipe

    def _proba(self, texts: Sequence[str]) -> np.ndarray:
        probs = self._pipe.predict_proba(list(texts))
        # Reorder columns into canonical LABELS order
        order = [list(self._pipe.classes_).index(c) for c in self.classes_]
        return probs[:, order]

    def predict_one(self, text: str) -> Prediction:
        t0 = time.perf_counter()
        probs = self._proba([text])[0]
        latency = (time.perf_counter() - t0) * 1000
        idx = int(np.argmax(probs))
        return Prediction(
            label=self.classes_[idx],
            score=float(probs[idx]),
            neg_score=float(probs[self.classes_.index("Negative")]),
            latency_ms=latency,
        )

    def predict_batch(self, texts: Sequence[str]) -> list[Prediction]:
        texts = list(texts)
        t0 = time.perf_counter()
        probs = self._proba(texts)
        total_latency = (time.perf_counter() - t0) * 1000
        per_doc = total_latency / max(len(texts), 1)
        neg_idx = self.classes_.index("Negative")
        out: list[Prediction] = []
        for row in probs:
            idx = int(np.argmax(row))
            out.append(Prediction(
                label=self.classes_[idx],
                score=float(row[idx]),
                neg_score=float(row[neg_idx]),
                latency_ms=per_doc,
            ))
        return out


# ── Transformer ──────────────────────────────────────────────────────────────
class TransformerModel:
    """Lightweight wrapper around a 3-class sentiment transformer."""

    name = "RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest)"
    family = "Deep Learning / Transformer"
    model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    # The remote model emits LABEL_0/1/2 → negative/neutral/positive.
    _ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}

    def __init__(self) -> None:
        self.classes_ = list(LABELS)
        self.available = False
        self.unavailable_reason: str | None = None
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device = "cpu"
        self._load()

    def _load(self) -> None:
        try:
            import torch  # noqa: WPS433
            from transformers import (  # noqa: WPS433
                AutoModelForSequenceClassification, AutoTokenizer,
            )
        except Exception as exc:  # ImportError or runtime issues
            self.unavailable_reason = (
                "The `transformers` / `torch` packages are not installed in this "
                f"environment ({exc.__class__.__name__})."
            )
            return

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            # Force safetensors: torch < 2.6 refuses to load legacy .bin weights
            # under CVE-2025-32434, but safetensors is unaffected.
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id, use_safetensors=True,
            )
            self._model.eval()
            self._device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
            self._model.to(self._device)
            self._torch = torch
            self.available = True
        except Exception as exc:
            self.unavailable_reason = (
                "Could not download or load the transformer weights "
                f"({exc.__class__.__name__}: {exc}). "
                "Check your internet connection or pre-cache the model."
            )

    def _proba(self, texts: Sequence[str]) -> np.ndarray:
        assert self.available, "TransformerModel is not available."
        torch = self._torch
        enc = self._tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=256, return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            logits = self._model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        # Reorder model columns into canonical LABELS order
        # Model order: 0 Negative, 1 Neutral, 2 Positive — happens to match
        # canonical, but we map explicitly to be safe.
        n = probs.shape[1]
        col_for = {self._ID2LABEL[i]: i for i in range(n)}
        order = [col_for[c] for c in self.classes_]
        return probs[:, order]

    def predict_one(self, text: str) -> Prediction:
        if not self.available:
            raise RuntimeError(self.unavailable_reason or "Transformer not available")
        t0 = time.perf_counter()
        probs = self._proba([text])[0]
        latency = (time.perf_counter() - t0) * 1000
        idx = int(np.argmax(probs))
        return Prediction(
            label=self.classes_[idx],
            score=float(probs[idx]),
            neg_score=float(probs[self.classes_.index("Negative")]),
            latency_ms=latency,
        )

    def predict_batch(self, texts: Sequence[str], batch_size: int = 16) -> list[Prediction]:
        if not self.available:
            raise RuntimeError(self.unavailable_reason or "Transformer not available")
        texts = list(texts)
        out: list[Prediction] = []
        neg_idx = self.classes_.index("Negative")
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            t0 = time.perf_counter()
            probs = self._proba(chunk)
            total_latency = (time.perf_counter() - t0) * 1000
            per_doc = total_latency / max(len(chunk), 1)
            for row in probs:
                idx = int(np.argmax(row))
                out.append(Prediction(
                    label=self.classes_[idx],
                    score=float(row[idx]),
                    neg_score=float(row[neg_idx]),
                    latency_ms=per_doc,
                ))
        return out


# ── Streamlit-cached factories ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Training the TF-IDF baseline…")
def get_traditional_model() -> TraditionalModel:
    return TraditionalModel()


@st.cache_resource(show_spinner="Loading the transformer (first run downloads ~500 MB)…")
def get_transformer_model() -> TransformerModel:
    return TransformerModel()


# ── Evaluation helpers ───────────────────────────────────────────────────────
def evaluation_set() -> tuple[list[str], list[str]]:
    texts = [t for t, _ in EVAL_CORPUS]
    labels = [y for _, y in EVAL_CORPUS]
    return texts, labels


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str]) -> np.ndarray:
    idx = {lbl: i for i, lbl in enumerate(LABELS)}
    m = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[idx[t], idx[p]] += 1
    return m


def classification_report(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    """Per-class precision / recall / F1 and overall accuracy."""
    cm = confusion_matrix(y_true, y_pred)
    report: dict = {"per_class": {}, "accuracy": 0.0, "macro_f1": 0.0, "support": len(y_true)}
    accuracies = 0
    f1s = []
    for i, lbl in enumerate(LABELS):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        report["per_class"][lbl] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }
        accuracies += tp
        f1s.append(f1)
    report["accuracy"] = float(accuracies / max(len(y_true), 1))
    report["macro_f1"] = float(np.mean(f1s))
    return report
