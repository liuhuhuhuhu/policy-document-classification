"""
Hybrid Regex + Logistic Regression Classifier for Policy Indicator Detection

This script trains a weakly supervised multi-label classifier for legal/policy
text chunks. It combines:
1. Regex-based weak labels and explicit rule features
2. TF-IDF text features
3. One-vs-Rest Logistic Regression

Expected input columns:
- doc_id
- chunk_id
- chunk_text
- lang_chunk optional

Example:
    python src/logistic_regression_classifier.py \
        --input data/sample_chunks.csv \
        --output-dir results
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


# ---------------------------------------------------------------------
# Indicator definitions
# ---------------------------------------------------------------------

INDICATORS: Dict[str, Dict[str, str]] = {
    "5.1_passive_infra_sharing": {
        "question": "Does the text require or define sharing of passive telecom infrastructure?",
        "pattern": r"(infrastructure|mast|tower|duct|pole).*(share|sharing|co-?use|access)",
    },
    "5.2_foreign_equity_limits": {
        "question": "Does the text cap or limit foreign equity, ownership, or shareholding?",
        "pattern": r"(foreign|non[- ]?resident).*(equity|ownership|shareholding).*(cap|limit|restriction)",
    },
    "5.3_functional_or_accounting_separation": {
        "question": "Does the text impose functional or accounting separation for SMP or dominant operators?",
        "pattern": r"(functional|accounting)\s+separation|separate\s+accounts|SMP|dominant\s+operator",
    },
}

INDICATOR_KEYS: List[str] = list(INDICATORS.keys())


# ---------------------------------------------------------------------
# Sample data for demo mode
# ---------------------------------------------------------------------

SAMPLE_DATA = [
    {
        "doc_id": "D1",
        "chunk_id": 0,
        "chunk_text": "Applicants must provide access to telecom towers and ducts for infrastructure sharing.",
        "lang_chunk": "en",
    },
    {
        "doc_id": "D1",
        "chunk_id": 1,
        "chunk_text": "The regulator may require accounting separation for dominant operators.",
        "lang_chunk": "en",
    },
    {
        "doc_id": "D2",
        "chunk_id": 0,
        "chunk_text": "Foreign ownership in licensed operators shall not exceed the statutory cap.",
        "lang_chunk": "en",
    },
    {
        "doc_id": "D2",
        "chunk_id": 1,
        "chunk_text": "Patent applicants shall submit technical drawings and pay administrative fees.",
        "lang_chunk": "en",
    },
    {
        "doc_id": "D3",
        "chunk_id": 0,
        "chunk_text": "Mobile network operators may co-use masts, poles, and related facilities.",
        "lang_chunk": "en",
    },
    {
        "doc_id": "D3",
        "chunk_id": 1,
        "chunk_text": "The authority publishes annual reports and procedural guidelines.",
        "lang_chunk": "en",
    },
]


# ---------------------------------------------------------------------
# Regex utilities
# ---------------------------------------------------------------------

def regex_hit(text: str, pattern: str) -> Tuple[int, str]:
    """
    Return whether a regex pattern matches and a short evidence snippet.

    Args:
        text: Input text chunk.
        pattern: Regex pattern for one indicator.

    Returns:
        A tuple of (hit, evidence_snippet).
    """
    if not isinstance(text, str):
        return 0, ""

    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        start = max(0, match.start() - 60)
        end = min(len(text), match.end() + 60)
        snippet = text[start:end].replace("\n", " ")
        return 1, snippet[:250]

    return 0, ""


def build_regex_matrix(
    texts: List[str],
    indicators: Dict[str, Dict[str, str]],
) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    """
    Build regex feature matrix and evidence records.

    Returns:
        feature_matrix: shape [n_samples, n_indicators]
        evidence_records: one evidence dictionary per text chunk
    """
    feature_rows = []
    evidence_rows = []

    for text in texts:
        feature_row = []
        evidence_row = {}

        for key, config in indicators.items():
            hit, evidence = regex_hit(text, config["pattern"])
            feature_row.append(hit)
            evidence_row[key] = evidence

        feature_rows.append(feature_row)
        evidence_rows.append(evidence_row)

    return np.array(feature_rows, dtype=float), evidence_rows


def regex_feature_block(texts_array):
    """
    Convert raw text into regex-hit features for sklearn ColumnTransformer.
    """
    texts = pd.Series(texts_array.ravel()).fillna("").astype(str).tolist()
    features, _ = build_regex_matrix(texts, INDICATORS)
    return features


# ---------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------

def load_chunks(input_path: str | None = None) -> pd.DataFrame:
    """
    Load chunk-level data from CSV or use built-in sample data.

    The input file should contain:
    - doc_id
    - chunk_id
    - chunk_text
    - lang_chunk optional
    """
    if input_path is None:
        df = pd.DataFrame(SAMPLE_DATA)
    else:
        df = pd.read_csv(input_path)

    required = {"doc_id", "chunk_id", "chunk_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "lang_chunk" not in df.columns:
        df["lang_chunk"] = ""

    df["chunk_text"] = df["chunk_text"].fillna("").astype(str)
    return df


def create_weak_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[Dict[str, str]]]:
    """
    Create weak labels from regex hits.
    """
    regex_y, regex_evidence = build_regex_matrix(df["chunk_text"].tolist(), INDICATORS)
    weak_labels = pd.DataFrame(regex_y, columns=[f"y__{key}" for key in INDICATOR_KEYS])
    df_weak = pd.concat([df.reset_index(drop=True), weak_labels], axis=1)
    return df_weak, regex_y, regex_evidence


# ---------------------------------------------------------------------
# Model training and prediction
# ---------------------------------------------------------------------

def build_model():
    """
    Build TF-IDF + regex feature pipeline with One-vs-Rest Logistic Regression.
    """
    regex_featurizer = FunctionTransformer(regex_feature_block, validate=False)

    features = ColumnTransformer(
        transformers=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=30_000,
                    min_df=1,
                    lowercase=True,
                ),
                0,
            ),
            ("regex", regex_featurizer, 0),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = make_pipeline(
        features,
        OneVsRestClassifier(
            LogisticRegression(
                penalty="l2",
                solver="liblinear",
                class_weight="balanced",
                max_iter=500,
            )
        ),
    )

    return model


def safe_train_test_split(X: np.ndarray, Y: np.ndarray):
    """
    Split data when possible. If the dataset is too small, train and evaluate
    on the full sample for demo purposes.
    """
    n_samples = len(X)

    if n_samples < 8:
        return X, X, Y, Y

    try:
        return train_test_split(X, Y, test_size=0.25, random_state=42)
    except ValueError:
        return X, X, Y, Y


def get_probability_matrix(model, X: np.ndarray) -> np.ndarray:
    """
    Standardize predict_proba output into [n_samples, n_labels].
    """
    proba = model.predict_proba(X)

    if isinstance(proba, list):
        return np.column_stack([p[:, 1] for p in proba])

    return np.asarray(proba)


def train_classifier(df_weak: pd.DataFrame):
    """
    Train classifier using weak labels generated from regex rules.
    """
    X_all = df_weak["chunk_text"].values.reshape(-1, 1)
    Y_all = df_weak[[f"y__{key}" for key in INDICATOR_KEYS]].values

    X_train, X_test, y_train, y_test = safe_train_test_split(X_all, Y_all)

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        target_names=INDICATOR_KEYS,
        zero_division=0,
        output_dict=True,
    )

    return model, report


def build_outputs(
    df: pd.DataFrame,
    model,
    regex_y: np.ndarray,
    regex_evidence: List[Dict[str, str]],
    threshold: float = 0.5,
    alpha: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build chunk-level and document-level output tables.
    """
    X_all = df["chunk_text"].values.reshape(-1, 1)
    probability_matrix = get_probability_matrix(model, X_all)

    proba_df = pd.DataFrame(
        probability_matrix,
        columns=[f"proba__{key}" for key in INDICATOR_KEYS],
    )

    regex_hits = (regex_y > 0).astype(float)
    blended_scores = np.maximum(regex_hits, alpha * probability_matrix)

    score_df = pd.DataFrame(
        blended_scores,
        columns=[f"score__{key}" for key in INDICATOR_KEYS],
    )

    pred_binary = (probability_matrix >= threshold).astype(int)
    pred_df = pd.DataFrame(
        pred_binary,
        columns=[f"pred__{key}" for key in INDICATOR_KEYS],
    )

    evidence_df = pd.DataFrame(regex_evidence).rename(
        columns={key: f"evidence__{key}" for key in INDICATOR_KEYS}
    )

    chunk_output = pd.concat(
        [
            df.reset_index(drop=True)[["doc_id", "chunk_id", "chunk_text", "lang_chunk"]],
            proba_df,
            pred_df,
            score_df,
            evidence_df,
        ],
        axis=1,
    )

    doc_output = aggregate_doc_level(chunk_output)
    return chunk_output, doc_output


def aggregate_doc_level(chunk_table: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate chunk-level predictions into document-level indicator scores.
    """
    rows = []

    for doc_id, sub in chunk_table.groupby("doc_id"):
        for key in INDICATOR_KEYS:
            any_hit = int(sub[f"pred__{key}"].max() > 0)
            raw_sum = float(sub[f"score__{key}"].sum())
            doc_score_capped = float(min(1.0, raw_sum))

            rows.append(
                {
                    "doc_id": doc_id,
                    "indicator": key,
                    "scope": INDICATORS[key]["question"],
                    "any_hit": bool(any_hit),
                    "raw_sum": round(raw_sum, 4),
                    "doc_score_capped": round(doc_score_capped, 4),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Export utilities
# ---------------------------------------------------------------------

def export_jsonl(records: List[dict], output_path: Path) -> None:
    """
    Export a list of dictionaries to JSONL.
    """
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def export_results(
    chunk_output: pd.DataFrame,
    doc_output: pd.DataFrame,
    report: dict,
    output_dir: str,
) -> None:
    """
    Export chunk-level scores, document-level scores, and model report.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    chunk_records = []
    for _, row in chunk_output.iterrows():
        record = {
            "doc_id": row["doc_id"],
            "chunk_id": int(row["chunk_id"]),
            "lang_chunk": row.get("lang_chunk", ""),
            "text": str(row["chunk_text"])[:1000],
            "pred": {key: int(row[f"pred__{key}"]) for key in INDICATOR_KEYS},
            "proba": {key: float(row[f"proba__{key}"]) for key in INDICATOR_KEYS},
            "score": {key: float(row[f"score__{key}"]) for key in INDICATOR_KEYS},
            "evidence": {key: str(row.get(f"evidence__{key}", "")) for key in INDICATOR_KEYS},
        }
        chunk_records.append(record)

    doc_records = doc_output.to_dict(orient="records")

    export_jsonl(chunk_records, output_path / "pdf_chunks_scored.jsonl")
    export_jsonl(doc_records, output_path / "pdf_docs_scored.jsonl")

    with (output_path / "model_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    chunk_output.to_csv(output_path / "chunk_level_predictions.csv", index=False)
    doc_output.to_csv(output_path / "document_level_scores.csv", index=False)


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a hybrid regex + logistic regression policy indicator classifier."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input CSV with doc_id, chunk_id, chunk_text, and optional lang_chunk.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory where outputs will be saved.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for binary predictions.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Weight applied to logistic regression probabilities before blending with regex hits.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_chunks(args.input)
    df_weak, regex_y, regex_evidence = create_weak_labels(df)

    model, report = train_classifier(df_weak)

    chunk_output, doc_output = build_outputs(
        df=df,
        model=model,
        regex_y=regex_y,
        regex_evidence=regex_evidence,
        threshold=args.threshold,
        alpha=args.alpha,
    )

    export_results(
        chunk_output=chunk_output,
        doc_output=doc_output,
        report=report,
        output_dir=args.output_dir,
    )

    print(f"Processed {len(df)} chunks across {df['doc_id'].nunique()} documents.")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
