"""
Pillar 5 Indicator Classifier

This script classifies policy text chunks into RDTII-style Pillar 5 telecom
indicators using a hybrid approach:

1. Regex rules create weak labels and interpretable evidence.
2. TF-IDF features represent the policy text.
3. One-vs-Rest Logistic Regression learns flexible patterns from weak labels.
4. Regex and model probabilities are blended for final chunk-level predictions.

Expected input CSV columns:
- text_clean or text_raw
Optional:
- doc_id
- chunk_id
- lang_chunk

Example:
    python src/pillar5_indicator_classifier.py \
        --input data/sample_chunks.csv \
        --output results/pillar5_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier


QUESTION_LIST: Dict[str, List[str]] = {
    "5.1": [
        "require or regulate passive infrastructure sharing",
        "mandatory or required sharing",
        "define scope of passive infrastructure such as towers, ducts, or poles",
        "procedures or criteria for access or disputes",
        "charges, price principles, or non-discrimination for access",
        "regulator can impose sharing in rural or new rollout areas",
    ],
    "5.2": [
        "ban on foreign ownership in the telecom sector",
        "maximum percentage of foreign equity, ownership, or shareholding",
        "domestic control or majority must be kept",
        "numeric ceilings such as 49% or less than 50%",
        "limits for specific telecom activities",
        "different limits for SOEs/public operators and private firms",
    ],
    "5.3": [
        "functional, operational, or structural separation obligations",
        "accounting separation such as separate books, ledgers, or cost accounts",
        "definition of SMP or dominant operator",
        "non-discrimination and cost transparency with separated accounts",
        "regulator authority to impose separation after market analysis",
    ],
    "5.4": [
        "economy is not a signatory, not appended, or not committed to the WTO Telecom Reference Paper",
        "Reference Paper is not applied or not incorporated into schedule of commitments",
    ],
    "5.5": [
        "independent telecom regulator or authority separate from operators and ministry",
        "functional independence such as no interference or no conflicts of interest",
        "institutional or financial independence such as legal status and stable budget",
        "no preferential treatment and equal application of procedures",
        "explicit lack of independence such as direct control or budgetary control",
    ],
}


REGEX_PATTERNS: Dict[str, List[str]] = {
    "5.1": [
        r"passive infrastructure sharing",
        r"(tower|mast|site|duct|pole|cabinet|tray)s?\s+(sharing|access)",
        r"mandatory\s+sharing|required\s+sharing",
        r"co-?location|infrastructure\s+sharing",
        r"non[-\s]?discriminat(e|ion).{0,30}(access|sharing|infrastructure)",
    ],
    "5.2": [
        r"foreign (ownership|equity|shareholding|investment)",
        r"maximum\s+\d{1,2}%\s+foreign",
        r"(foreign|non-?citizen).{0,30}(cap|limit|ceiling)",
        r"\b49%\b|\bless than 50%\b",
        r"domestic (control|majority)|must remain (domestic|national)",
    ],
    "5.3": [
        r"(functional|structural|operational)\s+separation",
        r"accounting\s+separation|separate\s+(accounts|books|ledgers)",
        r"significant market power|SMP|dominant operator",
        r"cost\s+transparency|non[-\s]?discrimination",
        r"regulator.{0,40}impose\s+separation|market\s+analysis",
    ],
    "5.4": [
        r"(WTO )?Telecom(munications)? Reference Paper",
        r"not (a )?(signatory|party|committed|appended)",
        r"Reference Paper.{0,20}(not applied|not incorporated)",
    ],
    "5.5": [
        r"independent (telecom|telecommunications) (regulator|authority|commission)",
        r"no (political )?interference|conflict of interest",
        r"financial independence|separate legal status|stable budget",
        r"preferential treatment|equal application",
        r"lack of independence|under direct control",
    ],
}


INDICATOR_KEYS: List[str] = list(REGEX_PATTERNS.keys())


def clean_text(text: object) -> str:
    """Normalize whitespace and safely convert missing values to empty strings."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    value = str(text)
    value = re.sub(r"\s+", " ", value).strip()
    return "" if value.lower() == "nan" else value


def regex_hit(text: str, patterns: List[str], window: int = 80) -> Tuple[int, str]:
    """
    Return whether any pattern matches and the first evidence snippet.

    Parameters
    ----------
    text:
        Policy text chunk.
    patterns:
        Regex patterns for one indicator.
    window:
        Number of characters to keep before/after the match as evidence.
    """
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            return 1, text[start:end].strip()
    return 0, ""


def build_weak_labels(
    texts: List[str],
    patterns: Dict[str, List[str]],
) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    """Create weak-label matrix and evidence snippets from regex patterns."""
    labels = np.zeros((len(texts), len(patterns)), dtype=int)
    evidence_records: List[Dict[str, str]] = []

    for i, text in enumerate(texts):
        row_evidence: Dict[str, str] = {}
        for j, indicator in enumerate(patterns.keys()):
            hit, evidence = regex_hit(text, patterns[indicator])
            labels[i, j] = hit
            row_evidence[indicator] = evidence
        evidence_records.append(row_evidence)

    return labels, evidence_records


def choose_text_column(df: pd.DataFrame, requested_col: str | None = None) -> str:
    """Select the text column used for modeling."""
    if requested_col:
        if requested_col not in df.columns:
            raise ValueError(f"Requested text column '{requested_col}' not found.")
        return requested_col

    if "text_clean" in df.columns:
        return "text_clean"
    if "text_raw" in df.columns:
        return "text_raw"
    if "chunk_text" in df.columns:
        return "chunk_text"

    raise ValueError("Input file must contain one of: text_clean, text_raw, chunk_text.")


def ensure_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create standard metadata columns if they are missing."""
    output = df.copy()
    if "doc_id" not in output.columns:
        output["doc_id"] = "doc_1"
    if "chunk_id" not in output.columns:
        output["chunk_id"] = np.arange(len(output))
    if "lang_chunk" not in output.columns:
        output["lang_chunk"] = ""
    return output


def can_train_model(y: np.ndarray) -> bool:
    """
    Check whether Logistic Regression can be trained.

    Training is skipped if all labels are only 0 or only 1 across all samples.
    """
    if len(y) < 4:
        return False
    return any(len(np.unique(y[:, j])) > 1 for j in range(y.shape[1]))


def train_classifier(
    texts: List[str],
    weak_labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train TF-IDF + One-vs-Rest Logistic Regression.

    Returns
    -------
    vectorizer, classifier
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        max_features=50_000,
    )

    x = vectorizer.fit_transform(texts)

    clf = OneVsRestClassifier(
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
        )
    )

    if len(texts) >= 8:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            weak_labels,
            test_size=test_size,
            random_state=random_state,
        )
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)
        print("=== Validation against regex weak labels ===")
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=INDICATOR_KEYS,
                zero_division=0,
            )
        )
    else:
        clf.fit(x, weak_labels)

    return vectorizer, clf


def predict_with_blending(
    texts: List[str],
    vectorizer,
    classifier,
    weak_labels: np.ndarray,
    regex_floor: float = 0.8,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict probabilities and blend model probabilities with regex hits.

    If regex hits an indicator, its probability is forced to at least regex_floor.
    """
    x = vectorizer.transform(texts)
    probabilities = classifier.predict_proba(x)

    if isinstance(probabilities, list):
        probabilities = np.column_stack([p[:, 1] for p in probabilities])

    blended = probabilities.copy()
    for j in range(len(INDICATOR_KEYS)):
        blended[:, j] = np.where(
            weak_labels[:, j] == 1,
            np.maximum(blended[:, j], regex_floor),
            blended[:, j],
        )

    predictions = (blended >= threshold).astype(int)
    return blended, predictions


def build_output_table(
    df: pd.DataFrame,
    text_col: str,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    evidence_records: List[Dict[str, str]],
) -> pd.DataFrame:
    """Attach probabilities, predictions, and evidence snippets to the input data."""
    output = ensure_metadata_columns(df)
    output = output.reset_index(drop=True)

    for j, indicator in enumerate(INDICATOR_KEYS):
        output[f"prob_{indicator}"] = probabilities[:, j]
        output[f"pred_{indicator}"] = predictions[:, j]
        output[f"evidence_{indicator}"] = [
            row.get(indicator, "") for row in evidence_records
        ]

    prob_cols = [f"prob_{indicator}" for indicator in INDICATOR_KEYS]
    output["best_indicator"] = output[prob_cols].idxmax(axis=1).str.replace("prob_", "")

    keep_cols = ["doc_id", "chunk_id", "lang_chunk", text_col]
    model_cols = [
        col
        for col in output.columns
        if col.startswith("prob_") or col.startswith("pred_") or col.startswith("evidence_")
    ]

    return output[keep_cols + ["best_indicator"] + model_cols]


def aggregate_document_scores(chunk_output: pd.DataFrame) -> pd.DataFrame:
    """Aggregate chunk-level predictions into document-level indicator scores."""
    rows = []

    for doc_id, group in chunk_output.groupby("doc_id"):
        for indicator in INDICATOR_KEYS:
            prob_col = f"prob_{indicator}"
            pred_col = f"pred_{indicator}"
            evidence_col = f"evidence_{indicator}"

            evidence_samples = [
                value for value in group[evidence_col].dropna().astype(str).tolist() if value
            ][:3]

            rows.append(
                {
                    "doc_id": doc_id,
                    "indicator": indicator,
                    "question": "; ".join(QUESTION_LIST[indicator]),
                    "any_hit": bool(group[pred_col].max() > 0),
                    "max_probability": float(group[prob_col].max()),
                    "mean_probability": float(group[prob_col].mean()),
                    "evidence_samples": evidence_samples,
                }
            )

    return pd.DataFrame(rows)


def save_jsonl(records: List[dict], path: Path) -> None:
    """Save records to a JSONL file."""
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    text_col: str | None = None,
    threshold: float = 0.5,
    regex_floor: float = 0.8,
) -> None:
    """Run the full Pillar 5 classification pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    selected_text_col = choose_text_column(df, text_col)

    df[selected_text_col] = df[selected_text_col].map(clean_text)
    df = df[df[selected_text_col] != ""].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid text rows found after cleaning.")

    texts = df[selected_text_col].tolist()
    weak_labels, evidence_records = build_weak_labels(texts, REGEX_PATTERNS)

    if can_train_model(weak_labels):
        vectorizer, classifier = train_classifier(texts, weak_labels)
        probabilities, predictions = predict_with_blending(
            texts,
            vectorizer,
            classifier,
            weak_labels,
            regex_floor=regex_floor,
            threshold=threshold,
        )
    else:
        print("Not enough label variation to train Logistic Regression. Using regex-only scores.")
        probabilities = np.where(weak_labels == 1, regex_floor, 0.0)
        predictions = (probabilities >= threshold).astype(int)

    chunk_output = build_output_table(
        df,
        selected_text_col,
        probabilities,
        predictions,
        evidence_records,
    )
    doc_output = aggregate_document_scores(chunk_output)

    chunk_csv = output_dir / "pillar5_chunk_predictions.csv"
    doc_csv = output_dir / "pillar5_document_scores.csv"
    chunk_jsonl = output_dir / "pillar5_chunk_predictions.jsonl"
    doc_jsonl = output_dir / "pillar5_document_scores.jsonl"

    chunk_output.to_csv(chunk_csv, index=False, encoding="utf-8-sig")
    doc_output.to_csv(doc_csv, index=False, encoding="utf-8-sig")
    save_jsonl(chunk_output.to_dict(orient="records"), chunk_jsonl)
    save_jsonl(doc_output.to_dict(orient="records"), doc_jsonl)

    print(f"Saved chunk predictions to: {chunk_csv}")
    print(f"Saved document scores to: {doc_csv}")
    print(f"Saved chunk JSONL to: {chunk_jsonl}")
    print(f"Saved document JSONL to: {doc_jsonl}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pillar 5 telecom indicator classifier.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV containing text_clean, text_raw, or chunk_text.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for output CSV and JSONL files.",
    )
    parser.add_argument(
        "--text-col",
        default=None,
        help="Optional text column name. Defaults to text_clean, text_raw, or chunk_text.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for final predictions.",
    )
    parser.add_argument(
        "--regex-floor",
        type=float,
        default=0.8,
        help="Minimum probability assigned when regex hits.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        text_col=args.text_col,
        threshold=args.threshold,
        regex_floor=args.regex_floor,
    )
