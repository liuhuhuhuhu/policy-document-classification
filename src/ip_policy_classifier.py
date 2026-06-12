"""
Intellectual Property Policy Text Classifier

This module provides three lightweight ways to screen whether a policy text is
related to intellectual property (IP):

1. TF-IDF cosine similarity against IP and non-IP reference texts
2. TF-IDF + Naive Bayes classifier trained on small labeled examples
3. Optional zero-shot classification with a Hugging Face NLI model

The goal is quick, explainable policy-document screening before more detailed
manual review or downstream classification.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline


IP_REFERENCE = """
Intellectual property policy covering patents, trademarks, copyrights,
trade secrets, IP licensing agreements, royalty payments, infringement,
enforcement, WIPO standards, TRIPS obligations, technology transfer,
patent pools, industrial designs, geographical indications, and innovation incentives.
"""

NON_IP_REFERENCE = """
General financial regulation including licensing for financial institutions,
prudential supervision, AML/CFT, digital payments, regulatory sandboxes,
consumer protection, digital banking, data governance, and financial digitalisation.
"""

DEFAULT_TRAINING_TEXTS = [
    # IP-related examples
    "Policy on patents and copyrights protection, IP licensing, and royalties.",
    "Trademark enforcement and technology transfer guidelines.",
    "Copyright exceptions and fair use in digital content.",
    "IP licensing agreements for software and patent pools.",
    "Measures to prevent patent infringement and protect industrial designs.",
    "WIPO-aligned intellectual property enforcement and TRIPS compliance.",
    # Non-IP examples
    "Licensing requirements for digital banks and payment providers.",
    "Regulatory framework for AML/CFT and consumer protection.",
    "Prudential supervision of financial institutions.",
    "Open banking policy and data governance for financial digitalisation.",
    "Regulatory sandbox for fintech companies and digital payments.",
    "Financial-sector compliance rules for electronic money issuers.",
]

DEFAULT_TRAINING_LABELS = [
    1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0,
]


@dataclass
class ClassificationResult:
    text: str
    cosine_ip_similarity: float
    cosine_non_ip_similarity: float
    cosine_label: str
    naive_bayes_ip_probability: float
    naive_bayes_non_ip_probability: float
    naive_bayes_label: str
    zero_shot_label: Optional[str] = None
    zero_shot_scores: Optional[Dict[str, float]] = None
    final_label: Optional[str] = None


def classify_with_cosine(
    text: str,
    ip_reference: str = IP_REFERENCE,
    non_ip_reference: str = NON_IP_REFERENCE,
    threshold: float = 0.25,
) -> Dict[str, float | str]:
    """Classify text by TF-IDF cosine similarity to reference descriptions."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    tfidf = vectorizer.fit_transform([ip_reference, non_ip_reference, text])

    ip_similarity = float(cosine_similarity(tfidf[2], tfidf[0])[0, 0])
    non_ip_similarity = float(cosine_similarity(tfidf[2], tfidf[1])[0, 0])

    if ip_similarity > non_ip_similarity and ip_similarity >= threshold:
        label = "IP-related"
    else:
        label = "Non-IP-related"

    return {
        "ip_similarity": ip_similarity,
        "non_ip_similarity": non_ip_similarity,
        "label": label,
    }


def train_naive_bayes_classifier(
    texts: Sequence[str] = DEFAULT_TRAINING_TEXTS,
    labels: Sequence[int] = DEFAULT_TRAINING_LABELS,
) -> Pipeline:
    """Train a simple TF-IDF + Multinomial Naive Bayes classifier."""
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), stop_words="english"),
        MultinomialNB(),
    )
    model.fit(list(texts), list(labels))
    return model


def classify_with_naive_bayes(text: str, model: Pipeline) -> Dict[str, float | str]:
    """Return Naive Bayes probabilities and predicted label."""
    probabilities = model.predict_proba([text])[0]
    predicted_class = int(np.argmax(probabilities))

    return {
        "ip_probability": float(probabilities[1]),
        "non_ip_probability": float(probabilities[0]),
        "label": "IP-related" if predicted_class == 1 else "Non-IP-related",
    }


def classify_with_zero_shot(
    text: str,
    model_name: str = "facebook/bart-large-mnli",
) -> Dict[str, object]:
    """
    Optional zero-shot classification using a Hugging Face NLI model.

    This requires transformers and torch. It is kept optional because it may
    download a large model the first time it runs.
    """
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "Zero-shot classification requires transformers and torch. "
            "Install them with: pip install transformers torch"
        ) from exc

    classifier = pipeline("zero-shot-classification", model=model_name)
    labels = ["IP-related", "Non-IP-related"]
    result = classifier(
        text,
        candidate_labels=labels,
        hypothesis_template="This policy document is {}.",
        multi_label=False,
    )

    scores = {
        label: float(score)
        for label, score in zip(result["labels"], result["scores"])
    }

    return {
        "label": result["labels"][0],
        "scores": scores,
    }


def classify_policy_text(text: str, use_zero_shot: bool = False) -> ClassificationResult:
    """Run all enabled classifiers and return a structured result."""
    cosine_result = classify_with_cosine(text)
    nb_model = train_naive_bayes_classifier()
    nb_result = classify_with_naive_bayes(text, nb_model)

    zero_label = None
    zero_scores = None
    labels_for_vote = [cosine_result["label"], nb_result["label"]]

    if use_zero_shot:
        zero_result = classify_with_zero_shot(text)
        zero_label = str(zero_result["label"])
        zero_scores = dict(zero_result["scores"])
        labels_for_vote.append(zero_label)

    final_label = max(set(labels_for_vote), key=labels_for_vote.count)

    return ClassificationResult(
        text=text,
        cosine_ip_similarity=float(cosine_result["ip_similarity"]),
        cosine_non_ip_similarity=float(cosine_result["non_ip_similarity"]),
        cosine_label=str(cosine_result["label"]),
        naive_bayes_ip_probability=float(nb_result["ip_probability"]),
        naive_bayes_non_ip_probability=float(nb_result["non_ip_probability"]),
        naive_bayes_label=str(nb_result["label"]),
        zero_shot_label=zero_label,
        zero_shot_scores=zero_scores,
        final_label=final_label,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify whether a policy text is IP-related."
    )
    parser.add_argument(
        "--text",
        type=str,
        default=(
            "Sep 7, 2024 ... BNM, DITOs, Policy Document on Licensing "
            "and Regulatory Framework, Financial digitalisation."
        ),
        help="Policy text to classify.",
    )
    parser.add_argument(
        "--zero-shot",
        action="store_true",
        help="Also run Hugging Face zero-shot classification.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save JSON output.",
    )

    args = parser.parse_args()
    result = classify_policy_text(args.text, use_zero_shot=args.zero_shot)
    result_dict = asdict(result)

    print(json.dumps(result_dict, indent=2, ensure_ascii=False))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(result_dict, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
