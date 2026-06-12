"""
Telecom Indicator Classification Pipeline

This script extracts text from policy/regulatory PDF documents, splits the text
into overlapping chunks, detects the dominant language of each chunk, and applies
regex-based rules to identify telecom-related RDTII-style indicators.

Outputs:
- pdf_chunks_scored.jsonl: chunk-level scores and evidence
- scores.jsonl: document-level indicator scores with evidence samples

Example:
    python src/telecom_indicator_classification.py \
        --pdf data/raw/sample_policy.pdf \
        --output-dir results
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import pdfplumber
from langdetect import detect_langs
from pypdf import PdfReader


# ---------------------------------------------------------------------------
# 1. Text extraction
# ---------------------------------------------------------------------------

def extract_text_direct(pdf_path: str | Path) -> str:
    """
    Extract text directly from a PDF.

    The function first tries pdfplumber, then falls back to pypdf.
    This works best for machine-readable PDFs.
    """
    pdf_path = Path(pdf_path)
    text_parts: List[str] = []

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        text = "\n".join(text_parts).strip()
        if text:
            return text
    except Exception:
        pass

    text_parts = []
    try:
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts).strip()
    except Exception:
        return ""


def extract_text_ocr(pdf_path: str | Path, dpi: int = 300, lang: str = "eng+msa") -> str:
    """
    Extract text from scanned PDFs using OCR.

    Requirements:
    - Python packages: pdf2image, pytesseract, pillow
    - System tools: poppler and tesseract installed locally

    Use only when direct PDF extraction returns little or no text.
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError as exc:
        raise ImportError(
            "OCR mode requires pdf2image, pytesseract, and pillow. "
            "Install them and make sure Poppler/Tesseract are available."
        ) from exc

    pages = convert_from_path(str(pdf_path), dpi=dpi)
    ocr_chunks = []

    for page_number, image in enumerate(pages, start=1):
        text = pytesseract.image_to_string(image, lang=lang)
        ocr_chunks.append(text)
        print(f"OCR page {page_number}/{len(pages)} extracted {len(text)} characters.")

    return "\n".join(ocr_chunks).strip()


# ---------------------------------------------------------------------------
# 2. Cleaning, chunking, and language detection
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Basic text cleaning for policy/legal documents."""
    text = re.sub(r"\u00A0", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def normalize_for_match(text: str) -> str:
    """
    Normalize text for robust regex matching.

    Handles OCR artifacts such as hyphenated line breaks:
    "telecom-\\nmunications" -> "telecommunications".
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"-\s*\n\s*", "", text)
    text = text.replace("\r", "\n")
    text = re.sub(r"\s*\n\s*", " ", text)
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    """
    Split text into overlapping character-based chunks.

    Default parameters follow the original RDTII-style workflow:
    chunk size = 1000, overlap = 150.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap.")

    chunks = []
    start = 0

    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap

    return chunks


def detect_lang_label(text: str, threshold: float = 0.15) -> str:
    """
    Detect whether a chunk is English, Malay, mixed, or unknown.
    """
    try:
        probabilities = detect_langs(text[:2000])
        scores = {item.lang: item.prob for item in probabilities}
        en_score = scores.get("en", 0.0)
        ms_score = scores.get("ms", 0.0)

        if abs(en_score - ms_score) <= threshold and (en_score > 0.2 or ms_score > 0.2):
            return "mixed"

        return "en" if en_score >= ms_score else "ms"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# 3. Telecom Pillar 5 indicator rules
# ---------------------------------------------------------------------------

TELECOM_INDICATORS: Dict[str, Dict[str, object]] = {
    "5.1_passive_infra_sharing": {
        "scope": "Rules for sharing non-electronic/physical telecom infrastructure.",
        "en": [
            r"\bpassive infrastructure sharing\b",
            r"\btower(?:/| )?sharing\b",
            r"\bsite(?:/| )?sharing\b",
            r"\bduct(?:/| )?sharing\b",
            r"\bpole(?:/| )?sharing\b",
            r"\baccess to (physical )?infrastructure\b",
            r"\bmandatory sharing\b",
            r"\bnon-?discriminatory access\b",
            r"\bprice\b.*\b(sharing|access)\b",
            r"\bcharge\b.*\b(sharing|access)\b",
        ],
        "ms": [
            r"\bperkongsian (?:infrastruktur|tapak|menara|saluran|tiang)\b",
            r"\bakses ke infrastruktur (fizikal)?\b",
            r"\bperkongsi(?:an)? mandatori\b|\bwajib berkongsi\b",
            r"\bakses tanpa diskriminasi\b",
            r"\b(harga|caj)\b.*\b(perkongsian|akses)\b",
        ],
    },
    "5.2_foreign_equity_limits": {
        "scope": "Caps or limits on foreign ownership/shareholding in telecom companies.",
        "en": [
            r"\bforeign (?:equity|ownership|shareholding) (?:cap|limit|restriction)s?\b",
            r"\bno more than\s*\d{1,2}%\b|\bless than\s*50%\b|\b49%\b",
            r"\bdomestic (?:control(?:ling)?|majority)\b",
            r"\bceilings?\b.*\bforeign\b",
            r"\brestrict\b.*\bforeign (?:ownership|participation)\b",
        ],
        "ms": [
            r"\bhad (?:ekuiti|pemilikan|pegangan) asing\b",
            r"\b(?:tidak melebihi|kurang daripada)\s*\d{1,2}%\b|\b49%\b|\bkurang 50%\b",
            r"\b(kawalan|majoriti) domestik\b",
            r"\bsekatan\b.*\bpemilikan asing\b",
        ],
    },
    "5.3_functional_or_accounting_separation": {
        "scope": "Separation obligations for SMP/dominant operators.",
        "en": [
            r"\bfunctional(?:/| |-)?(?:operational|structural)? separation\b",
            r"\baccounting separation\b|\bseparate (?:books|ledgers|cost accounts)\b",
            r"\bSMP\b|\bsignificant market power\b|\bdominant operator\b",
            r"\bnon-?discrimination\b",
            r"\bcost transparency\b",
        ],
        "ms": [
            r"\bpemisahan (fungsional|operasi|struktur)\b",
            r"\bpemisahan perakaunan\b|\bakaun kos berasingan\b",
            r"\bkuasa pasaran signifikan\b|\boperator dominan\b",
            r"\btiada diskriminasi\b",
            r"\bketelusan kos\b",
        ],
    },
    "5.4_not_in_WTO_Telecom_Reference_Paper": {
        "scope": "Whether an economy has not committed to the WTO Telecom Reference Paper.",
        "en": [
            r"\bnot (?:a )?(?:party|signatory)\b.*\bWTO Telecommunications Reference Paper\b",
            r"\bnot appended\b.*\bReference Paper\b",
            r"\bnot committed\b.*\bReference Paper\b",
            r"\bReference Paper\b.*\bnot (?:applied|incorporated)\b",
        ],
        "ms": [
            r"\b(bukan pihak|tidak menandatangani)\b.*\bKertas Rujukan Telekom WTO\b",
            r"\btidak dilampirkan\b.*\bKertas Rujukan\b",
            r"\btidak komited\b.*\bKertas Rujukan\b",
        ],
    },
    "5.5_independent_telecom_authority": {
        "scope": "Independence of the telecom regulator or explicit lack of independence.",
        "en": [
            r"\bindependent (?:telecommunications )?(?:regulator|authority)\b",
            r"\bfunctional independence\b|\bno political influence\b|\bno interference\b",
            r"\binstitutional\b.*\bindependence\b|\bfinancial\b.*\bindependence\b",
            r"\bseparate legal status\b|\bbudget autonomy\b",
            r"\bregulator not independent\b|\black of independence\b",
        ],
        "ms": [
            r"\bpihak berkuasa(?:/| )?telekom (?:yang )?bebas\b|\bpengawal selia bebas\b",
            r"\bkebebasan fungsional\b|\btiada pengaruh politik\b|\btiada campur tangan\b",
            r"\bkebebasan institusi(?:si)?(?:onal)?\b|\bkebebasan kewangan\b",
            r"\bstatus undang-undang berasingan\b|\bautonomi belanjawan\b",
            r"\bregulator tidak bebas\b|\bkekurangan kebebasan\b",
        ],
    },
}

WEAK_PATTERNS: Dict[str, Dict[str, List[str]]] = {
    "5.1_passive_infra_sharing": {
        "en": [r"\bprice principles?\b", r"\bnon-?discriminatory access\b"],
        "ms": [r"\bakses tanpa diskriminasi\b"],
    },
    "5.2_foreign_equity_limits": {
        "en": [r"\bforeign shareholding\b"],
        "ms": [],
    },
    "5.3_functional_or_accounting_separation": {
        "en": [r"\btransparency\b"],
        "ms": [],
    },
    "5.4_not_in_WTO_Telecom_Reference_Paper": {
        "en": [r"\bReference Paper\b"],
        "ms": [r"\bKertas Rujukan\b"],
    },
    "5.5_independent_telecom_authority": {
        "en": [r"\bindependence\b"],
        "ms": [r"\bkebebasan\b"],
    },
}

SEP = r"[ \t\-\/]*"


def loose_phrase(*tokens: str) -> str:
    """Create a regex phrase that tolerates spaces, hyphens, and slashes."""
    return rf"\b{SEP.join(map(re.escape, tokens))}\b"


EXTRA_LOOSE: Dict[str, Dict[str, List[str]]] = {
    "5.1_passive_infra_sharing": {
        "en": [
            loose_phrase("passive", "infrastructure", "sharing"),
            loose_phrase("tower", "sharing"),
            loose_phrase("site", "sharing"),
            loose_phrase("duct", "sharing"),
            loose_phrase("pole", "sharing"),
            loose_phrase("non", "discriminatory", "access"),
            loose_phrase("mandatory", "sharing"),
            loose_phrase("access", "to", "physical", "infrastructure"),
        ],
        "ms": [
            loose_phrase("perkongsian", "infrastruktur"),
            loose_phrase("akses", "tanpa", "diskriminasi"),
            loose_phrase("perkongsi", "mandatori"),
        ],
    },
    "5.2_foreign_equity_limits": {
        "en": [
            loose_phrase("foreign", "equity"),
            loose_phrase("foreign", "ownership"),
            loose_phrase("foreign", "shareholding"),
            loose_phrase("domestic", "majority"),
            r"\bno more than[ \t\-\/]*\d{1,2}%\b",
            r"\bless than[ \t\-\/]*50%\b|\b49%\b",
        ],
        "ms": [
            loose_phrase("had", "ekuiti", "asing"),
            loose_phrase("pegangan", "asing"),
            loose_phrase("kawalan", "domestik"),
            r"\bkurang[ \t\-\/]*\d{1,2}%\b|\bkurang[ \t\-\/]*50%\b|\b49%\b",
        ],
    },
    "5.3_functional_or_accounting_separation": {
        "en": [
            loose_phrase("functional", "separation"),
            loose_phrase("operational", "separation"),
            loose_phrase("structural", "separation"),
            loose_phrase("accounting", "separation"),
            loose_phrase("significant", "market", "power"),
            loose_phrase("dominant", "operator"),
            loose_phrase("non", "discrimination"),
            loose_phrase("cost", "transparency"),
        ],
        "ms": [
            loose_phrase("pemisahan", "fungsional"),
            loose_phrase("pemisahan", "perakaunan"),
            loose_phrase("kuasa", "pasaran", "signifikan"),
            loose_phrase("operator", "dominan"),
        ],
    },
    "5.4_not_in_WTO_Telecom_Reference_Paper": {
        "en": [
            loose_phrase("wto", "telecommunications", "reference", "paper"),
            r"\bnot[ \t\-\/]*(?:a )?(?:party|signatory)\b.*\breference paper\b",
            r"\bnot[ \t\-\/]*appended\b.*\breference paper\b",
            r"\bnot[ \t\-\/]*committed\b.*\breference paper\b",
        ],
        "ms": [
            loose_phrase("kertas", "rujukan", "telekom", "wto"),
            r"\b(bukan pihak|tidak menandatangani)\b.*\bkertas rujukan\b",
            r"\btidak[ \t\-\/]*dilampirkan\b.*\bkertas rujukan\b",
            r"\btidak[ \t\-\/]*komited\b.*\bkertas rujukan\b",
        ],
    },
    "5.5_independent_telecom_authority": {
        "en": [
            loose_phrase("independent", "telecommunications", "regulator"),
            loose_phrase("independent", "telecom", "authority"),
            loose_phrase("functional", "independence"),
            loose_phrase("institutional", "independence"),
            loose_phrase("financial", "independence"),
            loose_phrase("separate", "legal", "status"),
            loose_phrase("budget", "autonomy"),
            loose_phrase("no", "political", "influence"),
            loose_phrase("regulator", "not", "independent"),
        ],
        "ms": [
            loose_phrase("pengawal", "selia", "bebas"),
            loose_phrase("pihak", "berkuasa", "telekom", "bebas"),
            loose_phrase("kebebasan", "fungsional"),
            loose_phrase("kebebasan", "institusi"),
            loose_phrase("kebebasan", "kewangan"),
            loose_phrase("status", "undang-undang", "berasingan"),
            loose_phrase("autonomi", "belanjawan"),
            loose_phrase("tiada", "pengaruh", "politik"),
            loose_phrase("regulator", "tidak", "bebas"),
        ],
    },
}


# ---------------------------------------------------------------------------
# 4. Matching and scoring
# ---------------------------------------------------------------------------

def compile_patterns() -> Dict[str, Dict[str, object]]:
    """Compile strong, weak, and loose regex patterns."""
    compiled: Dict[str, Dict[str, object]] = {}

    for indicator, config in TELECOM_INDICATORS.items():
        compiled[indicator] = {
            "scope": config["scope"],
            "en": {"strong": [], "weak": []},
            "ms": {"strong": [], "weak": []},
        }

        for language in ("en", "ms"):
            strong_patterns = list(config.get(language, []))
            strong_patterns.extend(EXTRA_LOOSE.get(indicator, {}).get(language, []))
            weak_patterns = WEAK_PATTERNS.get(indicator, {}).get(language, [])

            compiled[indicator][language]["strong"] = [
                re.compile(pattern, re.IGNORECASE) for pattern in strong_patterns
            ]
            compiled[indicator][language]["weak"] = [
                re.compile(pattern, re.IGNORECASE) for pattern in weak_patterns
            ]

    return compiled


COMPILED_PATTERNS = compile_patterns()


def score_chunk(
    text: str,
    lang_chunk: str,
    scoring_mode: str = "weighted",
    window: int = 140,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    """
    Score one text chunk and return per-indicator scores plus evidence hits.

    scoring_mode:
    - weighted: strong hit = 1.0, weak hit = 0.5
    - any_one_is_1: any hit gives the indicator a score of 1.0
    """
    normalized_text = normalize_for_match(text)
    per_indicator = {indicator: 0.0 for indicator in TELECOM_INDICATORS}
    evidence_hits: List[Dict[str, object]] = []

    for indicator, config in COMPILED_PATTERNS.items():
        for language in ("en", "ms"):
            for strength, weight in (("strong", 1.0), ("weak", 0.5)):
                for regex in config[language][strength]:
                    for match in regex.finditer(normalized_text):
                        start = max(0, match.start() - window)
                        end = min(len(normalized_text), match.end() + window)

                        hit = {
                            "indicator": indicator,
                            "lang_chunk": lang_chunk,
                            "lang_hit": language,
                            "strength": strength,
                            "weight": weight,
                            "span": [match.start(), match.end()],
                            "evidence": normalized_text[start:end],
                        }
                        evidence_hits.append(hit)

                        if scoring_mode == "any_one_is_1":
                            per_indicator[indicator] = 1.0
                        else:
                            per_indicator[indicator] += weight

    return per_indicator, evidence_hits


def score_document(
    chunks: Iterable[str],
    lang_labels: Iterable[str],
    scoring_mode: str = "weighted",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, object]]]:
    """
    Score all chunks and return:
    - chunk-level score DataFrame
    - document-level score DataFrame
    - evidence hit records
    """
    chunk_rows = []
    all_hits: List[Dict[str, object]] = []
    doc_raw_sum = {indicator: 0.0 for indicator in TELECOM_INDICATORS}
    doc_any_hit = {indicator: False for indicator in TELECOM_INDICATORS}

    for chunk_id, (chunk, lang_label) in enumerate(zip(chunks, lang_labels)):
        per_indicator, hits = score_chunk(chunk, lang_label, scoring_mode=scoring_mode)

        row = {"chunk_id": chunk_id, "lang_chunk": lang_label}
        row.update(per_indicator)
        chunk_rows.append(row)

        for indicator, score in per_indicator.items():
            doc_raw_sum[indicator] += score

        for hit in hits:
            hit = {"chunk_id": chunk_id, **hit}
            all_hits.append(hit)
            doc_any_hit[hit["indicator"]] = True

    chunk_scores = pd.DataFrame(chunk_rows).fillna(0.0)

    doc_rows = []
    for indicator, config in TELECOM_INDICATORS.items():
        if scoring_mode == "any_one_is_1":
            raw_score = 1.0 if doc_any_hit[indicator] else 0.0
        else:
            raw_score = doc_raw_sum[indicator]

        doc_rows.append(
            {
                "indicator": indicator,
                "scope": config["scope"],
                "any_hit": doc_any_hit[indicator],
                "raw_sum": round(raw_score, 3),
                "doc_score_capped": round(min(1.0, raw_score), 3),
            }
        )

    doc_scores = pd.DataFrame(doc_rows).sort_values(
        ["doc_score_capped", "raw_sum", "indicator"],
        ascending=[False, False, True],
    )

    return chunk_scores, doc_scores, all_hits


# ---------------------------------------------------------------------------
# 5. Export helpers and CLI
# ---------------------------------------------------------------------------

def write_jsonl(records: Iterable[Dict[str, object]], output_path: str | Path) -> None:
    """Write records to a JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_evidence_bundle(
    doc_scores: pd.DataFrame,
    all_hits: List[Dict[str, object]],
    evidence_per_indicator: int = 5,
) -> List[Dict[str, object]]:
    """Attach compact evidence samples to document-level scores."""
    records = []

    for row in doc_scores.to_dict(orient="records"):
        indicator = row["indicator"]
        evidence = [
            {key: value for key, value in hit.items() if key != "indicator"}
            for hit in all_hits
            if hit["indicator"] == indicator
        ][:evidence_per_indicator]

        row["evidence_samples"] = evidence
        records.append(row)

    return records


def run_pipeline(
    pdf_path: str | Path,
    output_dir: str | Path = "results",
    use_ocr: bool = False,
    chunk_size: int = 1000,
    overlap: int = 150,
    scoring_mode: str = "weighted",
) -> None:
    """Run the full extraction, chunking, classification, and export pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_ocr:
        raw_text = extract_text_ocr(pdf_path)
    else:
        raw_text = extract_text_direct(pdf_path)

    if not raw_text:
        raise ValueError(
            "No text was extracted from the PDF. Try running again with --ocr."
        )

    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(cleaned_text, chunk_size=chunk_size, overlap=overlap)
    lang_labels = [detect_lang_label(chunk) for chunk in chunks]

    chunk_scores, doc_scores, all_hits = score_document(
        chunks,
        lang_labels,
        scoring_mode=scoring_mode,
    )

    chunk_records = []
    for chunk_id, (chunk, lang_label) in enumerate(zip(chunks, lang_labels)):
        score_row = chunk_scores.loc[chunk_scores["chunk_id"] == chunk_id].iloc[0].to_dict()
        chunk_records.append(
            {
                "chunk_id": chunk_id,
                "lang_chunk": lang_label,
                "text": chunk,
                "scores": {
                    key: value
                    for key, value in score_row.items()
                    if key not in ("chunk_id", "lang_chunk")
                },
            }
        )

    write_jsonl(chunk_records, output_dir / "pdf_chunks_scored.jsonl")
    write_jsonl(
        build_evidence_bundle(doc_scores, all_hits),
        output_dir / "scores.jsonl",
    )

    chunk_scores.to_csv(output_dir / "chunk_scores.csv", index=False)
    doc_scores.to_csv(output_dir / "document_scores.csv", index=False)

    print(f"Processed {len(chunks)} chunks.")
    print(f"Results saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Classify telecom policy indicators from PDF text."
    )
    parser.add_argument("--pdf", required=True, help="Path to the input PDF file.")
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Use OCR extraction for scanned PDFs.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Character length of each chunk.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=150,
        help="Character overlap between chunks.",
    )
    parser.add_argument(
        "--scoring-mode",
        choices=["weighted", "any_one_is_1"],
        default="weighted",
        help="Scoring mode for indicator classification.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        pdf_path=args.pdf,
        output_dir=args.output_dir,
        use_ocr=args.ocr,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        scoring_mode=args.scoring_mode,
    )
