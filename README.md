# Policy Document Classification for Digital Trade Regulations

## Project Overview

This project was developed to automate the classification of digital trade policy and regulatory documents.

The system processes policy documents from multiple countries and predicts whether specific regulatory indicators are present based on the RDTII (Regional Digital Trade Integration Index) framework.


## Problem Statement

Manual review of regulatory documents is time-consuming and difficult to scale.

This project aims to:

- Extract text from PDF policy documents
- Clean and preprocess unstructured text
- Identify policy indicators automatically
- Reduce manual review effort


## Workflow

PDF Documents

↓

OCR Extraction

↓

Text Cleaning

↓

Feature Engineering

↓

TF-IDF Representation

↓

Machine Learning Classification

↓

Indicator Prediction


## Methods

### Data Processing

- PDF parsing
- OCR extraction
- Text normalization
- Metadata extraction

### Machine Learning

- TF-IDF
- Logistic Regression
- One-vs-Rest Classification
- Embedding-based Retrieval

### Evaluation

- Accuracy
- Precision
- Recall
- Manual validation

## Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- PyMuPDF
- Tesseract OCR


## Example Use Case

Input:

Telecommunications Regulation Document

Output:

Indicator 5.3 = Yes

Confidence Score = 0.89


## Impact

This approach reduced manual policy review effort and improved the efficiency of regulatory document analysis.


## Author

Huhu Liu

M.A. Economics, Duke University

Applied AI | NLP | Data Science
