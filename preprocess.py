"""
preprocess.py — News Headline Classifier
CIS 4190/5190 Final Project
 
Backend contract:
    prepare_data(csv_path: str) -> (X, y)
 
Backend CSV has columns: `url`, `headline`.
We read headlines directly and infer labels from URL domain
(1 = Fox, 0 = NBC).
"""
from __future__ import annotations
from urllib.parse import urlparse
 
import pandas as pd
 
 
def _label_from_url(url: str) -> int:
    """1 = Fox, 0 = NBC."""
    domain = urlparse(str(url)).netloc.lower()
    if "foxnews.com" in domain or "foxbusiness.com" in domain:
        return 1
    return 0
 
 
def prepare_data(csv_path: str):
    df = pd.read_csv(csv_path)
 
    # Read headlines directly
    X = df["headline"].fillna("").astype(str).str.strip().tolist()
 
    # Infer labels from URL domain
    y = [_label_from_url(u) for u in df["url"].astype(str).tolist()]
 
    return X, y
 