"""BM25 lexical index build and search."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple

from utils import ARTIFACTS_DIR

BM25_INDEX_NAME = "bm25_index.json"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def build_bm25(chunks, out_dir, k1=1.5, b=0.75):
    """Build page-level BM25 index from chunks."""
    print("[bm25] building index...")

    page_texts = {}
    for chunk in chunks:
        pid = chunk.page_id
        if pid not in page_texts:
            page_texts[pid] = []
        page_texts[pid].append(chunk.text)

    page_ids  = list(page_texts.keys())
    page_docs = [" ".join(page_texts[pid]) for pid in page_ids]
    tokenized = [tokenize(doc) for doc in page_docs]

    avg_dl = sum(len(t) for t in tokenized) / max(len(tokenized), 1)

    df = Counter()
    for toks in tokenized:
        for t in set(toks):
            df[t] += 1

    N = len(page_ids)
    idf = {term: math.log((N - freq + 0.5) / (freq + 0.5) + 1)
           for term, freq in df.items()}

    page_tf = [dict(Counter(toks)) for toks in tokenized]

    bm25_data = {
        "page_ids": page_ids,
        "page_tf":  page_tf,
        "idf":      idf,
        "avg_dl":   avg_dl,
        "k1":       k1,
        "b":        b,
    }

    (out_dir / BM25_INDEX_NAME).write_text(
        json.dumps(bm25_data), encoding="utf-8"
    )
    print(f"[bm25] built index for {N} pages, {len(idf)} unique terms")
    return bm25_data


def load_bm25(artifacts_dir=None):
    """Load BM25 index from disk."""
    root = artifacts_dir or ARTIFACTS_DIR
    bm25_path = root / BM25_INDEX_NAME
    if not bm25_path.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {bm25_path}. "
            "Run scripts/build_index.py first."
        )
    return json.loads(bm25_path.read_text(encoding="utf-8"))


def bm25_search(query: str, bm25_data: dict, n: int = 100):
    """Return top-n (page_id, score) pairs for a query."""
    import numpy as np
    page_ids = bm25_data["page_ids"]
    page_tf  = bm25_data["page_tf"]
    idf      = bm25_data["idf"]
    avg_dl   = bm25_data["avg_dl"]
    k1       = bm25_data["k1"]
    b        = bm25_data["b"]

    q_tokens = set(tokenize(query))
    scores   = np.zeros(len(page_ids), dtype=np.float32)

    for term in q_tokens:
        if term not in idf:
            continue
        term_idf = idf[term]
        for pi, tf_dict in enumerate(page_tf):
            tf = tf_dict.get(term, 0)
            if tf == 0:
                continue
            dl  = sum(tf_dict.values())
            num = tf * (k1 + 1)
            den = tf + k1 * (1 - b + b * dl / avg_dl)
            scores[pi] += term_idf * num / den

    top_idx = np.argsort(-scores)[:n]
    return [(int(page_ids[i]), float(scores[i])) for i in top_idx]
