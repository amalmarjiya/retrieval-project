"""Offline index build and load."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from chunk import chunk_corpus, CHUNK_WORDS, CHUNK_OVERLAP
from embed import embed_texts
from lexical import build_bm25
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, iter_entries

INDEX_VECTORS_NAME = "index_vectors.npy"
INDEX_META_NAME    = "index_meta.json"


def _l2_normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vectors / norms).astype(np.float32)


def build_index(*, entries_dir=None, artifacts_dir=None):
    """Embed corpus chunks and save index artifacts."""
    out_dir = artifacts_dir or ensure_artifacts_dir()
    records = list(iter_entries(entries_dir))
    chunks  = chunk_corpus(records)
    texts   = [c.text for c in chunks]

    vectors  = embed_texts(texts)
    vectors  = _l2_normalise(vectors)
    page_ids = [c.page_id for c in chunks]

    np.save(out_dir / INDEX_VECTORS_NAME, vectors)

    meta = {
        "page_ids":     page_ids,
        "chunk_ids":    [c.chunk_id for c in chunks],
        "texts":        texts,
        "model":        "sentence-transformers/all-MiniLM-L6-v2",
        "chunk_words":  CHUNK_WORDS,
        "chunk_overlap": CHUNK_OVERLAP,
        "num_vectors":  len(page_ids),
        "normalised":   True,
    }
    (out_dir / INDEX_META_NAME).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    build_bm25(chunks, out_dir)

    print(
        f"[index] built {len(page_ids)} chunks from {len(records)} pages "
        f"-- vectors shape {vectors.shape}"
    )
    return vectors, page_ids


def load_index(artifacts_dir=None):
    """Load prebuilt index from artifacts/."""
    root = artifacts_dir or ARTIFACTS_DIR

    vectors_path = root / INDEX_VECTORS_NAME
    meta_path    = root / INDEX_META_NAME

    if not vectors_path.exists():
        raise FileNotFoundError(
            f"Index vectors not found at {vectors_path}. "
            "Run scripts/build_index.py first."
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Index metadata not found at {meta_path}. "
            "Run scripts/build_index.py first."
        )

    vectors  = np.load(vectors_path)
    meta     = json.loads(meta_path.read_text(encoding="utf-8"))
    page_ids = np.array(meta["page_ids"], dtype=np.int32)

    if vectors.ndim != 2:
        raise ValueError(f"Expected 2-D vectors, got shape {vectors.shape}.")
    if len(page_ids) != vectors.shape[0]:
        raise ValueError(
            f"page_ids length ({len(page_ids)}) != vectors rows ({vectors.shape[0]})."
        )
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)

    return vectors, page_ids
