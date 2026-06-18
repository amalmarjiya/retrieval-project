"""Hybrid retrieval for Section B.

General retrieval signals only:
- MiniLM semantic chunk similarity
- sparse token overlap
- number/date overlap
- title overlap
- first chunk prior
- agreement bonus between semantic and lexical/number/title signals
- optional BM25 page boost if bm25_index.json exists
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from embed import embed_queries
from utils import ARTIFACTS_DIR, K_EVAL

TOKEN_RE = re.compile(r"[a-z0-9]+")
YEAR_RE = re.compile(r"\b\d{3,4}s?\b|\b\d+(?:\.\d+)?\b")

LEXICAL_WEIGHT = 0.08
NUMBER_WEIGHT = 0.18
BM25_WEIGHT = 0.03
TITLE_WEIGHT = 0.04
FIRST_CHUNK_BONUS = 0.30
AGREEMENT_WEIGHT = 0.04
SEMANTIC_MARGIN = 0.08

RANK_EXPAND = 5
SIBLING_SCAN = 40

_CORPUS_VECTORS = None
_PAGE_IDS = None
_TEXTS = None
_CHUNK_IDS = None
_BM25 = None
_PAGE_SIBLINGS = None


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def _expand_query_tokens(query: str) -> List[str]:
    tokens = tokenize(query)
    out = []
    seen = set()

    for tok in tokens:
        variants = [tok]
        if tok.endswith("s") and len(tok) > 4:
            variants.append(tok[:-1])
        elif len(tok) > 3:
            variants.append(tok + "s")

        for variant in variants:
            if variant and variant not in seen:
                seen.add(variant)
                out.append(variant)

    return out


def extract_numbers(text: str, *, expand_decades: bool = True) -> List[str]:
    nums = []
    for raw in YEAR_RE.findall(text.lower()):
        clean = raw.strip()
        nums.append(clean)
        if expand_decades and clean.endswith("s") and len(clean) == 5 and clean[:4].isdigit():
            nums.append(clean[:4])
    return nums


def _normalise_rows(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat.astype(np.float32)

    row_min = mat.min(axis=1, keepdims=True)
    row_max = mat.max(axis=1, keepdims=True)
    scale = np.where(row_max > row_min, row_max - row_min, 1.0)
    return ((mat - row_min) / scale).astype(np.float32)


def _load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _find_page_siblings(root: Path) -> Dict[int, List[int]]:
    for path in root.glob("*.json"):
        data = _load_json_if_exists(path)
        siblings = data.get("page_to_siblings") if isinstance(data, dict) else None
        if siblings:
            return {int(k): [int(x) for x in v] for k, v in siblings.items()}
    return {}


def _load_resources(artifacts_dir=None):
    global _CORPUS_VECTORS, _PAGE_IDS, _TEXTS, _CHUNK_IDS, _BM25, _PAGE_SIBLINGS

    root = Path(artifacts_dir or ARTIFACTS_DIR)

    if artifacts_dir is None and _CORPUS_VECTORS is not None:
        return _CORPUS_VECTORS, _PAGE_IDS, _TEXTS, _CHUNK_IDS, _BM25, _PAGE_SIBLINGS

    vectors = np.load(root / "index_vectors.npy").astype(np.float32)
    meta = json.loads((root / "index_meta.json").read_text(encoding="utf-8"))

    page_ids = np.array(meta["page_ids"], dtype=np.int32)
    texts = [str(t) for t in meta.get("texts", [""] * len(page_ids))]
    chunk_ids = np.array(meta.get("chunk_ids", [0] * len(page_ids)), dtype=np.int32)

    bm25 = _load_json_if_exists(root / "bm25_index.json")
    page_siblings = _find_page_siblings(root)

    if artifacts_dir is None:
        _CORPUS_VECTORS = vectors
        _PAGE_IDS = page_ids
        _TEXTS = texts
        _CHUNK_IDS = chunk_ids
        _BM25 = bm25
        _PAGE_SIBLINGS = page_siblings

    return vectors, page_ids, texts, chunk_ids, bm25, page_siblings


def _sparse_overlap(query_token_lists: List[List[str]], chunk_token_lists: List[List[str]]) -> np.ndarray:
    n_q = len(query_token_lists)
    n_c = len(chunk_token_lists)
    mat = np.zeros((n_q, n_c), dtype=np.float32)

    inverted = defaultdict(list)
    for ci, toks in enumerate(chunk_token_lists):
        for tok in set(toks):
            inverted[tok].append(ci)

    for qi, q_tokens in enumerate(query_token_lists):
        q_set = set(q_tokens)
        if not q_set:
            continue

        denom = float(max(len(q_set), 1))
        row = mat[qi]
        for tok in q_set:
            for ci in inverted.get(tok, []):
                row[ci] += 1.0 / denom

    return np.clip(mat, 0.0, 1.0)


def _number_overlap(queries: Sequence[str], texts: Sequence[str]) -> np.ndarray:
    q_nums = [set(extract_numbers(q, expand_decades=True)) for q in queries]
    c_nums = [set(extract_numbers(t, expand_decades=False)) for t in texts]

    mat = np.zeros((len(queries), len(texts)), dtype=np.float32)

    for qi, nums in enumerate(q_nums):
        if not nums:
            continue

        denom = float(max(len(nums), 1))
        row = mat[qi]
        for ci, chunk_nums in enumerate(c_nums):
            if chunk_nums:
                row[ci] = len(nums & chunk_nums) / denom

    return mat


def _title_part(text: str) -> str:
    return text.split(":", 1)[0].split("[", 1)[0].strip().lower()


def _title_overlap(query_token_lists: List[List[str]], texts: Sequence[str]) -> np.ndarray:
    title_tokens = [set(tokenize(_title_part(t))) for t in texts]
    mat = np.zeros((len(query_token_lists), len(texts)), dtype=np.float32)

    for qi, q_tokens in enumerate(query_token_lists):
        q_set = set(q_tokens)
        if not q_set:
            continue

        for ci, t_set in enumerate(title_tokens):
            if t_set:
                mat[qi, ci] = len(q_set & t_set) / max(len(t_set), 1)

    return np.clip(mat, 0.0, 1.0)


def _bm25_scores(queries: Sequence[str], bm25: dict, unique_pids: np.ndarray) -> np.ndarray:
    if not bm25 or BM25_WEIGHT <= 0.0:
        return np.zeros((len(queries), len(unique_pids)), dtype=np.float32)

    bm25_page_ids = bm25.get("page_ids", [])
    page_tf = bm25.get("page_tf", [])
    page_lens = bm25.get("page_lens")
    idf = bm25.get("idf", {})
    avg_dl = float(bm25.get("avg_dl", 1.0) or 1.0)
    k1 = float(bm25.get("k1", 1.5))
    b = float(bm25.get("b", 0.75))

    if not bm25_page_ids or not page_tf:
        return np.zeros((len(queries), len(unique_pids)), dtype=np.float32)

    page_pos = {int(pid): i for i, pid in enumerate(unique_pids)}
    out = np.zeros((len(queries), len(unique_pids)), dtype=np.float32)

    for qi, query in enumerate(queries):
        terms = set(_expand_query_tokens(query))
        if not terms:
            continue

        for term in terms:
            term_idf = idf.get(term)
            if term_idf is None:
                continue
            term_idf = float(term_idf)

            for bi, tf_dict in enumerate(page_tf):
                tf = float(tf_dict.get(term, 0.0))
                if tf <= 0.0:
                    continue

                dl = float(page_lens[bi] if page_lens is not None else sum(tf_dict.values()))
                denom = tf + k1 * (1.0 - b + b * dl / avg_dl)
                score = term_idf * tf * (k1 + 1.0) / max(denom, 1e-6)

                pos = page_pos.get(int(bm25_page_ids[bi]))
                if pos is not None:
                    out[qi, pos] += score

    return _normalise_rows(out)


def search_batch(queries: Sequence[str], *, top_k=K_EVAL, artifacts_dir=None) -> List[List[int]]:
    if not queries:
        return []

    vectors, page_ids, texts, chunk_ids, bm25, page_siblings = _load_resources(artifacts_dir)

    query_vectors = embed_queries(list(queries)).astype(np.float32)
    if query_vectors.size == 0 or vectors.size == 0:
        return [[] for _ in queries]

    semantic_scores = query_vectors @ vectors.T

    query_token_lists = [_expand_query_tokens(q) for q in queries]
    chunk_token_lists = [tokenize(t) for t in texts]

    lex_mat = _sparse_overlap(query_token_lists, chunk_token_lists)
    num_mat = _number_overlap(queries, texts)
    title_mat = _title_overlap(query_token_lists, texts)

    first_chunk_bonus = np.where(chunk_ids == 0, FIRST_CHUNK_BONUS, 0.0).astype(np.float32)

    semantic_cutoff = np.max(semantic_scores, axis=1, keepdims=True) - SEMANTIC_MARGIN
    agreement_bonus = (
        (semantic_scores >= semantic_cutoff)
        & ((lex_mat > 0.0) | (num_mat > 0.0) | (title_mat > 0.0))
    ).astype(np.float32)

    chunk_scores = (
        semantic_scores
        + LEXICAL_WEIGHT * lex_mat
        + NUMBER_WEIGHT * num_mat
        + TITLE_WEIGHT * title_mat
        + first_chunk_bonus[None, :]
        + AGREEMENT_WEIGHT * agreement_bonus
    ).astype(np.float32)

    unique_pids, inverse = np.unique(page_ids, return_inverse=True)
    n_queries = chunk_scores.shape[0]
    n_pages = len(unique_pids)

    page_scores = np.full((n_queries, n_pages), -np.inf, dtype=np.float32)
    np.maximum.at(
        page_scores,
        (np.arange(n_queries)[:, None], inverse[None, :]),
        chunk_scores,
    )

    page_scores += BM25_WEIGHT * _bm25_scores(queries, bm25, unique_pids)

    ranked = []
    take = min(n_pages, max(top_k * RANK_EXPAND, top_k))

    for row in page_scores:
        if take >= n_pages:
            order = np.argsort(-row)
        else:
            part = np.argpartition(-row, take - 1)[:take]
            order = part[np.argsort(-row[part])]

        candidates = [int(unique_pids[i]) for i in order]

        expanded = list(candidates)
        seen = set(expanded)

        for pid in candidates[:SIBLING_SCAN]:
            for sib in page_siblings.get(pid, []):
                sib = int(sib)
                if sib not in seen:
                    seen.add(sib)
                    expanded.append(sib)

        ranked.append(expanded[:top_k])

    return ranked
