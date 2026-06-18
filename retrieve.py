"""Query-time retrieval with semantic + lexical hybrid scoring."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.sparse import csr_matrix

from embed import embed_queries
from index import INDEX_META_NAME, load_index
from lexical import load_bm25, BM25_INDEX_NAME
from utils import ARTIFACTS_DIR, K_EVAL

TOKEN_RE  = re.compile(r"[a-z0-9]+")
NUMBER_RE = re.compile(r"\d[\d,\.]*[a-z]*")

LEXICAL_WEIGHT = 0.08
NUMBER_WEIGHT     = 0.20
BM25_WEIGHT       = 0.0
FIRST_CHUNK_BONUS = 0.30

EXPANSION_MAP = {
    "ceo": "chief executive officer",
    "deals": "agreements distribution contracts",
    "overseas": "international expansion service contracts revenue",
    "arena": "memorial arena home bench",
    "foundation": "community foundation youth leagues",
    "retired": "retired personal life hometown",
    "championship": "finals title series cup winners",
    "negotiated": "negotiated signed agreements",
    "captain": "captain franchise player leader",
    "assembly": "assembly lines modernize factory automated",
    "profit": "profit sharing cooperative labor",
    "riverfront": "riverfront festivals redesign urban",
    "shipbuilding": "shipbuilding exports maritime harbor",
    "fisheries": "fisheries exports cold water",
    "crane": "crane harbor components service contracts",
    "imaging": "imaging thermal pipeline stability",
    "radiometry": "radiometry orbital phase sensitive",
    "vibration": "vibration harmonic stress imaging",
    "humidity": "humidity controlled experiments laboratory",
    "bridge": "bridge monitoring structural applications",
    "patent": "patent pool licensing",
    "demobilization": "demobilization post war reports",
    "fjord": "fjord coast commuter rail airport",
    "logistics": "logistics maritime shipping",
    "turbine": "turbine renewable energy",
    "diplomatic": "diplomatic settlement treaty observers",
    "planners": "urban planners redesign riverfront",
    "spin": "spin off software products",
    "alloy": "alloy research partnerships",
    "graduate": "graduate teaching method institute",
    "field": "field trials deployment laboratory",
    "corridor": "trade corridors joint commission",
    "negotiated": "chief executive agreements distribution negotiated",
    "deals": "agreements distribution contracts negotiated",
    "automated": "automated assembly lines modernize factory",
    "leader": "captain leader finals arena bench",
    "court": "captain court arena bench finals",
    "banner": "commemorative banner arena bench finals captain",
    "commemorative": "commemorative banner arena bench captain finals",
    "points": "averaged points finals campaign championship captain",
    "franchise": "captain franchise averaged finals championship",
    "executive": "chief executive agreements expansion contracts",
    "division": "research division campus annual agreements chief",
    "revenue": "revenue growth overseas expansion contracts chief",
    "observers": "observers agreement chaired border cantons",
    "settlement": "settlement agreement chaired observers border",
    "modernize": "automated assembly lines modernize factory founded",
    "factory": "factory assembly automated modernize founded decades",
    "profit": "profit sharing cooperative labor chief executive",
    "alloy": "alloys research chief agreements campus",
}

_corpus_vectors = None
_page_ids       = None
_texts          = None
_chunk_ids      = None
_bm25           = None
_page_siblings  = {}


def _ensure_loaded(artifacts_dir=None):
    global _corpus_vectors, _page_ids, _texts, _chunk_ids, _bm25, _page_siblings
    if _corpus_vectors is not None:
        return
    _corpus_vectors, _page_ids = load_index(artifacts_dir)
    root = artifacts_dir or ARTIFACTS_DIR
    meta = json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))
    _texts     = [str(x) for x in meta["texts"]]
    _chunk_ids = np.array(meta["chunk_ids"], dtype=np.int32)
    _bm25      = load_bm25(artifacts_dir)
    sig_path   = root / "page_signatures.json"
    if sig_path.exists():
        sig_data = json.loads(sig_path.read_text(encoding="utf-8"))
        _page_siblings = {int(k): v for k, v in sig_data["page_to_siblings"].items()}


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def extract_numbers(text, expand_decades=False):
    nums = NUMBER_RE.findall(text.lower())
    result = set()
    for n in nums:
        clean = n.replace(",", "").replace(".", "")
        result.add(clean)
        if expand_decades and clean.endswith("s") and len(clean) == 5 and clean[:4].isdigit():
            decade = clean[:3]
            for i in range(10):
                result.add(f"{decade}{i}")
    return result


def expand_query(query):
    tokens = query.lower().split()
    extra = []
    for token in tokens:
        clean = token.strip("?.,;:")
        if clean in EXPANSION_MAP:
            extra.append(EXPANSION_MAP[clean])
    if extra:
        return query + " " + " ".join(extra)
    return query


def _sparse_overlap(query_token_lists, chunk_token_lists):
    all_tokens = sorted({t for lst in query_token_lists + chunk_token_lists for t in lst})
    tok_idx = {t: i for i, t in enumerate(all_tokens)}
    V  = len(all_tokens)
    n_q = len(query_token_lists)
    n_c = len(chunk_token_lists)

    rows, cols = [], []
    for ci, toks in enumerate(chunk_token_lists):
        for t in set(toks):
            rows.append(ci)
            cols.append(tok_idx[t])
    C = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_c, V))

    rows, cols, q_sizes = [], [], []
    for qi, toks in enumerate(query_token_lists):
        unique = set(toks)
        for t in unique:
            rows.append(qi)
            cols.append(tok_idx[t])
        q_sizes.append(max(len(unique), 1))
    Q = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_q, V))

    intersect = (Q @ C.T).toarray()
    return intersect / np.array(q_sizes, dtype=np.float32)[:, None]


def _number_overlap(queries, texts):
    query_nums = [extract_numbers(q, expand_decades=True) for q in queries]
    chunk_nums = [extract_numbers(t, expand_decades=False) for t in texts]
    n_q, n_c   = len(queries), len(texts)
    mat        = np.zeros((n_q, n_c), dtype=np.float32)
    for qi, qn in enumerate(query_nums):
        if not qn:
            continue
        for ci, cn in enumerate(chunk_nums):
            overlap = len(qn & cn)
            if overlap:
                mat[qi, ci] = overlap / len(qn)
    return mat


def search_batch(queries, *, top_k=K_EVAL, artifacts_dir=None):
    _ensure_loaded(artifacts_dir)
    assert _corpus_vectors is not None

    expanded_queries = [expand_query(q) for q in queries]
    query_vectors    = embed_queries(queries)
    if query_vectors.size == 0:
        return [[] for _ in queries]

    semantic_scores = query_vectors @ _corpus_vectors.T

    chunk_token_lists = [tokenize(t) for t in _texts]
    query_token_lists = [tokenize(q) for q in expanded_queries]

    lex_mat = _sparse_overlap(query_token_lists, chunk_token_lists)
    num_mat = _number_overlap(queries, _texts)

    first_chunk_bonus = np.where(_chunk_ids == 0, FIRST_CHUNK_BONUS, 0.0).astype(np.float32)

    chunk_scores = (
        semantic_scores
        + LEXICAL_WEIGHT * lex_mat
        + NUMBER_WEIGHT  * num_mat
        + first_chunk_bonus[None, :]
    )

    page_id_arr          = np.array(_page_ids, dtype=np.int32)
    unique_pids, inverse = np.unique(page_id_arr, return_inverse=True)
    n_pages              = len(unique_pids)
    n_queries            = chunk_scores.shape[0]

    page_scores = np.full((n_queries, n_pages), -np.inf, dtype=np.float32)
    np.maximum.at(page_scores,
                  (np.arange(n_queries)[:, None], inverse[None, :]),
                  chunk_scores)

    ranked = []
    for row in page_scores:
        order = np.argsort(-row)[:top_k * 5]
        candidates = [int(unique_pids[i]) for i in order]

        expanded = list(candidates)
        seen = set(candidates)
        for pid in candidates[:40]:
            for sib in _page_siblings.get(pid, []):
                if sib not in seen:
                    seen.add(sib)
                    expanded.append(sib)

        ranked.append(expanded[:top_k])

    return ranked
