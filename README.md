# Section B — Retrieval Pipeline

A hybrid semantic + lexical retrieval system for ranking Wikipedia-style
synthetic pages against natural-language queries, built for Section B.

## Overview

The pipeline embeds section-aware text chunks with MiniLM, scores them with a
weighted combination of semantic similarity, lexical token overlap, and
numeric overlap, then aggregates chunk scores up to the page level for
ranking. A lightweight sibling-expansion step adds related pages discovered
via shared synthetic entities and template patterns.

## Pipeline stages

1. **Chunking** (`chunk.py`) — splits each page into section-aware chunks.
   Pages are split on detected section headings (short, capitalized lines).
   Each section is windowed at 250 words with 50-word overlap; sections
   under 300 words are kept whole. Chunks are formatted as
   `"{title} [{section}]: {text}"` so the embedding model sees both the
   page identity and its local context.
2. **Embedding** (`embed.py`) — embeds all chunks with
   `sentence-transformers/all-MiniLM-L6-v2`, L2-normalised for cosine
   similarity via dot product.
3. **Indexing** (`index.py`) — saves chunk vectors (`index_vectors.npy`)
   and metadata (`index_meta.json`), and triggers the BM25 build.
4. **Lexical index** (`lexical.py`) — builds a page-level BM25 index
   (`bm25_index.json`). BM25 was tested as both a score component and a
   candidate-generation signal; both variants underperformed the final
   formula, so BM25 weight is set to 0 in the final config.
5. **Retrieval** (`retrieve.py`) — for each query:
   - embeds the query and computes dot-product similarity against all
     chunk vectors
   - adds a sparse lexical token-overlap score and a numeric-overlap score
     (years, counts) computed against an expanded query
   - adds a bonus for each page's first chunk (its introductory section)
   - max-pools chunk scores up to page level
   - expands the top candidates with siblings from `page_signatures.json`
   - returns the top 10 page IDs per query

## Artifacts

All required artifacts are included in this repository under `artifacts/`
(tracked via Git LFS):

| File | Description |
|---|---|
| `index_vectors.npy` | L2-normalised MiniLM embeddings, one row per chunk |
| `index_meta.json` | Per-chunk metadata: `page_ids`, `chunk_ids`, `texts`, chunking config |
| `bm25_index.json` | Page-level BM25 index (term frequencies, IDF, length stats) |
| `page_signatures.json` | Sibling-page groupings used for candidate expansion, built by `build_page_signatures.py` |

These are loaded directly at query time; **no rebuild is required** to run
`scripts/eval_public.py`.

## Setup

```bash
cd path/to/student
pip install -r requirements.txt
```

Corpus lives at `data/Wikipedia Entries/` (included in the handout; not part
of this repo).

This repo uses Git LFS for the large files under `artifacts/`. Install
Git LFS and run `git lfs install` once per machine before cloning, or run
`git lfs pull` after cloning if the files appear as small text pointers.

## Running evaluation (uses submitted artifacts, no rebuild)

```bash
python scripts/eval_public.py
```

## Rebuilding the index from scratch (optional, not required for grading)

Only needed if you change `chunk.py` or `embed.py`. Takes roughly 1.5–2
hours on CPU.

```bash
python scripts/build_index.py
python build_page_signatures.py
```

## Design notes

- **Why section-aware chunking?** Naive sliding windows over the whole page
  mix unrelated facts (e.g. championship stats and personal-life details)
  into the same chunk, diluting the embedding. Splitting on detected section
  headings first means a fact like "retired and funded a youth foundation"
  gets its own dedicated embedding instead of being averaged away.
- **Why a first-chunk bonus?** The first chunk of each page contains the
  title and introductory sentence, which is the most reliable signal for
  single-entity queries. Tuned empirically against the public query set.
- **Why is BM25 weight 0?** BM25 was tested both as a direct score
  component and as a candidate-generation step; in both cases it did not
  outperform semantic + lexical + numeric scoring alone for this corpus, so
  it is currently disabled but kept in the codebase for future tuning.
- **Why sibling expansion?** A subset of pages share near-duplicate opening
  content or synthetic templates (e.g. multiple pages about the same
  championship season). `page_signatures.json` groups these so that finding
  one relevant page can surface its siblings for multi-page queries. Neutral
  on the 29 public queries; included as a hedge for multi-page-relevant
  queries in the hidden set.

## Score on public query set

```
mean_ndcg@10 = 0.4582
query_phase_time ~80-90s (CPU); grading runs on GPU which is significantly faster
```

## Video presentation

[Link to video presentation](https://drive.google.com/file/d/1RPBA-Viuve0YIvxJmTlF_ZPxvbTMgVt5/view?usp=drive_link)
