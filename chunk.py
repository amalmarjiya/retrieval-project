"""Section-aware chunking - our own implementation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import re

@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str

CHUNK_WORDS     = 250
CHUNK_OVERLAP   = 50
SHORT_PAGE_WORDS = 300

def _is_heading(paragraph: str) -> bool:
    """Detect section headings: short, capitalized, ends with period."""
    p = paragraph.strip().rstrip(".")
    words = p.split()
    if not words or len(words) > 8:
        return False
    if any(c in p for c in "?!;,"):
        return False
    return words[0][0].isupper()

def split_sections(content: str) -> List[tuple]:
    """Split content into (section_name, section_text) pairs."""
    paragraphs = [p.strip() for p in content.split("\n") if p.strip()]
    sections = []
    current_section = "Introduction"
    current_parts = []

    for para in paragraphs:
        if _is_heading(para):
            if current_parts:
                sections.append((current_section, " ".join(current_parts)))
            current_section = para.rstrip(".")
            current_parts = []
        else:
            current_parts.append(para)

    if current_parts:
        sections.append((current_section, " ".join(current_parts)))

    return sections if sections else [("Introduction", content)]

def make_windows(words: List[str], size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Sliding window over word list."""
    if not words:
        return []
    windows = []
    stride = max(1, size - overlap)
    for start in range(0, len(words), stride):
        window = words[start:start + size]
        if window:
            windows.append(" ".join(window))
        if start + size >= len(words):
            break
    return windows

def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    page_id = int(record["page_id"])
    title   = str(record.get("title", "")).strip()
    content = str(record.get("content", "")).strip()

    if not content:
        return [Chunk(page_id=page_id, chunk_id=0, text=title)]

    words = content.split()

    # Short pages: single chunk
    if len(words) <= SHORT_PAGE_WORDS:
        text = f"{title}: {content}"
        return [Chunk(page_id=page_id, chunk_id=0, text=text)]

    # Long pages: section-aware chunking
    sections = split_sections(content)
    chunks = []
    seen_texts = set()
    for section, section_text in sections:
        section_words = section_text.split()
        if not section_words:
            continue
        for window in make_windows(section_words):
            # Deduplicate chunks
            key = window[:50]
            if key in seen_texts:
                continue
            seen_texts.add(key)
            text = f"{title} [{section}]: {window}"
            chunks.append(Chunk(page_id=page_id, chunk_id=len(chunks), text=text))

    if not chunks:
        text = f"{title}: {content}"
        chunks.append(Chunk(page_id=page_id, chunk_id=0, text=text))

    return chunks

def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks
