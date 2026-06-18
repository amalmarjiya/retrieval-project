"""
Build page_signatures.json: groups pages that are likely relevant to the
same queries even without shared proper nouns, using two signals:

1. Named-entity grouping - pages mentioning the same proper nouns
   (people, places, organizations) extracted via simple capitalized-phrase
   heuristics.
2. Template-pattern grouping - pages whose content matches the same
   synthetic phrase template (e.g. "averaged a team-high N points"),
   since this corpus reuses template structures across many pages.

Output: artifacts/page_signatures.json with a "page_to_siblings" map.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import iter_entries, ensure_artifacts_dir, entry_text

# Template patterns observed in this synthetic corpus - phrases with the
# variable part stripped out, used as a normalized "fingerprint".
TEMPLATE_PATTERNS = [
    (r"averaged a team-high \d+ points", "basketball_player"),
    (r"chief executive (officer )?during its international expansion", "company_expansion"),
    (r"assembly lines? (were |was )?modernized", "assembly_factory"),
    (r"(shipbuilding|maritime) (exports|harbor)", "maritime_company"),
    (r"riverfront (festivals|redesign)", "city_economy"),
    (r"research (division|institute)", "research_institute"),
    (r"(diplomatic|observers) (settlement|agreement)", "diplomacy"),
    (r"(memorial|commemorative) (arena|banner)", "championship_arena"),
]

PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]{2,}(?:\s[A-Z][a-z]{2,}){0,2}\b")
COMMON_WORDS = {
    "The", "This", "That", "These", "Those", "It", "He", "She", "They",
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
}


def extract_entities(text: str) -> set[str]:
    candidates = PROPER_NOUN_RE.findall(text)
    return {c for c in candidates if c not in COMMON_WORDS and len(c) > 3}


def extract_templates(text: str) -> set[str]:
    text_lower = text.lower()
    found = set()
    for pattern, label in TEMPLATE_PATTERNS:
        if re.search(pattern, text_lower):
            found.add(label)
    return found


def main():
    entity_to_pages: dict[str, list[int]] = defaultdict(list)
    template_to_pages: dict[str, list[int]] = defaultdict(list)
    page_count = 0

    for record in iter_entries():
        page_id = record["page_id"]
        text = entry_text(record)
        page_count += 1

        for entity in extract_entities(text):
            entity_to_pages[entity].append(page_id)

        for template in extract_templates(text):
            template_to_pages[template].append(page_id)

    page_to_siblings: dict[int, set[int]] = defaultdict(set)

    # Entity-based siblings: pages sharing a distinctive proper noun
    for entity, pages in entity_to_pages.items():
        if 2 <= len(pages) <= 8:  # skip ultra-common or singleton entities
            for p in pages:
                page_to_siblings[p].update(pg for pg in pages if pg != p)

    # Template-based siblings: pages sharing a synthetic phrase template
    for template, pages in template_to_pages.items():
        if len(pages) >= 2:
            for p in pages:
                # cap sibling list size to avoid huge fan-out groups
                others = [pg for pg in pages if pg != p][:15]
                page_to_siblings[p].update(others)

    result = {
        "page_to_siblings": {
            str(k): sorted(v) for k, v in page_to_siblings.items()
        }
    }

    artifacts_dir = ensure_artifacts_dir()
    out_path = artifacts_dir / "page_signatures.json"
    out_path.write_text(json.dumps(result), encoding="utf-8")

    print(f"[signatures] processed {page_count} pages")
    print(f"[signatures] {len(page_to_siblings)} pages have at least one sibling")
    print(f"[signatures] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
