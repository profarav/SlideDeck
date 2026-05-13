"""
Retriever — three-dimension pipeline:
  Stage 1: Hard metadata filter (industry + service_category) with 5-level fallback
  Stage 2: Keyword pre-rank → Claude final ranking
"""

import json
import os
import re as _re
from pathlib import Path

import anthropic

from ingest import INDEX_PATH

VISION_INDEX_PATH = str(Path(__file__).parent / "vision_index.json")

_slides_cache: list[dict] | None = None


def _load_slides() -> list[dict]:
    global _slides_cache
    if _slides_cache is None:
        # Prefer vision index (per-slide accuracy) over the sheet-based index
        path = VISION_INDEX_PATH if os.path.exists(VISION_INDEX_PATH) else INDEX_PATH
        with open(path) as f:
            _slides_cache = json.load(f)
    return _slides_cache


def _keyword_rank(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    """
    Fast keyword pre-filter to trim candidate list before Claude ranking.
    Scores each candidate by counting how many query tokens appear in its text blob.
    No external dependencies — pure Python, safe for Vercel's 250 MB limit.
    """
    if len(candidates) <= top_n:
        return candidates

    tokens = set(_re.findall(r"[a-z]+", query.lower()))
    # Remove common stopwords
    stopwords = {"a", "an", "the", "for", "of", "in", "on", "and", "or", "to", "with", "is", "are", "that", "this", "we", "our"}
    tokens -= stopwords

    def _score(c: dict) -> int:
        blob = c.get("text", "").lower()
        return sum(1 for t in tokens if t in blob)

    scored = sorted(candidates, key=_score, reverse=True)
    return scored[:top_n]


def _claude_rank(
    query: str,
    candidates: list[dict],
    visual_style: str,
    service_categories: list[str],
    n: int,
) -> list[dict]:
    """Ask Claude to pick and order the most relevant slides from the candidate list."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    slide_list = "\n".join(
        f"[{i}] Slide {c['slide_number']} | {c['client']} ({c.get('industry_raw', c.get('industry', ''))}) "
        f"| Type: {c.get('service_type', '?')} "
        f"| Style: {c.get('visual_style_raw', c.get('visual_style', ''))[:50]} | {c['content'][:100]}"
        for i, c in enumerate(candidates)
    )

    service_str = ", ".join(service_categories) if service_categories else "Not specified"

    prompt = f"""You are a creative agency business development analyst for Klimt & Design.

A salesperson is building a pitch deck and needs the {n} most relevant case study slides.

Prospect needs: {query}
Required work type: {service_str}
Preferred visual style: {visual_style}

Available slides:
{slide_list}

Return a JSON object with exactly two keys:
- "indices": array of the {n} best slide indices (0-based), ordered best-first
- "reasons": object mapping each chosen index (as a string) to a single sentence explaining specifically why this slide fits this prospect

Prioritize: (1) work type match, (2) industry match, (3) content relevance, (4) visual style.
Example: {{"indices": [2, 0, 4], "reasons": {{"2": "Directly showcases B2B SaaS dashboard work for a fintech client.", "0": "Landing page for a payments company matches the prospect's embedded finance product.", "4": "Investor deck for a healthcare startup mirrors the client's fundraising context."}}}}
Return only the JSON object, nothing else."""

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()

    # Parse the response — try full object first, fall back to array-only
    indices = []
    reasons: dict[str, str] = {}
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            indices = obj.get("indices", [])
            reasons = {str(k): v for k, v in obj.get("reasons", {}).items()}
        elif isinstance(obj, list):
            indices = obj
    except json.JSONDecodeError:
        arr_match = _re.search(r"\[[\d,\s]+\]", raw)
        if arr_match:
            indices = json.loads(arr_match.group(0))

    valid = [i for i in indices if isinstance(i, int) and 0 <= i < len(candidates)]
    total = len(valid)
    results = []
    for rank, i in enumerate(valid[:n]):
        slide = dict(candidates[i])
        slide["score"] = round(1.0 - (rank / max(total, 1)) * 0.3, 3)
        slide["why"] = reasons.get(str(i), "")
        results.append(slide)
    return results


def retrieve_case_studies(persona: dict, n_results: int = 8) -> list[dict]:
    """
    Full retrieval pipeline with 5-level fallback:
      Level 1: industry ∩ service_category  (tightest — both match)
      Level 2: service_category only        (right work type beats right industry)
      Level 3: industry only                (drop service_category constraint)
      Level 4: General Agency slides        (agency overview fallback)
      Level 5: entire library               (last resort)
    """
    slides = _load_slides()
    industries: list[str] = persona["industries"]
    query: str = persona["search_query"]
    visual_style: str = persona["visual_style"]
    service_categories: list[str] = persona.get("service_categories", [])

    # ── Level 1: industry + service_category (perfect match) ──────────────────
    filtered = [
        s for s in slides
        if s["industry"] in industries
        and s.get("service_category") in service_categories
    ]

    # ── Level 2: service_category only ────────────────────────────────────────
    # Right work type from any industry beats wrong work type from right industry.
    if len(filtered) < 3 and service_categories:
        seen = {s["slide_number"] for s in filtered}
        filtered += [
            s for s in slides
            if s.get("service_category") in service_categories
            and s["slide_number"] not in seen
        ]

    # ── Level 3: industry only ─────────────────────────────────────────────────
    if len(filtered) < 3:
        seen = {s["slide_number"] for s in filtered}
        filtered += [s for s in slides if s["industry"] in industries and s["slide_number"] not in seen]

    # ── Level 4: General Agency fallback ──────────────────────────────────────
    if len(filtered) < 3:
        seen = {s["slide_number"] for s in filtered}
        filtered += [s for s in slides if s["industry"] == "General Agency" and s["slide_number"] not in seen]

    # ── Level 5: entire library ────────────────────────────────────────────────
    if not filtered:
        filtered = slides

    # Deduplicate filtered list by slide_number before ranking
    seen_nums: set[str] = set()
    deduped: list[dict] = []
    for s in filtered:
        if s["slide_number"] not in seen_nums:
            seen_nums.add(s["slide_number"])
            deduped.append(s)
    filtered = deduped

    # ── Stage 2: Keyword pre-rank (keep top 20 for Claude) ────────────────────
    candidates = _keyword_rank(query, filtered, top_n=min(20, len(filtered)))

    # ── Stage 3: Claude final ranking ─────────────────────────────────────────
    ranked = _claude_rank(query, candidates, visual_style, service_categories, n=n_results)

    return ranked
