"""
Retriever — two-stage pipeline:
  Stage 1: Hard metadata filter (industry tags)
  Stage 2: TF-IDF pre-rank → Claude final ranking

No vector DB or model download required.
"""

import json
import os
import anthropic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ingest import INDEX_PATH

_slides_cache: list[dict] | None = None


def _load_slides() -> list[dict]:
    global _slides_cache
    if _slides_cache is None:
        with open(INDEX_PATH) as f:
            _slides_cache = json.load(f)
    return _slides_cache


def _tfidf_rank(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    """Fast TF-IDF pre-filter to trim candidate list before Claude ranking."""
    if len(candidates) <= top_n:
        return candidates
    texts = [c["text"] for c in candidates]
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform(texts + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1])[0]
    ranked_idx = scores.argsort()[::-1][:top_n]
    return [candidates[i] for i in ranked_idx]


def _claude_rank(query: str, candidates: list[dict], visual_style: str, n: int) -> list[dict]:
    """Ask Claude to pick and order the most relevant slides from the candidate list."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    slide_list = "\n".join(
        f"[{i}] Slides {c['slide_number']} | {c['client']} ({c['industry_raw']}) "
        f"| Style: {c['visual_style_raw'][:60]} | {c['content'][:120]}"
        for i, c in enumerate(candidates)
    )

    prompt = f"""You are a creative agency business development analyst for Klimt & Design.

A salesperson is building a pitch deck and needs the {n} most relevant case study slides to show a prospect.

Prospect needs: {query}
Preferred visual style: {visual_style}

Available slides:
{slide_list}

Return ONLY a JSON array of the {n} best slide indices (0-based, from the list above), ordered best-first.
Prioritize: (1) industry match, (2) content relevance, (3) visual style match.
Example: [2, 0, 4, 1, 3]
Return only the JSON array, nothing else."""

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=128,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = msg.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()

    indices = json.loads(raw)
    # Guard against out-of-range indices
    valid = [i for i in indices if isinstance(i, int) and 0 <= i < len(candidates)]
    total = len(valid)
    results = []
    for rank, i in enumerate(valid[:n]):
        slide = dict(candidates[i])
        slide["score"] = round(1.0 - (rank / max(total, 1)) * 0.3, 3)
        results.append(slide)
    return results


def retrieve_case_studies(persona: dict, n_results: int = 8) -> list[dict]:
    """
    Full retrieval pipeline:
      1. Load all slides from the JSON index.
      2. Hard filter by matched industries (fall back to General Agency if < 3 hits).
      3. TF-IDF pre-rank to top-20 candidates.
      4. Claude final ranking to top n_results.
    """
    slides = _load_slides()
    industries: list[str] = persona["industries"]
    query: str = persona["search_query"]
    visual_style: str = persona["visual_style"]

    # ── Stage 1: industry filter ───────────────────────────────────────────────
    filtered = [s for s in slides if s["industry"] in industries]

    # Fallback: if too few, add General Agency slides
    if len(filtered) < 3:
        general = [s for s in slides if s["industry"] == "General Agency"]
        seen = {s["slide_number"] for s in filtered}
        filtered += [s for s in general if s["slide_number"] not in seen]

    # If still nothing, use everything
    if not filtered:
        filtered = slides

    # ── Stage 2: TF-IDF pre-rank (keep top 20 for Claude) ─────────────────────
    candidates = _tfidf_rank(query, filtered, top_n=min(20, len(filtered)))

    # ── Stage 3: Claude final ranking ─────────────────────────────────────────
    ranked = _claude_rank(query, candidates, visual_style, n=n_results)

    return ranked
