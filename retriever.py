"""
Retriever — three-dimension pipeline:
  Stage 1: Hard metadata filter (industry + service_category) with 5-level fallback
  Stage 2: Keyword pre-rank → Claude final ranking
"""

import json
import os
import re as _re
from collections import Counter
from pathlib import Path

import anthropic

from ingest import INDEX_PATH

VISION_INDEX_PATH = str(Path(__file__).parent / "vision_index.json")

_slides_cache: list[dict] | None = None

# A sheet row that spans this many slides or more is treated as a multi-company
# "showcase" range (a portfolio grab-bag under one generic heading) rather than a
# single client. See _load_slides for how the two regimes are merged. Real single
# clients in this library never exceed five slides in one row; every longer row is a
# section header or a vertical showcase bundling many different companies.
_SHOWCASE_MIN_SLIDES = 6

# Client values that must never surface as a case study example. Two groups:
#   1. Agency-internal section headers (short sheet rows: intro, team, pricing, …) —
#      these are fly sheets, not client work.
#   2. Vision misreads / placeholders that leak in from showcase ranges (UI chrome it
#      mistook for a brand, lorem-ipsum, redaction markers).
_EXCLUDE_CLIENTS: frozenset[str] = frozenset({
    "Klimt & Design",
    "Unknown",
    # Agency/section headers (short single-client-regime rows)
    "Intro", "Team", "Process", "Testimonials", "Clients",
    "UX/UI Design", "Identity Design", "Interaction", "Information", "Content/Ads",
    "Capability Intro", "SEO Offerings", "Pricing & Packages",
    "Misc. Logos", "Paid Media / Ads",
    "Stylescape", "Color & Palette", "Logo Concepts", "Team & Workflow",
    # Vision-hallucinated placeholders (never real clients)
    "Gmail", "Okta", "ACME", "ACME Inc.", "ACME Insight", "amet", "Redacted",
    "YOUR BRAND", "ario", "hinlab", "Bookkeeping Software", "ABC Offer",
})


def _parse_sheet_slide_numbers(sn: str) -> list[int]:
    """Parse slide_index slide_number formats: '1-3', '4 & 6', '62 & 99-100'."""
    nums: list[int] = []
    for part in sn.replace("&", ",").split(","):
        part = part.strip()
        if "-" in part:
            try:
                lo, hi = map(int, part.split("-", 1))
                nums.extend(range(lo, hi + 1))
            except ValueError:
                pass
        elif part.isdigit():
            nums.append(int(part))
    return nums


def _load_slides() -> list[dict]:
    """
    Always load both indices and merge them.
    Vision index provides rich per-slide descriptions; sheet index provides
    authoritative client names and curated content descriptions.
    For slides where vision has 'Unknown' client, the sheet name is used.
    Sheet content is appended to vision text for richer keyword matching.
    Falls back to sheet-only if vision index is absent.
    """
    global _slides_cache
    if _slides_cache is not None:
        return _slides_cache

    # Build sheet lookup: slide_number (int) → sheet entry
    sheet_map: dict[int, dict] = {}
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH) as f:
            sheet_entries = json.load(f)
        for entry in sheet_entries:
            for num in _parse_sheet_slide_numbers(str(entry.get("slide_number", ""))):
                if num not in sheet_map:
                    sheet_map[num] = entry

    if not os.path.exists(VISION_INDEX_PATH):
        _slides_cache = list(sheet_map.values()) if sheet_map else []
        return _slides_cache

    with open(VISION_INDEX_PATH) as f:
        vision_slides = json.load(f)

    merged: list[dict] = []
    for slide in vision_slides:
        sn = str(slide.get("slide_number", ""))
        if not sn.isdigit():
            merged.append(slide)
            continue
        sheet = sheet_map.get(int(sn))
        if not sheet:
            merged.append(slide)
            continue
        enriched = dict(slide)
        # Sheet service_type/category is authoritative for both regimes (range-based
        # taxonomy from Excel; a whole range is one type of work).
        enriched["service_type"] = sheet.get("service_type", enriched.get("service_type", ""))
        enriched["service_category"] = sheet.get("service_category", enriched.get("service_category", ""))

        # Sheet industry is authoritative for every sheet-backed slide. It is
        # human-curated and consistent, whereas vision assigns industry per-slide from
        # the image and disagrees ~85% of the time — pure noise that scattered one
        # client across many industries. Industry is a hard Stage-1 filter dimension, so
        # a predictable curator-intended tag beats vision's scatter (including inside
        # showcase ranges, where the umbrella reflects how the deck author grouped them).
        if sheet.get("industry"):
            enriched["industry"] = sheet["industry"]

        # Client name uses two regimes based on how many slides the sheet row spans:
        is_showcase = len(_parse_sheet_slide_numbers(str(sheet.get("slide_number", "")))) >= _SHOWCASE_MIN_SLIDES
        if not is_showcase and sheet.get("client"):
            # Single-client row: the sheet name is the real client and is cleaner than
            # vision (which misreads/abbreviates), so it is authoritative.
            enriched["client"] = sheet["client"]
        # Showcase row (>= _SHOWCASE_MIN_SLIDES): the sheet packs many different companies
        # under one generic label ("Consumer Brands", "UX/UI Deep Dive"), so its single
        # client name is useless. Keep vision's per-slide company name so the range splits
        # into real, individually-named examples instead of one mislabelled blob.
        # Unnamed / hallucinated vision clients are dropped later via _EXCLUDE_CLIENTS.

        # Append curated sheet content to vision text for keyword matching
        sheet_content = sheet.get("content", "")
        if sheet_content and sheet_content not in enriched.get("text", ""):
            enriched["text"] = enriched.get("text", "") + " " + sheet_content
        merged.append(enriched)

    _slides_cache = merged
    return _slides_cache


_STOPWORDS = frozenset({
    "a", "an", "the", "for", "of", "in", "on", "and", "or", "to", "with", "is",
    "are", "that", "this", "we", "our", "their", "they", "it", "its", "as", "at",
    "by", "be", "from", "need", "needs", "want", "looking", "help",
})


def _query_tokens(query: str) -> set[str]:
    """Whole-word query tokens with stopwords and single characters removed."""
    return {t for t in _re.findall(r"[a-z][a-z0-9]+", query.lower()) if t not in _STOPWORDS}


def _keyword_score(tokens: set[str], text: str) -> int:
    """
    Frequency-weighted overlap between query tokens and a text blob, matched on
    whole words. Whole-word matching matters: a substring test lets short tokens
    like "ai" or "ux" match "captain"/"email"/"auxiliary", which polluted the
    pre-rank and could drop genuinely relevant slides before Claude ever saw them.
    """
    if not tokens:
        return 0
    counts = Counter(_re.findall(r"[a-z][a-z0-9]+", text.lower()))
    return sum(counts[t] for t in tokens)


def _keyword_rank(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    """
    Fast keyword pre-filter to trim candidate list before Claude ranking.
    No external dependencies — pure Python, safe for Vercel's 250 MB limit.
    """
    if len(candidates) <= top_n:
        return candidates

    tokens = _query_tokens(query)
    scored = sorted(candidates, key=lambda c: _keyword_score(tokens, c.get("text", "")), reverse=True)
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


def _build_filtered_pool(persona: dict) -> list[dict]:
    """
    Build filtered pool with 5-level fallback:
      Level 1: industry ∩ service_category  (tightest — both match)
      Level 2: service_category only        (right work type beats right industry)
      Level 3: industry only                (drop service_category constraint)
      Level 4: General Agency slides        (agency overview fallback)
      Level 5: entire library               (last resort)
    """
    slides = _load_slides()
    industries: list[str] = persona["industries"]
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

    # Deduplicate and strip agency-internal slides before ranking.
    seen_nums: set[str] = set()
    deduped: list[dict] = []
    for s in filtered:
        if s["slide_number"] not in seen_nums and s.get("client") not in _EXCLUDE_CLIENTS:
            seen_nums.add(s["slide_number"])
            deduped.append(s)
    return deduped


def _merge_ranked_with_pins(
    *,
    filtered: list[dict],
    ranked: list[dict],
    pinned: list[str],
    excluded: set[str],
    n_results: int,
) -> list[dict]:
    by_num = {s["slide_number"]: s for s in filtered}
    results: list[dict] = []
    seen: set[str] = set()

    for slide_num in pinned:
        if slide_num in excluded:
            continue
        slide = by_num.get(slide_num)
        if not slide:
            continue
        item = dict(slide)
        item.setdefault("score", 1.0)
        item.setdefault("why", "Pinned by user request.")
        results.append(item)
        seen.add(slide_num)

    for s in ranked:
        num = s["slide_number"]
        if num in excluded or num in seen:
            continue
        results.append(s)
        seen.add(num)
        if len(results) >= n_results:
            return results[:n_results]

    for s in filtered:
        num = s["slide_number"]
        if num in excluded or num in seen:
            continue
        item = dict(s)
        item.setdefault("score", 0.7)
        item.setdefault("why", "Backfill to complete requested count.")
        results.append(item)
        seen.add(num)
        if len(results) >= n_results:
            break

    return results[:n_results]


def retrieve_case_studies_with_controls(
    persona: dict,
    *,
    n_results: int = 8,
    pinned: list[str] | None = None,
    excluded: list[str] | None = None,
    previous_visible: list[str] | None = None,
    variant: str = "",
) -> list[dict]:
    query: str = persona["search_query"]
    visual_style: str = persona["visual_style"]
    service_categories: list[str] = persona.get("service_categories", [])

    filtered = _build_filtered_pool(persona)
    pinned = pinned or []
    excluded_set = set(excluded or [])
    previous_visible_set = set(previous_visible or [])

    hard_excluded = set(excluded_set)
    if variant == "more_like_this":
        hard_excluded.update(s for s in previous_visible_set if s not in set(pinned))

    pinned_set = set(pinned)
    rank_pool = [
        s for s in filtered
        if s["slide_number"] not in pinned_set
        and s["slide_number"] not in hard_excluded
    ]

    top_n = min(max(20, n_results * 4), len(rank_pool))
    candidates = _keyword_rank(query, rank_pool, top_n=top_n)

    ranked_count = min(len(candidates), max(n_results * 3, n_results))
    ranked = _claude_rank(query, candidates, visual_style, service_categories, n=ranked_count) if candidates else []
    merged = _merge_ranked_with_pins(
        filtered=filtered,
        ranked=ranked,
        pinned=pinned,
        excluded=excluded_set,
        n_results=n_results,
    )

    if len(merged) >= n_results or variant != "more_like_this":
        return merged[:n_results]

    # Backfill in "more_like_this" mode if temporary exclusions removed too many options.
    relaxed_pool = [
        s for s in filtered
        if s["slide_number"] not in pinned_set
        and s["slide_number"] not in excluded_set
    ]
    relaxed_top_n = min(max(20, n_results * 4), len(relaxed_pool))
    relaxed_candidates = _keyword_rank(query, relaxed_pool, top_n=relaxed_top_n)
    relaxed_ranked_count = min(len(relaxed_candidates), max(n_results * 3, n_results))
    relaxed_ranked = _claude_rank(
        query, relaxed_candidates, visual_style, service_categories, n=relaxed_ranked_count
    ) if relaxed_candidates else []
    return _merge_ranked_with_pins(
        filtered=filtered,
        ranked=relaxed_ranked,
        pinned=pinned,
        excluded=excluded_set,
        n_results=n_results,
    )


def retrieve_case_studies(persona: dict, n_results: int = 8) -> list[dict]:
    filtered = _build_filtered_pool(persona)
    query: str = persona["search_query"]
    visual_style: str = persona["visual_style"]
    service_categories: list[str] = persona.get("service_categories", [])

    # ── Stage 2: Keyword pre-rank (keep top 20 for Claude) ────────────────────
    candidates = _keyword_rank(query, filtered, top_n=min(20, len(filtered)))

    # ── Stage 3: Claude final ranking ─────────────────────────────────────────
    ranked = _claude_rank(query, candidates, visual_style, service_categories, n=n_results)

    return ranked


# ── Examples API ──────────────────────────────────────────────────────────────

def _make_example(slides: list[dict]) -> dict:
    sorted_slides = sorted(slides, key=lambda x: int(x["slide_number"]))
    nums = [int(s["slide_number"]) for s in sorted_slides]
    rep = sorted_slides[0]
    text = " ".join(
        (s.get("text", "") + " " + s.get("content", "")).strip()
        for s in sorted_slides
    )
    return {
        "client": rep.get("client", "Unknown"),
        "service_type": rep.get("service_type", ""),
        "service_category": rep.get("service_category", ""),
        "industry": rep.get("industry", ""),
        "visual_style": rep.get("visual_style", ""),
        "slide_range": [min(nums), max(nums)],
        "n_slides": len(sorted_slides),
        "representative_slide": str(sorted_slides[0]["slide_number"]),
        "_slides": sorted_slides,
        "_text": text,
        "score": 0.0,
        "why": "",
    }


def _group_into_examples(slides: list[dict]) -> list[dict]:
    """
    Group slides into client examples by scanning sorted slide numbers.
    A new group starts when client name, service_category, or slide gap (>5) changes.
    """
    sorted_slides = sorted(slides, key=lambda x: int(x["slide_number"]))
    if not sorted_slides:
        return []

    groups: list[list[dict]] = [[sorted_slides[0]]]
    for s in sorted_slides[1:]:
        prev = groups[-1][-1]
        same_client = s.get("client", "").lower() == prev.get("client", "").lower()
        same_service = s.get("service_category", "") == prev.get("service_category", "")
        gap = int(s["slide_number"]) - int(prev["slide_number"])
        if same_client and same_service and gap <= 5:
            groups[-1].append(s)
        else:
            groups.append([s])

    return [_make_example(g) for g in groups]


def _keyword_rank_examples(query: str, examples: list[dict], top_n: int) -> list[dict]:
    if len(examples) <= top_n:
        return examples
    tokens = _query_tokens(query)
    return sorted(examples, key=lambda e: _keyword_score(tokens, e.get("_text", "")), reverse=True)[:top_n]


def _claude_rank_examples(
    query: str,
    examples: list[dict],
    visual_style: str,
    service_categories: list[str],
) -> list[dict]:
    """Ask Claude to score and rank all candidate examples. Returns examples with score and why populated."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    example_list = "\n".join(
        f"[{i}] {e['client']} | {e['service_type']} | "
        f"Slides {e['slide_range'][0]}–{e['slide_range'][1]} | {e['industry']} | {e['n_slides']} slides\n"
        f"    {e['_text'][:250]}"
        for i, e in enumerate(examples)
    )

    service_str = ", ".join(service_categories) if service_categories else "Not specified"

    prompt = f"""You are a creative agency business development analyst for Klimt & Design.
A salesperson needs client examples from the portfolio to show a prospect.

Prospect needs: {query}
Required work type: {service_str}
Preferred visual style: {visual_style}

Available case study examples:
{example_list}

Rank ALL examples and assign a strict relevance score. Only truly relevant examples should score above 0.7.

Return a JSON object with exactly three keys:
- "indices": array of ALL example indices ordered best-first
- "scores": object mapping each index (as a string) to a 0.0–1.0 relevance score
- "reasons": object mapping each index (as a string) to one sentence explaining the fit

Example: {{"indices": [2, 0, 1], "scores": {{"2": 0.91, "0": 0.74, "1": 0.45}}, "reasons": {{"2": "Fintech landing page for a payments startup matches the prospect perfectly.", "0": "...", "1": "..."}}}}
Return only the JSON object, nothing else."""

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()

    indices: list[int] = []
    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            indices = obj.get("indices", [])
            scores = {str(k): float(v) for k, v in obj.get("scores", {}).items()}
            reasons = {str(k): v for k, v in obj.get("reasons", {}).items()}
    except (json.JSONDecodeError, ValueError):
        arr_match = _re.search(r"\[[\d,\s]+\]", raw)
        if arr_match:
            indices = json.loads(arr_match.group(0))

    # If Claude omitted some indices, append them at the end with score 0.0
    seen = set(indices)
    for i in range(len(examples)):
        if i not in seen:
            indices.append(i)

    results = []
    for i in indices:
        if not (isinstance(i, int) and 0 <= i < len(examples)):
            continue
        e = dict(examples[i])
        e["score"] = round(scores.get(str(i), 0.0), 3)
        e["why"] = reasons.get(str(i), "")
        results.append(e)
    return results


def retrieve_examples(persona: dict, n_results: int = 8) -> list[dict]:
    """
    Retrieve and score client case study examples rather than individual slides.
    Returns all scored examples (up to cap) sorted by score — caller applies cutoff.
    """
    filtered = _build_filtered_pool(persona)
    examples = _group_into_examples(filtered)

    query: str = persona["search_query"]
    visual_style: str = persona["visual_style"]
    service_categories: list[str] = persona.get("service_categories", [])

    top_n = min(max(25, n_results * 4), len(examples))
    candidates = _keyword_rank_examples(query, examples, top_n=top_n)

    ranked = _claude_rank_examples(query, candidates, visual_style, service_categories)
    ranked.sort(key=lambda e: e["score"], reverse=True)

    # One example per client. The 101-200 "second pass" reuses real client names
    # (Pantera, Guru, LightOn …) as section labels over unrelated grab-bag content,
    # so a client can otherwise appear two or three times. Keep the highest-scored.
    deduped: list[dict] = []
    seen_clients: set[str] = set()
    for e in ranked:
        key = e.get("client", "").strip().lower()
        if key and key in seen_clients:
            continue
        seen_clients.add(key)
        deduped.append(e)

    cap = min(20, n_results * 3)
    return deduped[:cap]
