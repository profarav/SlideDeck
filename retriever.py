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

# The Master Sheet was authored for an older, shorter cut of the deck, so its
# slide-number ranges only line up with the actual slides through the front of the
# deck. Up to this slide the sheet is trusted for client + industry; beyond it, vision
# (which reads the actual image) is trusted instead. See _load_slides.
#
# Boundary verified against the PNGs: slides through the "62 & 99-100 SketchPro" row
# are aligned — 45 is Ta'Da branding, 60 is 1860 Equity, 70-98 are the Pantera/CodeThread/
# LightOn/HumanFirst/Guru landing pages, and 99-100 literally read "SketchPro / Andre,
# SketchPro CEO". Drift starts at 103 (shows Denim, sheet says "Agency Portfolio") and is
# pervasive after: 104 sheet "ClarityPay" is the Denim slide; 411-420 "Recuro Health /
# Healthcare" is the Ivana Asset private-credit finance deck; 235 "Consumer Brands" is a
# Physical-Therapy billing page.
_SHEET_ALIGNED_MAX_SLIDE = 100

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


# Canonical display name keyed by a squashed form of the client name (lowercased,
# punctuation and corporate suffixes removed, spaces stripped). Vision spells the same
# company several ways — casing (KITCO/Kitco), spacing (Evergreen Wealth/EvergreenWealth),
# suffixes (Toodo/Toodo Inc.) and OCR misreads (Pantera/Parteira) — which fragmented
# example grouping and the dedupe-by-client in retrieve_examples. Only clear variants of
# the SAME company are mapped; look-alikes that are different firms (ClarityPay vs Clarity
# Health vs Clarity) are deliberately left alone.
_CLIENT_CANONICAL: dict[str, str] = {
    "evergreenwealth": "Evergreen Wealth", "everproperwealth": "Evergreen Wealth",
    "claritypay": "ClarityPay", "claripay": "ClarityPay", "clearlypay": "ClarityPay",
    "lighton": "LightOn", "lightn": "LightOn", "lightai": "LightOn",
    "pantera": "Pantera", "parteira": "Pantera",
    "kawin": "Kawin", "kawm": "Kawin",
    "sohva": "Sohva", "solva": "Sohva",
    "guru": "Guru", "formguru": "Guru", "guruformerlyformguru": "Guru",
    "traumacare": "TraumaCare", "traumacareai": "TraumaCare",
    "connectivehealth": "Connective Health",
    "toodo": "Toodo",
    "longshortlist": "Long Short List", "longsmartlist": "Long Short List",
    "kitco": "Kitco",
    "cyberwhyze": "Cyberwhyze", "cyberwhale": "Cyberwhyze",
    "eventio": "Eventio", "evento": "Eventio",
    "matthewcorwin": "Matthew Corwin", "matthewcorwink": "Matthew Corwin",
    "aerolinktechnologies": "AeroLink Technologies", "aerolinktecnologias": "AeroLink Technologies",
    "svlinvestmentmanagement": "SVL Investment Management", "svl": "SVL Investment Management",
    "svi": "SVL Investment Management",
    "availity": "Availity", "avality": "Availity",
    "seafare": "SeaFare", "seafire": "SeaFare",
}


def _canonical_client(name: str) -> str:
    """Fold known spelling variants of a client name to one display name (else unchanged)."""
    if not name:
        return name
    key = _re.sub(r"[^a-z0-9]+", " ", name.lower())
    key = _re.sub(r"\b(inc|incorporated|llc|ltd|co|corp|company|the)\b", " ", key)
    key = key.replace(" ", "")
    return _CLIENT_CANONICAL.get(key, name)


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
    Load and merge the vision index (per-slide image analysis) with the Master Sheet
    index (curated client/industry/content).

    The two are aligned only through the front of the deck; the sheet was authored for
    an older, shorter cut, so from ~slide _SHEET_ALIGNED_MAX_SLIDE its rows drift onto
    the wrong slides. So we trust the sheet's client + industry for the aligned front
    and fall back to vision's image-derived values for the drifted back. Service
    category (position-based) and the vision text blob are used throughout.
    Falls back to sheet-only if the vision index is absent.
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
        # Service type/category is a position-based taxonomy (get_service_tags keyed on
        # slide number). Sheet and vision compute it identically, and the deck's
        # work-type sections stay broadly stable despite the drift below, so it is kept
        # from the sheet for every slide.
        enriched["service_type"] = sheet.get("service_type", enriched.get("service_type", ""))
        enriched["service_category"] = sheet.get("service_category", enriched.get("service_category", ""))

        if int(sn) <= _SHEET_ALIGNED_MAX_SLIDE:
            # Front of the deck: the sheet's slide numbers line up with the real slides,
            # and its curated client + industry are cleaner and more consistent than
            # vision (which yields Unknowns and per-slide industry flips). Trust the sheet
            # and enrich the keyword blob with its curated content.
            if sheet.get("industry"):
                enriched["industry"] = sheet["industry"]
            if sheet.get("client"):
                enriched["client"] = sheet["client"]
            sheet_content = sheet.get("content", "")
            if sheet_content and sheet_content not in enriched.get("text", ""):
                enriched["text"] = enriched.get("text", "") + " " + sheet_content
        # Back of the deck (slide > _SHEET_ALIGNED_MAX_SLIDE): the sheet row describes a
        # DIFFERENT slide than is actually here, so grafting on its client/industry/content
        # would mislabel the slide (this is what put finance decks under "Healthcare").
        # Vision reads the real image, so its client + industry + text are the aligned
        # source and are kept untouched. Unnamed/hallucinated vision clients are dropped
        # downstream via _EXCLUDE_CLIENTS.
        merged.append(enriched)

    # Fold client-name spelling variants to one canonical name (in memory only).
    for s in merged:
        s["client"] = _canonical_client(s.get("client", ""))

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
    # "_core" slides matched the persona's industry as well as its work type. Give
    # them a prior so unconditional service_category widening (see
    # _build_filtered_pool) broadens reach without swamping on-industry work.
    scored = sorted(
        candidates,
        key=lambda c: _keyword_score(tokens, c.get("text", "")) + (2 if c.get("_core") else 0),
        reverse=True,
    )
    return scored[:top_n]


def _parse_rank_json(raw: str) -> tuple[list[int], dict[str, float], dict[str, str]]:
    """
    Parse a ranking response of the form
        {"indices": [...], "scores": {"i": 0.x}, "reasons": {"i": "..."}}
    Tolerant of truncation: when the model's JSON is cut off mid-string (which happens
    when many reasons overflow max_tokens), fall back to regex-salvaging whatever
    scores/reasons/indices are present rather than dropping them all to 0.0.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return (
                obj.get("indices", []),
                {str(k): float(v) for k, v in obj.get("scores", {}).items()},
                {str(k): v for k, v in obj.get("reasons", {}).items()},
            )
        if isinstance(obj, list):
            return obj, {}, {}
    except (json.JSONDecodeError, ValueError):
        pass

    # Lenient salvage from malformed/truncated JSON.
    indices: list[int] = []
    m = _re.search(r'"indices"\s*:\s*\[([\d,\s]+)', raw)
    if m:
        indices = [int(x) for x in _re.findall(r"\d+", m.group(1))]
    scores = {k: float(v) for k, v in _re.findall(r'"(\d+)"\s*:\s*(-?\d+(?:\.\d+)?)', raw)}
    reasons = {k: v for k, v in _re.findall(r'"(\d+)"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)}
    if not indices and scores:
        indices = [int(k) for k in sorted(scores, key=lambda k: scores[k], reverse=True)]
    return indices, scores, reasons


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

Return a JSON object with exactly three keys:
- "indices": array of the {n} best slide indices (0-based), ordered best-first
- "scores": object mapping each chosen index (as a string) to a strict 0.0–1.0 relevance score. Only a genuinely strong match scores above 0.7; a wrong work type or unrelated industry should score below 0.5.
- "reasons": object mapping each chosen index (as a string) to a single sentence explaining specifically why this slide fits this prospect

Prioritize: (1) work type match, (2) industry match, (3) content relevance, (4) visual style.
Example: {{"indices": [2, 0, 4], "scores": {{"2": 0.92, "0": 0.71, "4": 0.44}}, "reasons": {{"2": "Directly showcases B2B SaaS dashboard work for a fintech client.", "0": "Landing page for a payments company matches the prospect's embedded finance product.", "4": "Investor deck, but for a healthcare startup rather than this prospect's space."}}}}
Return only the JSON object, nothing else."""

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    indices, scores, reasons = _parse_rank_json(msg.content[0].text)
    valid = [i for i in indices if isinstance(i, int) and 0 <= i < len(candidates)]
    total = len(valid)
    results = []
    for rank, i in enumerate(valid[:n]):
        slide = dict(candidates[i])
        # Use Claude's relevance score when provided; otherwise fall back to a mild
        # rank-position estimate (not the confident 1.0-anchored value used before).
        if str(i) in scores:
            slide["score"] = round(max(0.0, min(1.0, scores[str(i)])), 3)
        else:
            slide["score"] = round(0.65 - (rank / max(total, 1)) * 0.3, 3)
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
    # Strip agency-internal / unnamed / hallucinated clients up front so the fallback
    # levels below count only presentable slides. (Doing this after the fallback let a
    # level fill with all-excluded slides, suppress Level 5, then dedup to nothing.)
    slides = [s for s in _load_slides() if s.get("client") not in _EXCLUDE_CLIENTS]
    industries: list[str] = persona["industries"]
    service_categories: list[str] = persona.get("service_categories", [])

    # ── Level 1: industry + service_category (perfect match) ──────────────────
    filtered = [
        s for s in slides
        if s["industry"] in industries
        and s.get("service_category") in service_categories
    ]

    for s in filtered:
        s["_core"] = True

    # ── Level 2: service_category only ────────────────────────────────────────
    # Right work type from any industry beats wrong work type from right industry.
    #
    # This widening is UNCONDITIONAL, not a <3 fallback. Industry tags are known
    # to be unreliable on the back half of the deck (see _SHEET_ALIGNED_MAX_SLIDE),
    # so gating on industry makes a mis-tagged slide structurally unreachable no
    # matter how well it matches: e.g. the Primer design-agency site is tagged
    # "AI & Technology", so a design-agency prospect could never surface it even
    # though it is the single most on-point example in the library. Industry is a
    # prior, not a gate — core (industry-matching) slides keep a ranking bonus
    # below, but strong cross-industry work is allowed to compete.
    if service_categories:
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

    # Deduplicate by slide number (exclusion already applied to `slides` above).
    seen_nums: set[str] = set()
    deduped: list[dict] = []
    for s in filtered:
        if s["slide_number"] not in seen_nums:
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
        # Backfill is not a judged match — score it low and honestly so it never
        # outranks a real Claude-scored result or reads as a strong recommendation.
        item.setdefault("score", 0.35)
        item.setdefault("why", "Added to fill the requested count — not a strong match.")
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
        # True when any slide in the group matched the persona's industry, so the
        # example-level pre-rank can apply the same prior as the slide-level one.
        "_core": any(s.get("_core") for s in sorted_slides),
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
    # Deliberately NO industry ("_core") bonus here. Examples are already few enough
    # that Claude can see most of them, and a thumb on the scale for on-industry work
    # crowded out the strongest cross-industry matches (the Primer / Chemistry design
    # agencies) before the ranker ever got a look at them.
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

What makes an example relevant, in priority order:
1. The client's OWN business is the same as or closely adjacent to the prospect's.
   A salesperson showing a design studio wants to open with "here's work we did for
   another design studio" — a peer beats a stylistic lookalike from an unrelated
   sector. Weight this above visual style.
2. The work type matches what the prospect needs ({service_str}).
3. The visual style is a good reference for the prospect's taste. This is a
   tiebreaker, NOT a substitute for 1 and 2 — do not rank a finance or e-commerce
   client highly for a creative-agency prospect just because it looks bold.

The "industry" label on each example is auto-generated and is often wrong. Judge the
client's actual business from its name and description text, not from that label.

Return a JSON object with exactly three keys:
- "indices": array of ALL example indices ordered best-first
- "scores": object mapping each index (as a string) to a 0.0–1.0 relevance score
- "reasons": object mapping each index (as a string) to one sentence explaining the fit

Example: {{"indices": [2, 0, 1], "scores": {{"2": 0.91, "0": 0.74, "1": 0.45}}, "reasons": {{"2": "Fintech landing page for a payments startup matches the prospect perfectly.", "0": "...", "1": "..."}}}}
Return only the JSON object, nothing else."""

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    indices, scores, reasons = _parse_rank_json(msg.content[0].text)

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

    # Widening the pool on service_category (see _build_filtered_pool) means far more
    # examples reach this point, and a 25-wide keyword cut was dropping the strongest
    # matches before ranking. Keyword overlap is a weak proxy for relevance, so let
    # Claude — the part that actually judges well — see a much larger slice.
    top_n = min(max(60, n_results * 4), len(examples))
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
