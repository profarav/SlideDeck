"""
Deck Builder — assembles a full ordered pitch deck:
  Section 1: Agency Overview  (2 intro slides from General Agency pool)
  Section 2: Case Studies     (3-5 relevant case studies via retrieval pipeline)
  Section 3: Next Steps       (1 closing slide from General Agency pool)
"""

import json
import os
import re as _re

import anthropic

from retriever import retrieve_case_studies, _load_slides


def _pick_structural_slides(
    agency_slides: list[dict],
    persona: dict,
) -> tuple[list[dict], list[dict]]:
    """Use Claude Haiku to pick the best intro (2) and closing (1) slides."""
    if not agency_slides:
        return [], []

    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    slide_list = "\n".join(
        f"[{i}] Slide {s['slide_number']} | {s.get('content', '')[:150]}"
        for i, s in enumerate(agency_slides[:30])
    )

    prompt = f"""You are assembling a pitch deck for Klimt & Design agency.
Prospect: {', '.join(persona['industries'])} company needing {', '.join(persona.get('service_categories', ['design work']))}

These are Klimt & Design's agency overview slides:
{slide_list}

Pick the best slides for:
- "intro": 2 slides to open the deck (agency intro, what we do, capabilities overview)
- "closing": 1 slide to close the deck (contact info, CTA, next steps, or thank you)

Return only JSON: {{"intro": [0, 2], "closing": [7]}}"""

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        match = _re.search(r'\{[^}]+\}', raw)
        if match:
            obj = json.loads(match.group(0))
            intro_idx = obj.get("intro", [0, 1])
            closing_idx = obj.get("closing", [len(agency_slides) - 1])
        else:
            intro_idx = [0, 1]
            closing_idx = [len(agency_slides) - 1]
    except Exception:
        intro_idx = [0, 1]
        closing_idx = [len(agency_slides) - 1]

    def safe_pick(indices):
        return [agency_slides[i] for i in indices if isinstance(i, int) and 0 <= i < len(agency_slides)]

    return safe_pick(intro_idx), safe_pick(closing_idx)


def build_full_deck(persona: dict, n_case_studies: int = 4) -> dict:
    """
    Assemble a complete ordered pitch deck.
    Returns: {sections: [{label, slides}], ordered_slides: [...], total_slides: int}
    """
    all_slides = _load_slides()

    # General Agency pool for structural slides (slides 1-24)
    agency_slides = [
        s for s in all_slides
        if s.get("service_category") == "General"
        or s.get("service_type") == "General Agency"
    ]

    # Case studies via existing retrieval pipeline
    case_studies = retrieve_case_studies(persona, n_results=n_case_studies)
    case_study_nums = {s["slide_number"] for s in case_studies}

    # Remove any agency slides that also appear as case studies
    agency_slides = [s for s in agency_slides if s["slide_number"] not in case_study_nums]

    # Pick structural slides with Claude
    intro_slides, closing_slides = _pick_structural_slides(agency_slides, persona)

    def _tag(slides, default_why):
        result = []
        for s in slides:
            slide = dict(s)
            slide.setdefault("score", 1.0)
            slide.setdefault("why", default_why)
            result.append(slide)
        return result

    intro = _tag(intro_slides, "Klimt & Design agency introduction and capabilities overview.")
    studies = _tag(case_studies, "")
    closing = _tag(closing_slides, "Closing slide with contact information and next steps.")

    sections = []
    if intro:
        sections.append({"label": "Agency Overview", "slides": intro})
    if studies:
        sections.append({"label": "Case Studies", "slides": studies})
    if closing:
        sections.append({"label": "Next Steps", "slides": closing})

    return {
        "sections": sections,
        "ordered_slides": intro + studies + closing,
        "total_slides": len(intro) + len(studies) + len(closing),
    }
