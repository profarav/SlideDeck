"""
Prospect Profiler — uses Claude to analyze a raw prospect description and
return a structured JSON "Search Persona" used to drive retrieval.
"""

import json
import os
import anthropic
from tags import CANONICAL_INDUSTRIES, CANONICAL_VISUAL_STYLES

_SYSTEM_PROMPT = f"""You are a creative agency business development analyst for Klimt & Design,
a premium design agency. Your job is to analyze a prospect description and extract a
structured "Search Persona" so the team can pull the most relevant case studies from the
slide library.

Return ONLY a valid JSON object with these exact keys:
{{
  "industries": <list of 1-3 strings, chosen from the canonical list>,
  "visual_style": <single string, chosen from the canonical list>,
  "search_query": <a rich 1-2 sentence description of what design work this prospect needs,
                  written as if searching for matching portfolio slides>,
  "reasoning": <1 sentence explaining your choices>
}}

Canonical industry tags (pick the closest matches):
{json.dumps(CANONICAL_INDUSTRIES, indent=2)}

Canonical visual style tags (pick the single best fit):
{json.dumps(CANONICAL_VISUAL_STYLES, indent=2)}

Rules:
- industries: pick 1-3 tags that best describe the prospect's sector. If it's a fintech
  payments startup, pick ["Fintech & Payments", "B2B SaaS"].
- visual_style: pick the style that matches the prospect's brand personality. A luxury
  wealth manager → "Luxury". A developer tools startup → "Technical" or "Futuristic".
- search_query: write this as a rich description of the case studies you'd want to find —
  mention the industry, the type of design work needed, and any relevant visual qualities.
- Do NOT invent new tag names. Only use tags from the lists above.
"""


def build_search_persona(prospect_description: str) -> dict:
    """
    Analyze a prospect description and return a Search Persona dict:
    {industries, visual_style, search_query, reasoning}
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set in environment.")

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # cache the large system prompt
            }
        ],
        messages=[
            {
                "role": "user",
                "content": f"Prospect description:\n\n{prospect_description}",
            }
        ],
    )

    raw = message.content[0].text.strip()

    # Strip markdown code fences if Claude wraps in ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    persona = json.loads(raw)

    # Validate structure
    required = {"industries", "visual_style", "search_query", "reasoning"}
    missing = required - set(persona.keys())
    if missing:
        raise ValueError(f"Persona missing keys: {missing}")

    return persona
