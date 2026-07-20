"""
Vision Ingestion — sends every slide image through Claude Vision to generate
accurate per-slide metadata, then writes vision_index.json.

Usage:
    python vision_ingest.py --slides /Users/aravlohe/Downloads/slides_png

Resumable: already-processed slides are skipped on re-run.
Parallel:  10 concurrent API calls.
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import anthropic
from PIL import Image
from dotenv import load_dotenv

from tags import (
    normalize_industry,
    normalize_visual_style,
    get_service_tags,
    CANONICAL_INDUSTRIES,
    CANONICAL_SERVICE_CATEGORIES,
)

load_dotenv()

VISION_INDEX_PATH = "./vision_index.json"
RESIZE_TO = (800, 450)
MAX_WORKERS = 10
MODEL = "claude-haiku-4-5"   # Faster + cheaper for vision extraction

# Vision judges industry directly from the slide. This is far more reliable than
# keyword-matching the prose description after the fact — that mislabels heavily because
# design-agency descriptions are saturated with generic words ("design", "data", "ai").
_SYSTEM_PROMPT = f"""You are analyzing portfolio slides from Klimt & Design, a premium design agency.
For each slide image, extract exactly 5 fields and return ONLY valid JSON — no other text.

A salesperson uses these slides to show a prospect "here's comparable work we've
done." The `description` is the ONLY thing a downstream matcher sees — it never sees
the image — so make it rich enough to judge fit WITHOUT looking. Be concrete: name
what you actually see.

{{
  "client": "<company or project name visible on the slide, or 'Klimt & Design' if it's an agency overview slide, or 'Unknown' if unclear>",
  "industry": "<the SECTOR THE CLIENT'S OWN BUSINESS OPERATES IN — choose the single closest match from this list: {', '.join(CANONICAL_INDUSTRIES)}. Judge the client's business, NOT the kind of design work shown: a bank whose slide shows brand identity work is 'Finance & Wealth Management', not 'Branding & Design'. Use 'Creative & Marketing Agency' when the client is itself an agency or studio (design, branding, marketing, growth, social, content). Use 'General Agency' ONLY for Klimt & Design's own overview/capability slides, never for a client. Use 'Branding & Design' only when the client sells design tools or design products.>",
  "deliverable": "<what design work this slide shows, in a few words: e.g. 'website landing page redesign', 'full brand identity system', 'mobile app UI screens', 'pitch/investor deck', 'social media ad campaign', 'logo suite & stylescape'.>",
  "description": "<4-5 rich, concrete sentences covering, in this order: (1) what the client's COMPANY does — its product/service, who it serves, its niche; (2) what is physically ON THIS SLIDE — layout and sections, headline/tagline copy quoted verbatim if legible, any REAL client logos or brand names visible, any numbers/metrics/claims, specific UI or product elements, the photography or illustration style. Pack it with specifics; avoid generic filler like 'clean modern design'.>",
  "visual_style_raw": "<1-2 sentences: dominant colors (name them), typography style, mood, and overall aesthetic — enough to picture the look.>"
}}"""


def _slide_num_from_filename(filename: str) -> int:
    """Extract integer slide number from filename like 'slide-042.png' → 42."""
    nums = re.findall(r"\d+", filename)
    return int(nums[-1]) if nums else 0


def _resize_image_b64(path: Path) -> str:
    """Open image, resize to RESIZE_TO, return base64 string."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize(RESIZE_TO, Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def _analyze_slide(client_api: anthropic.Anthropic, slide_path: Path) -> dict:
    """Send one slide to Claude Vision and return extracted metadata dict."""
    slide_num = _slide_num_from_filename(slide_path.name)
    img_b64 = _resize_image_b64(slide_path)

    msg = client_api.messages.create(
        model=MODEL,
        max_tokens=900,  # richer 6-field description needs room; 400 truncated content
        system=_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"This is slide {slide_num} from the portfolio deck. Extract the metadata.",
                    },
                ],
            }
        ],
    )

    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()

    try:
        extracted = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback if Claude returns non-JSON
        extracted = {
            "client": "Unknown",
            "description": raw[:200],
            "visual_style_raw": "",
        }

    # Derive normalized tags
    service_type, service_category = get_service_tags(
        str(slide_num),
        extracted.get("description", ""),
        extracted.get("visual_style_raw", ""),
    )
    # Prefer the industry vision picked directly from the image; fall back to keyword
    # derivation only if it omitted the field or returned something off-list.
    vision_industry = str(extracted.get("industry", "")).strip()
    if vision_industry in CANONICAL_INDUSTRIES:
        industry = vision_industry
    else:
        industry = normalize_industry(
            f"{extracted.get('client', '')} {extracted.get('description', '')}"
        )
    visual_style = normalize_visual_style(extracted.get("visual_style_raw", ""))

    return {
        "slide_number": str(slide_num),
        "filename": slide_path.name,
        "client": extracted.get("client", "Unknown"),
        "industry": industry,
        "visual_style_raw": extracted.get("visual_style_raw", ""),
        "visual_style": visual_style,
        "service_type": service_type,
        "service_category": service_category,
        "deliverable": extracted.get("deliverable", ""),
        # `content` is the example blurb shown in the UI.
        "content": extracted.get("description", ""),
        # `text` is the matcher's full signal — every field, untruncated downstream.
        "text": " ".join(filter(None, [
            extracted.get("client", ""),
            extracted.get("deliverable", ""),
            extracted.get("description", ""),
            extracted.get("visual_style_raw", ""),
        ])),
    }


def _process_one(args) -> tuple[str, dict | None, str | None]:
    """Worker function for thread pool. Returns (filename, result, error)."""
    client_api, slide_path, retries = args
    for attempt in range(retries):
        try:
            result = _analyze_slide(client_api, slide_path)
            return slide_path.name, result, None
        except anthropic.RateLimitError:
            time.sleep(2 ** attempt)
        except Exception as e:
            if attempt == retries - 1:
                return slide_path.name, None, str(e)
            time.sleep(1)
    return slide_path.name, None, "Max retries exceeded"


def run_vision_ingest(slides_dir: str, resume: bool = True):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    slides_path = Path(slides_dir)
    all_pngs = sorted(slides_path.glob("*.png"), key=lambda p: _slide_num_from_filename(p.name))

    if not all_pngs:
        print(f"No PNG files found in {slides_dir}")
        sys.exit(1)

    # Load existing results if resuming
    existing: dict[str, dict] = {}
    if resume and Path(VISION_INDEX_PATH).exists():
        with open(VISION_INDEX_PATH) as f:
            for entry in json.load(f):
                existing[entry["filename"]] = entry
        print(f"  Resuming — {len(existing)} slides already processed, skipping.")

    to_process = [p for p in all_pngs if p.name not in existing]
    total = len(all_pngs)
    done = len(existing)
    errors = 0

    print(f"\n  Slides to process: {len(to_process)} of {total}")
    print(f"  Model: {MODEL} | Workers: {MAX_WORKERS} | Resize: {RESIZE_TO[0]}×{RESIZE_TO[1]}")
    print(f"  Estimated cost: ~${len(to_process) * 0.013:.2f}\n")

    client_api = anthropic.Anthropic(api_key=api_key)
    results = dict(existing)

    def _save():
        with open(VISION_INDEX_PATH, "w") as f:
            json.dump(list(results.values()), f, indent=2)

    work = [(client_api, p, 3) for p in to_process]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_process_one, w): w[1] for w in work}
        for future in as_completed(futures):
            filename, result, error = future.result()
            done += 1
            if result:
                results[filename] = result
                slide_num = result["slide_number"]
                print(
                    f"  [{done:>3}/{total}] ✓ slide-{int(slide_num):03d} "
                    f"| {result['client'][:25]:<25} "
                    f"| {result['service_type']:<28} "
                    f"| {result['content'][:55]}"
                )
            else:
                errors += 1
                print(f"  [{done:>3}/{total}] ✗ {filename} — {error}")

            # Save every 25 slides
            if done % 25 == 0:
                _save()

    _save()
    print(f"\n  ✓ Done. {done - errors}/{total} slides indexed → {VISION_INDEX_PATH}")
    if errors:
        print(f"  ✗ {errors} errors — re-run with --resume to retry.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision-index all slides via Claude")
    parser.add_argument("--slides", default="/Users/aravlohe/Downloads/slides_png")
    parser.add_argument("--no-resume", action="store_true", help="Start from scratch")
    args = parser.parse_args()
    run_vision_ingest(args.slides, resume=not args.no_resume)
