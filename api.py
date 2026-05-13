"""
FastAPI backend — POST /generate-deck

Start with:
    uvicorn api:app --reload --port 8000
"""

import html as _html
import re as _re
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from profiler import build_search_persona
from retriever import retrieve_case_studies

_BASE = Path(__file__).parent
_STATIC_DIR = _BASE / "static"
_INDEX_HTML = _STATIC_DIR / "index.html"

app = FastAPI(title="Klimt Slide Retrieval Engine", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
def serve_ui():
    if _INDEX_HTML.exists():
        return FileResponse(str(_INDEX_HTML))
    return HTMLResponse("<h1>Klimt Slide Engine</h1><p>UI not found.</p>")


# ── URL scraping ──────────────────────────────────────────────────────────────

def _fetch_url_text(url: str) -> str:
    """Fetch a URL and return cleaned plain text (max 3000 chars for profiler)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        raw = r.read().decode("utf-8", errors="ignore")
    # Strip <script> and <style> blocks
    raw = _re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", "", raw, flags=_re.DOTALL | _re.IGNORECASE)
    # Strip remaining HTML tags
    text = _re.sub(r"<[^>]+>", " ", raw)
    # Decode HTML entities and collapse whitespace
    text = _html.unescape(text)
    text = _re.sub(r"\s+", " ", text).strip()
    return text[:3000]


# ── Image URL helper ──────────────────────────────────────────────────────────

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/profarav/SlideDeck/main/slides_png"


def _image_url(slide: dict) -> str:
    filename = slide.get("filename", "")
    if filename:
        return f"{GITHUB_RAW_BASE}/{filename}"
    try:
        num = int(str(slide["slide_number"]).split("-")[0])
        return f"{GITHUB_RAW_BASE}/slide-{num:03d}.png"
    except (ValueError, KeyError):
        return ""


# ── Models ────────────────────────────────────────────────────────────────────

class ProspectRequest(BaseModel):
    description: str = ""
    url: str = ""
    n_results: int = 8


class SlideResult(BaseModel):
    slide_number: str
    client: str
    industry: str
    visual_style: str
    service_type: str
    service_category: str
    content: str
    score: float
    image_url: str
    why: str = ""


class DeckResponse(BaseModel):
    persona: dict
    slides: list[SlideResult]


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post("/generate-deck", response_model=DeckResponse)
def generate_deck(req: ProspectRequest):
    description = req.description.strip()
    url = req.url.strip()

    if not description and not url:
        raise HTTPException(status_code=400, detail="Provide either a description or a URL.")

    # If URL given, scrape it and use as description
    if url and not description:
        try:
            description = _fetch_url_text(url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not fetch URL: {e}")
        if not description:
            raise HTTPException(status_code=400, detail="URL returned no usable text.")

    try:
        persona = build_search_persona(description)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiler error: {e}")

    try:
        slides = retrieve_case_studies(persona, n_results=req.n_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retriever error: {e}")

    return DeckResponse(
        persona=persona,
        slides=[SlideResult(**s, image_url=_image_url(s)) for s in slides],
    )


@app.get("/health")
def health():
    return {"status": "ok"}
