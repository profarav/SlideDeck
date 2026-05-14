"""
FastAPI backend

Endpoints:
  GET  /                  → UI
  POST /generate-deck     → case study finder (existing)
  POST /build-deck        → full ordered pitch deck
  POST /export-pdf        → compile slide PNGs → PDF download
  GET  /health
"""

import html as _html
import io
import re as _re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from deck_builder import build_full_deck
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


# ── Helpers ───────────────────────────────────────────────────────────────────

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/profarav/SlideDeck/main/slides_png"


def _fetch_url_text(url: str) -> str:
    """Fetch a URL and return cleaned plain text (max 3000 chars)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        raw = r.read().decode("utf-8", errors="ignore")
    raw = _re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", "", raw, flags=_re.DOTALL | _re.IGNORECASE)
    text = _re.sub(r"<[^>]+>", " ", raw)
    text = _html.unescape(text)
    text = _re.sub(r"\s+", " ", text).strip()
    return text[:3000]


def _image_url(slide: dict) -> str:
    filename = slide.get("filename", "")
    if filename:
        return f"{GITHUB_RAW_BASE}/{filename}"
    try:
        num = int(str(slide["slide_number"]).split("-")[0])
        return f"{GITHUB_RAW_BASE}/slide-{num:03d}.png"
    except (ValueError, KeyError):
        return ""


def _load_slide_image(slide_num_str: str) -> tuple[str, Image.Image | None]:
    """Load a PNG — local first, then GitHub raw."""
    try:
        num = int(str(slide_num_str).split("-")[0])
        local = _BASE / "slides_png" / f"slide-{num:03d}.png"
        if local.exists():
            return slide_num_str, Image.open(local).convert("RGB")
        url = f"{GITHUB_RAW_BASE}/slide-{num:03d}.png"
        with urllib.request.urlopen(url, timeout=15) as r:
            return slide_num_str, Image.open(io.BytesIO(r.read())).convert("RGB")
    except Exception:
        return slide_num_str, None


def _resolve_description(description: str, url: str) -> str:
    has_description = bool(description)
    has_url = bool(url)

    if has_url:
        try:
            url_text = _fetch_url_text(url)
        except Exception as e:
            if has_description:
                return description
            raise HTTPException(status_code=400, detail=f"Could not fetch URL: {e}")

        if not url_text:
            if has_description:
                return description
            raise HTTPException(status_code=400, detail="URL returned no usable text.")

        if has_description:
            return f"Prospect website context:\n{url_text}\n\nAdditional user guidance:\n{description}"
        return url_text

    if has_description:
        return description

    raise HTTPException(status_code=400, detail="Provide either a description or a URL.")


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


def _to_slide_result(s: dict) -> SlideResult:
    return SlideResult(
        slide_number=s["slide_number"],
        client=s.get("client", "Klimt & Design"),
        industry=s.get("industry", ""),
        visual_style=s.get("visual_style", ""),
        service_type=s.get("service_type", ""),
        service_category=s.get("service_category", ""),
        content=s.get("content", ""),
        score=s.get("score", 1.0),
        image_url=_image_url(s),
        why=s.get("why", ""),
    )


class DeckResponse(BaseModel):
    persona: dict
    slides: list[SlideResult]


class DeckSection(BaseModel):
    label: str
    slides: list[SlideResult]


class FullDeckResponse(BaseModel):
    persona: dict
    sections: list[DeckSection]
    total_slides: int


class ExportRequest(BaseModel):
    slide_numbers: list[str]
    filename: str = "klimt-pitch-deck"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/generate-deck", response_model=DeckResponse)
def generate_deck(req: ProspectRequest):
    description = _resolve_description(req.description.strip(), req.url.strip())
    try:
        persona = build_search_persona(description)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiler error: {e}")
    try:
        slides = retrieve_case_studies(persona, n_results=req.n_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retriever error: {e}")
    return DeckResponse(persona=persona, slides=[_to_slide_result(s) for s in slides])


@app.post("/build-deck", response_model=FullDeckResponse)
def build_deck_endpoint(req: ProspectRequest):
    description = _resolve_description(req.description.strip(), req.url.strip())
    try:
        persona = build_search_persona(description)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiler error: {e}")
    try:
        deck = build_full_deck(persona, n_case_studies=req.n_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deck builder error: {e}")

    sections = [
        DeckSection(
            label=sec["label"],
            slides=[_to_slide_result(s) for s in sec["slides"]],
        )
        for sec in deck["sections"]
    ]
    return FullDeckResponse(
        persona=persona,
        sections=sections,
        total_slides=deck["total_slides"],
    )


@app.post("/export-pdf")
def export_pdf(req: ExportRequest):
    if not req.slide_numbers:
        raise HTTPException(status_code=400, detail="No slides specified.")

    # Fetch all images in parallel (handles both local and GitHub raw)
    images_by_num: dict[str, Image.Image] = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_load_slide_image, n): n for n in req.slide_numbers}
        for future in as_completed(futures):
            num, img = future.result()
            if img:
                images_by_num[num] = img

    # Preserve requested order, skip any that failed to load
    ordered = [images_by_num[n] for n in req.slide_numbers if n in images_by_num]

    if not ordered:
        raise HTTPException(status_code=500, detail="Could not load any slide images.")

    buf = io.BytesIO()
    ordered[0].save(
        buf,
        format="PDF",
        save_all=True,
        append_images=ordered[1:],
        resolution=150,
    )
    buf.seek(0)

    safe_name = _re.sub(r"[^\w\-]", "-", req.filename.strip()) or "klimt-pitch-deck"
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.pdf"'},
    )


@app.get("/health")
def health():
    return {"status": "ok"}
