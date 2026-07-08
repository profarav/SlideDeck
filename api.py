"""
FastAPI backend

Endpoints:
  GET  /                  → UI
  POST /generate-deck     → case study finder (existing)
  POST /build-deck        → full ordered pitch deck
  POST /refine-slides     → controlled refinement (pin/exclude/variant)
  POST /refine-full-deck  → full deck refinement (case studies only)
  POST /export-pdf        → compile slide PNGs → PDF download
  GET  /health
"""

import hashlib
import html as _html
import io
import re as _re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from deck_builder import build_full_deck, refine_full_deck
from profiler import build_search_persona
from retriever import retrieve_case_studies, retrieve_case_studies_with_controls, retrieve_examples

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


# ── Persona cache ───────────────────────────────────────────────────────────────
# The profiler (build_search_persona) is a nondeterministic LLM call. Without caching,
# every refine re-derives a fresh persona, so the ranking a user is pinning/excluding
# against silently shifts under them. Cache one persona per (description, url) input so
# the initial generate and its refines share it. Keyed on the RAW inputs (not the
# URL-resolved text, which can vary between fetches).

_PERSONA_TTL = 3600          # seconds
_PERSONA_MAX = 256           # entries
_persona_cache: dict[str, tuple[float, dict]] = {}
_persona_lock = threading.Lock()


def _persona_cache_key(description: str, url: str) -> str:
    return hashlib.sha256(f"{description}\x00{url}".encode("utf-8")).hexdigest()[:16]


def _persona_for_request(description: str, url: str, persona_key: str = "") -> tuple[dict, str]:
    """
    Return (persona, key), reusing a cached persona when possible. A caller-supplied
    persona_key forces reuse of that exact persona (e.g. across refine calls); otherwise
    the (description, url) inputs are hashed. On a hit, neither the URL fetch nor the
    profiler runs. On a miss, resolve + profile, then cache.
    """
    now = time.time()
    with _persona_lock:
        for k in [k for k, (ts, _) in _persona_cache.items() if now - ts > _PERSONA_TTL]:
            del _persona_cache[k]
        for k in (persona_key, _persona_cache_key(description, url)):
            if k and k in _persona_cache:
                _, persona = _persona_cache[k]
                _persona_cache[k] = (now, persona)   # refresh recency
                return persona, k

    resolved = _resolve_description(description, url)
    try:
        persona = build_search_persona(resolved)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiler error: {e}")

    key = _persona_cache_key(description, url)
    with _persona_lock:
        if len(_persona_cache) >= _PERSONA_MAX:
            del _persona_cache[min(_persona_cache, key=lambda k: _persona_cache[k][0])]
        _persona_cache[key] = (time.time(), persona)
    return persona, key


# ── Models ────────────────────────────────────────────────────────────────────

class ProspectRequest(BaseModel):
    description: str = ""
    url: str = ""
    n_results: int = 8


class RefineRequest(ProspectRequest):
    pinned: list[str] = Field(default_factory=list)
    excluded: list[str] = Field(default_factory=list)
    previous_visible: list[str] = Field(default_factory=list)
    variant: str = ""
    persona_key: str = ""   # reuse the exact persona from the initial generate call


class ExistingSectionRequest(BaseModel):
    label: str
    slides: list[dict]


class RefineFullDeckRequest(RefineRequest):
    existing_sections: list[ExistingSectionRequest] = Field(default_factory=list)


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


class ExampleResult(BaseModel):
    client: str
    industry: str
    visual_style: str
    service_type: str
    service_category: str
    slide_range: list[int]
    n_slides: int
    score: float
    image_url: str
    why: str
    representative_slide: str


class ExamplesResponse(BaseModel):
    persona: dict
    persona_key: str = ""
    examples: list[ExampleResult]


def _to_example_result(e: dict) -> ExampleResult:
    rep_slide = {"slide_number": e["representative_slide"], "filename": ""}
    for s in e.get("_slides", []):
        if str(s["slide_number"]) == str(e["representative_slide"]):
            rep_slide = s
            break
    return ExampleResult(
        client=e.get("client", "Unknown"),
        industry=e.get("industry", ""),
        visual_style=e.get("visual_style", ""),
        service_type=e.get("service_type", ""),
        service_category=e.get("service_category", ""),
        slide_range=e.get("slide_range", [0, 0]),
        n_slides=e.get("n_slides", 1),
        score=e.get("score", 0.0),
        image_url=_image_url(rep_slide),
        why=e.get("why", ""),
        representative_slide=str(e.get("representative_slide", "")),
    )


class DeckResponse(BaseModel):
    persona: dict
    persona_key: str = ""
    slides: list[SlideResult]


class DeckSection(BaseModel):
    label: str
    slides: list[SlideResult]


class FullDeckResponse(BaseModel):
    persona: dict
    persona_key: str = ""
    sections: list[DeckSection]
    total_slides: int


class ExportRequest(BaseModel):
    slide_numbers: list[str]
    filename: str = "klimt-pitch-deck"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/generate-deck", response_model=ExamplesResponse)
def generate_deck(req: ProspectRequest):
    persona, key = _persona_for_request(req.description.strip(), req.url.strip())
    try:
        examples = retrieve_examples(persona, n_results=req.n_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retriever error: {e}")
    return ExamplesResponse(
        persona=persona, persona_key=key, examples=[_to_example_result(e) for e in examples]
    )


@app.post("/build-deck", response_model=FullDeckResponse)
def build_deck_endpoint(req: ProspectRequest):
    persona, key = _persona_for_request(req.description.strip(), req.url.strip())
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
        persona_key=key,
        sections=sections,
        total_slides=deck["total_slides"],
    )


@app.post("/refine-slides", response_model=DeckResponse)
def refine_slides(req: RefineRequest):
    persona, key = _persona_for_request(req.description.strip(), req.url.strip(), req.persona_key)
    try:
        slides = retrieve_case_studies_with_controls(
            persona,
            n_results=req.n_results,
            pinned=req.pinned,
            excluded=req.excluded,
            previous_visible=req.previous_visible,
            variant=req.variant.strip(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retriever error: {e}")
    return DeckResponse(persona=persona, persona_key=key, slides=[_to_slide_result(s) for s in slides])


@app.post("/refine-full-deck", response_model=FullDeckResponse)
def refine_full_deck_endpoint(req: RefineFullDeckRequest):
    persona, key = _persona_for_request(req.description.strip(), req.url.strip(), req.persona_key)

    existing_sections = [sec.model_dump() for sec in req.existing_sections]
    try:
        deck = refine_full_deck(
            persona,
            existing_sections=existing_sections,
            n_case_studies=req.n_results,
            pinned=req.pinned,
            excluded=req.excluded,
            previous_visible=req.previous_visible,
            variant=req.variant.strip(),
        )
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
        persona_key=key,
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
