"""
FastAPI backend — POST /generate-deck

Start with:
    uvicorn api:app --reload --port 8000
"""

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

# Only mount static if directory exists (it won't be present in Vercel serverless)
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
def serve_ui():
    if _INDEX_HTML.exists():
        return FileResponse(str(_INDEX_HTML))
    return HTMLResponse("<h1>Klimt Slide Engine</h1><p>UI not found.</p>")


class ProspectRequest(BaseModel):
    description: str
    n_results: int = 8


GITHUB_RAW_BASE = "https://raw.githubusercontent.com/profarav/SlideDeck/main/slides_png"


def _image_url(slide: dict) -> str:
    filename = slide.get("filename", "")
    if filename:
        return f"{GITHUB_RAW_BASE}/{filename}"
    # Fallback: construct from slide_number
    try:
        num = int(str(slide["slide_number"]).split("-")[0])
        return f"{GITHUB_RAW_BASE}/slide-{num:03d}.png"
    except (ValueError, KeyError):
        return ""


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


class DeckResponse(BaseModel):
    persona: dict
    slides: list[SlideResult]


@app.post("/generate-deck", response_model=DeckResponse)
def generate_deck(req: ProspectRequest):
    if not req.description.strip():
        raise HTTPException(status_code=400, detail="description cannot be empty")

    try:
        persona = build_search_persona(req.description)
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
