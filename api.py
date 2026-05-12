"""
FastAPI backend — POST /generate-deck

Start with:
    uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from profiler import build_search_persona
from retriever import retrieve_case_studies

app = FastAPI(title="Klimt Slide Retrieval Engine", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


class ProspectRequest(BaseModel):
    description: str
    n_results: int = 8


class SlideResult(BaseModel):
    slide_number: str
    client: str
    industry: str
    visual_style: str
    content: str
    score: float


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
        slides=[SlideResult(**s) for s in slides],
    )


@app.get("/health")
def health():
    return {"status": "ok"}
