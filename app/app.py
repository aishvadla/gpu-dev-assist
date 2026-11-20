"""FastAPI service for GPU DevAssist."""

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from gpu_dev_assist.generate import AnswerGenerator
from gpu_dev_assist.hybrid_retrieve import build_retrievers


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    retriever: Literal["dense", "bm25", "hybrid"] = "hybrid"


class Source(BaseModel):
    id: int
    source: str
    page: int


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    retriever: str
    latency_ms: int
    input_tokens: int
    output_tokens: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the vector store, embedding model and BM25 index once at startup
    retrievers = build_retrievers()
    app.state.generators = {name: AnswerGenerator(r) for name, r in retrievers.items()}
    yield


app = FastAPI(title="GPU DevAssist", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

@app.get("/")
def root():
    """Serve the web interface."""
    return FileResponse(Path(__file__).parent / "templates" / "index.html")


@app.get("/health")
def health():
    """Service health check."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request):
    """Answer a GPU programming question with page citations."""
    start = time.perf_counter()
    result = request.app.state.generators[req.retriever].answer(req.question)
    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        retriever=req.retriever,
        latency_ms=int((time.perf_counter() - start) * 1000),
        **result["usage"],
    )