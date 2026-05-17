"""
FastAPI API for MedLex RAG
==========================
Endpoints:
- GET  /
- GET  /health
- GET  /stats
- POST /ask
- POST /ask/stream
"""

import json
from typing import List, Optional, Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

from retrieval.generator import answer_question, answer_question_stream
from state_tracker import get_tracker
from ingestion.graph_store import get_graph_store


app = FastAPI(
    title="MedLex RAG API",
    description="FDA drug-label RAG API using Pinecone, Neo4j, MySQL, and Ollama",
    version="1.0.0",
)


# ------------------------------------------------------------
# CORS for simple-ui
# ------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "*",  # okay for local testing; restrict later before deployment
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    drug: Optional[str] = "ibuprofen"
    domain: Optional[str] = "fda"
    top_k: Optional[int] = 3


class Citation(BaseModel):
    id: Optional[int] = None
    doc_title: Optional[str] = None
    drug: Optional[str] = None
    page_num: Optional[Any] = None
    section_title: Optional[str] = None
    score: Optional[Any] = None


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation] = []
    model: Optional[str] = None
    latency_ms: Optional[float] = None


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "message": "MedLex RAG API is running",
        "docs": "/docs",
        "health": "/health",
        "ask": "/ask",
        "stream": "/ask/stream",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "medlex-rag-api",
    }


@app.get("/stats")
async def stats():
    try:
        tracker = await get_tracker()
        db_stats = await tracker.stats()

        graph = await get_graph_store()
        graph_stats = await graph.stats()

        return {
            "status": "ok",
            "mysql": db_stats,
            "neo4j": graph_stats,
        }

    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    try:
        result = await answer_question(
            question=req.question,
            drug=req.drug,
            domain=req.domain or "fda",
            top_k=req.top_k,
        )

        return AskResponse(
            answer=result.get("answer", ""),
            citations=result.get("citations", []),
            model=result.get("model"),
            latency_ms=result.get("latency_ms"),
        )

    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    async def event_generator():
        try:
            async for event in answer_question_stream(
                question=req.question,
                drug=req.drug,
                domain=req.domain or "fda",
                top_k=req.top_k,
            ):
                yield json.dumps(event, ensure_ascii=False) + "\n"

        except Exception as e:
            logger.exception(e)
            yield json.dumps(
                {
                    "type": "error",
                    "message": str(e),
                },
                ensure_ascii=False,
            ) + "\n"

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
    )


@app.on_event("shutdown")
async def shutdown_event():
    # Optional cleanup if your stores have close methods.
    pass