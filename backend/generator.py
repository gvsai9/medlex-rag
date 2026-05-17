"""
RAG Generator — Ollama
======================
Takes retrieved chunks and generates a cited answer using local Ollama.
"""

from typing import List, Optional
from dataclasses import dataclass
from loguru import logger
import httpx

from retrieval.retriever import RetrievedChunk, get_retriever
from config import get_settings

settings = get_settings()


@dataclass
class Citation:
    doc_title: str
    source_url: str
    page_num: int
    section_title: str
    chunk_id: str
    relevance_score: float


@dataclass
class RAGAnswer:
    question: str
    answer: str
    citations: List[Citation]
    retrieval_method: str
    latency_ms: float
    model: str


SYSTEM_PROMPT = """You are MedLex, an expert assistant for public government documents
(FDA drug labels, court judgments, research papers, SEC filings).

Your job:
1. Answer questions ONLY based on the provided document context.
2. Be precise, factual, and cite specific parts of the context.
3. If context doesn't contain enough information, say so clearly.
4. Never hallucinate.
5. For medical/legal topics, recommend consulting professionals.

Format:
- Direct answer
- Key details/caveats
- Mention citation numbers like [1], [2] when using evidence
"""


def build_context(chunks: List[RetrievedChunk]) -> str:
    """Build context string from retrieved chunks with reference numbers."""
    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[{i}] From: {chunk.doc_title} (Page {chunk.page_num})\n"
            f"Section: {chunk.section_title}\n"
            f"Content: {chunk.content}\n"
        )

    return "\n---\n".join(context_parts)


async def _call_ollama(prompt: str, system: str) -> str:
    """
    Call local Ollama chat API.
    Make sure Ollama is running and model is pulled.
    """
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(
            f"{settings.ollama_base_url}/api/chat",
            json=payload,
        )

        response.raise_for_status()
        data = response.json()

        return data["message"]["content"]


async def answer_question(
    question: str,
    domain_filter: Optional[str] = None,
    top_k: int = None,
) -> RAGAnswer:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks
    2. Build context
    3. Generate answer with Ollama
    4. Return answer + citations
    """
    import time

    start = time.time()

    retriever = get_retriever()

    chunks = await retriever.retrieve(
        question,
        top_k=top_k or settings.top_k_retrieval,
        domain_filter=domain_filter,
    )

    if not chunks:
        return RAGAnswer(
            question=question,
            answer=(
                "I couldn't find relevant information in the indexed documents. "
                "Please try rephrasing your question or ingest documents first."
            ),
            citations=[],
            retrieval_method="hybrid",
            latency_ms=(time.time() - start) * 1000,
            model=settings.ollama_model,
        )

    context = build_context(chunks)

    prompt = f"""Context Documents:
{context}

Question: {question}

Answer based ONLY on the above context.
Use citation numbers like [1], [2] where relevant.
If the context is insufficient, say that clearly.
"""

    if settings.llm_provider == "ollama":
        try:
            answer_text = await _call_ollama(prompt, SYSTEM_PROMPT)
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            answer_text = _fallback_answer(question, chunks)
    else:
        answer_text = _fallback_answer(question, chunks)

    citations = [
        Citation(
            doc_title=c.doc_title,
            source_url=c.source_url,
            page_num=c.page_num,
            section_title=c.section_title,
            chunk_id=c.chunk_id,
            relevance_score=round(c.final_score, 4),
        )
        for c in chunks[:settings.top_k_rerank]
    ]

    latency = (time.time() - start) * 1000

    logger.info(
        f"Q: '{question[:50]}...' → {len(chunks)} chunks, {latency:.0f}ms"
    )

    return RAGAnswer(
        question=question,
        answer=answer_text,
        citations=citations,
        retrieval_method="hybrid",
        latency_ms=round(latency, 1),
        model=settings.ollama_model,
    )


def _fallback_answer(question: str, chunks: List[RetrievedChunk]) -> str:
    """Extractive fallback when LLM is unavailable."""
    top = chunks[0]

    return (
        f"Based on '{top.doc_title}' (Page {top.page_num}):\n\n"
        f"{top.content[:700]}...\n\n"
        f"[Note: Full LLM synthesis unavailable. "
        f"Make sure Ollama is running and model '{settings.ollama_model}' is pulled.]"
    )