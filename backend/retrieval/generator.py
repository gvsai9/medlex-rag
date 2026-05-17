"""
Answer Generator
================
MedLex RAG answer generator.

Rules:
- Basic greetings / identity questions are answered using Gemini with an app-identity system prompt.
- Out-of-domain questions are blocked without calling Gemini.
- FDA/drug-label/document questions use RAG retrieval.
- RAG answers must use retrieved context only.
- If context does not contain the answer, say not found.
- Supports normal and streaming responses.
"""

import re
import time
from typing import List, Dict, Optional, AsyncGenerator, Any, Tuple

from loguru import logger

from config import get_settings
from retrieval.retriever import get_retriever
from llm_client import call_llm_chat, stream_llm_chat

settings = get_settings()


# ============================================================
# Constants
# ============================================================

NOT_FOUND_ANSWER = "I could not find this information in the provided documents."

OUT_OF_DOMAIN_ANSWER = (
    "I can only answer questions related to the ingested FDA drug-label documents. "
    "Please ask a question about a drug label, warnings, dosage, usage, side effects, "
    "directions, contraindications, ingredients, storage, or related document content."
)


# ============================================================
# System prompts
# ============================================================

APP_IDENTITY_SYSTEM_PROMPT = """
You are MedLex RAG, a document-based FDA drug-label question answering assistant.

You may answer basic questions about yourself, your purpose, who built you, and how the app works.
Keep answers short, professional, and clear.

Important identity:
- You are MedLex RAG.
- You were built by Venkata Sai as a RAG-based FDA drug-label assistant.
- Your pipeline uses Pinecone for vector retrieval, Neo4j for graph retrieval, MySQL/BM25 for stored chunks, and Gemini for answer generation.

Safety rules:
- Do not answer medical, drug, dosage, warning, treatment, or side-effect questions from your own knowledge.
- If the user asks a medical/document question, tell them to ask about an ingested FDA drug-label document.
- Do not pretend to have access to documents unless retrieved context is provided.
"""

RAG_SYSTEM_PROMPT = """
You are MedLex RAG, an FDA drug-label document question answering assistant.

Critical rules:
1. Answer ONLY using the provided retrieved context.
2. Do NOT use your own medical knowledge.
3. Do NOT guess.
4. If the answer is not present in the context, say exactly:
   "I could not find this information in the provided documents."
5. Use clear, simple language.
6. Include citations using citation numbers like [1], [2].
7. Do not cite anything that is not in the provided context.
8. Do not provide personal medical advice.
9. If the user asks for medical decisions, explain only what the document says and suggest consulting a healthcare professional.

Answer format:
Direct answer:
...

Key details:
- ...

Citations:
[1], [2]
"""


# ============================================================
# Config helpers
# ============================================================

def get_llm_temperature() -> float:
    return float(getattr(settings, "llm_temperature", 0.1))


def get_llm_smalltalk_temperature() -> float:
    return float(getattr(settings, "llm_smalltalk_temperature", 0.3))


def get_model_name() -> str:
    provider = str(getattr(settings, "llm_provider", "gemini")).lower()

    if provider == "gemini":
        return str(getattr(settings, "gemini_model", "gemini-2.5-flash"))

    return provider


# ============================================================
# Guardrails
# ============================================================

def normalize_question(question: str) -> str:
    return (question or "").strip().lower()


def is_basic_greeting(question: str) -> bool:
    q = normalize_question(question)

    greeting_patterns = [
        r"^hi$",
        r"^hello$",
        r"^hey$",
        r"^hii+$",
        r"^good morning$",
        r"^good afternoon$",
        r"^good evening$",
        r"^namaste$",
        r"^thanks$",
        r"^thank you$",
        r"^ok$",
        r"^okay$",
    ]

    return any(re.match(pattern, q) for pattern in greeting_patterns)


def is_basic_identity_question(question: str) -> bool:
    q = normalize_question(question)

    identity_phrases = [
        "who are you",
        "what are you",
        "who built you",
        "who made you",
        "who created you",
        "what can you do",
        "what is medlex",
        "what is medlex rag",
        "how do you work",
        "what is this app",
        "what is this project",
        "explain yourself",
        "tell me about yourself",
    ]

    return any(phrase in q for phrase in identity_phrases)


def is_basic_allowed_question(question: str) -> bool:
    return is_basic_greeting(question) or is_basic_identity_question(question)


def is_medical_document_domain_question(question: str) -> bool:
    """
    Allows FDA/drug-label/document questions.
    Blocks general world knowledge, programming, sports, politics, movies, etc.
    """
    q = normalize_question(question)

    domain_keywords = [
        # Drug/document words
        "drug", "tablet", "capsule", "medicine", "medication", "label", "fda",
        "prescription", "nonprescription", "otc", "active ingredient",
        "inactive ingredient", "purpose", "uses", "use", "dosage", "dose",
        "directions", "warning", "warnings", "allergy", "side effect",
        "side effects", "contraindication", "contraindications", "do not use",
        "ask a doctor", "ask doctor", "ask a doctor before use",
        "stop use", "pregnant", "pregnancy", "breastfeeding", "children",
        "adult", "liver", "kidney", "heart", "stomach", "bleeding",
        "overdose", "storage", "temperature", "symptoms", "pain reliever",
        "fever reducer", "nsaid", "aspirin", "ibuprofen", "acetaminophen",
        "paracetamol", "naproxen", "metformin", "lisinopril", "atorvastatin",
        "omeprazole", "amoxicillin", "levothyroxine",

        # FDA label section words
        "indications", "usage", "adverse reactions", "drug interactions",
        "clinical pharmacology", "boxed warning", "contraindicated",
        "precautions", "description", "how supplied",

        # General document QA words
        "according to document", "according to the document",
        "in the document", "from the document", "based on the label",
        "what does the label say", "what does document say",
    ]

    return any(keyword in q for keyword in domain_keywords)


def is_obviously_out_of_domain(question: str) -> bool:
    q = normalize_question(question)

    out_keywords = [
        # Programming / tech unrelated
        "python programming", "java", "javascript", "react", "fastapi",
        "html", "css", "sql query", "code", "algorithm", "debug this",

        # Sports / celebrities / general GK
        "virat kohli", "cricket", "football", "movie", "actor", "actress",
        "prime minister", "president", "capital of", "history of",
        "weather", "stock price", "bitcoin", "ipl", "world cup",

        # Random tasks
        "write an essay", "write a poem", "make a story",
        "translate", "summarize this paragraph",
    ]

    return any(keyword in q for keyword in out_keywords)


def should_use_rag(question: str) -> bool:
    if is_obviously_out_of_domain(question):
        return False

    if is_medical_document_domain_question(question):
        return True

    return False


def make_basic_result(answer: str, model: str = "guardrail", latency_ms: float = 0) -> Dict[str, Any]:
    return {
        "answer": answer,
        "citations": [],
        "model": model,
        "latency_ms": latency_ms,
    }


# ============================================================
# Chunk helpers
# ============================================================

def chunk_to_dict(chunk: Any) -> Dict[str, Any]:
    if isinstance(chunk, dict):
        return chunk

    data = {}

    for attr in [
        "chunk_id", "doc_id", "doc_title", "title", "drug", "drug_name",
        "page_num", "page", "section_title", "section", "text", "content",
        "score", "metadata"
    ]:
        if hasattr(chunk, attr):
            data[attr] = getattr(chunk, attr)

    metadata = getattr(chunk, "metadata", None)
    if isinstance(metadata, dict):
        data["metadata"] = metadata

    return data


def get_chunk_metadata(chunk: Any) -> Dict[str, Any]:
    d = chunk_to_dict(chunk)
    meta = d.get("metadata", {})

    if not isinstance(meta, dict):
        meta = {}

    merged = {}
    merged.update(meta)
    merged.update({k: v for k, v in d.items() if k != "metadata"})

    return merged


def get_chunk_text(chunk: Any) -> str:
    meta = get_chunk_metadata(chunk)

    for key in ["text", "content", "chunk_text", "page_content"]:
        value = meta.get(key)
        if value:
            return str(value)

    if hasattr(chunk, "text"):
        return str(getattr(chunk, "text"))

    if hasattr(chunk, "content"):
        return str(getattr(chunk, "content"))

    if hasattr(chunk, "page_content"):
        return str(getattr(chunk, "page_content"))

    return ""


def get_chunk_score(chunk: Any) -> float:
    meta = get_chunk_metadata(chunk)

    try:
        return float(meta.get("score", 0.0) or 0.0)
    except Exception:
        return 0.0


def chunks_match_requested_drug(chunks: List[Any], drug: Optional[str]) -> bool:
    """
    If user selected a drug in UI, ensure retrieved chunks are about that drug.
    Prevents answering metformin question using ibuprofen chunks.
    """
    if not drug:
        return True

    requested = drug.strip().lower()
    if not requested:
        return True

    for chunk in chunks:
        meta = get_chunk_metadata(chunk)
        text = get_chunk_text(chunk).lower()

        chunk_drug = str(meta.get("drug", "") or meta.get("drug_name", "")).lower()
        doc_title = str(meta.get("doc_title", "") or meta.get("title", "")).lower()

        if requested in chunk_drug:
            return True

        if requested in doc_title:
            return True

        if requested in text[:1000]:
            return True

    return False


def build_context_and_citations(chunks: List[Any]) -> Tuple[str, List[Dict[str, Any]]]:
    context_blocks = []
    citations = []

    for idx, chunk in enumerate(chunks, start=1):
        meta = get_chunk_metadata(chunk)
        text = get_chunk_text(chunk).strip()

        if not text:
            continue

        doc_title = (
            meta.get("doc_title")
            or meta.get("title")
            or meta.get("document_title")
            or "Unknown document"
        )

        drug = meta.get("drug") or meta.get("drug_name") or ""
        page_num = meta.get("page_num") or meta.get("page") or ""
        section_title = meta.get("section_title") or meta.get("section") or ""
        score = meta.get("score", get_chunk_score(chunk))

        citation = {
            "id": idx,
            "doc_title": str(doc_title),
            "drug": str(drug),
            "page_num": page_num,
            "section_title": str(section_title),
            "score": score,
        }

        citations.append(citation)

        context_blocks.append(
            f"[{idx}] Document: {doc_title}\n"
            f"Drug: {drug}\n"
            f"Page: {page_num}\n"
            f"Section: {section_title}\n"
            f"Text:\n{text}\n"
        )

    return "\n\n---\n\n".join(context_blocks), citations


# ============================================================
# Retrieval wrapper
# ============================================================

async def retrieve_chunks(
    question: str,
    drug: Optional[str] = None,
    domain: str = "fda",
    top_k: Optional[int] = None,
) -> List[Any]:
    retriever = await get_retriever()

    k = top_k or int(getattr(settings, "top_k_rerank", 3))

    try:
        return await retriever.retrieve(
            query=question,
            drug=drug,
            domain=domain,
            top_k=k,
        )
    except TypeError:
        pass

    try:
        return await retriever.retrieve(
            question=question,
            drug=drug,
            domain=domain,
            top_k=k,
        )
    except TypeError:
        pass

    try:
        return await retriever.retrieve(question, top_k=k)
    except TypeError:
        pass

    return await retriever.retrieve(question)


# ============================================================
# Prompt builders
# ============================================================

def build_rag_user_prompt(question: str, context_text: str) -> str:
    return f"""
Question:
{question}

Retrieved context:
{context_text}

Instructions:
Answer using only the retrieved context.
Use citation numbers exactly as shown in the context, for example [1] or [2].
If the retrieved context does not contain the answer, say:
"{NOT_FOUND_ANSWER}"
"""


# ============================================================
# Public API: normal answer
# ============================================================

async def answer_question(
    question: str,
    drug: Optional[str] = None,
    domain: str = "fda",
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    start = time.time()

    question = (question or "").strip()

    if not question:
        return make_basic_result("Please enter a question.")

    # ------------------------------------------------------------
    # Guardrail 1: basic greetings / identity questions
    # ------------------------------------------------------------
    if is_basic_allowed_question(question):
        try:
            answer = await call_llm_chat(
                system_prompt=APP_IDENTITY_SYSTEM_PROMPT,
                user_prompt=question,
                temperature=get_llm_smalltalk_temperature(),
            )

            latency_ms = round((time.time() - start) * 1000, 2)

            return {
                "answer": answer,
                "citations": [],
                "model": get_model_name(),
                "latency_ms": latency_ms,
            }

        except Exception as e:
            logger.exception(f"Small-talk LLM call failed: {e}")
            return make_basic_result(
                "Hello! I am MedLex RAG. Ask me a question about an ingested FDA drug-label document.",
                model="guardrail-fallback",
                latency_ms=round((time.time() - start) * 1000, 2),
            )

    # ------------------------------------------------------------
    # Guardrail 2: out-of-domain block
    # ------------------------------------------------------------
    if not should_use_rag(question):
        return make_basic_result(OUT_OF_DOMAIN_ANSWER)

    # ------------------------------------------------------------
    # RAG retrieval
    # ------------------------------------------------------------
    try:
        chunks = await retrieve_chunks(
            question=question,
            drug=drug,
            domain=domain,
            top_k=top_k,
        )
    except Exception as e:
        logger.exception(f"Retrieval failed: {e}")
        return make_basic_result(
            "Retrieval failed. Please check Pinecone, Neo4j, MySQL, and retriever configuration.",
            model="retrieval-error",
            latency_ms=round((time.time() - start) * 1000, 2),
        )

    if not chunks:
        return make_basic_result(NOT_FOUND_ANSWER)

    if not chunks_match_requested_drug(chunks, drug):
        return make_basic_result(NOT_FOUND_ANSWER)

    context_text, citations = build_context_and_citations(chunks)

    if not context_text.strip():
        return make_basic_result(NOT_FOUND_ANSWER)

    # ------------------------------------------------------------
    # RAG generation
    # ------------------------------------------------------------
    user_prompt = build_rag_user_prompt(question, context_text)

    try:
        answer = await call_llm_chat(
            system_prompt=RAG_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=get_llm_temperature(),
        )
    except Exception as e:
        logger.exception(f"LLM generation failed: {e}")
        return make_basic_result(
            "LLM generation failed. Please check Gemini API key, model name, and network connection.",
            model="llm-error",
            latency_ms=round((time.time() - start) * 1000, 2),
        )

    if not answer:
        answer = NOT_FOUND_ANSWER

    latency_ms = round((time.time() - start) * 1000, 2)

    logger.info(f"Q: '{question[:60]}...' → {len(chunks)} chunks, {latency_ms}ms")

    return {
        "answer": answer,
        "citations": citations,
        "model": get_model_name(),
        "latency_ms": latency_ms,
    }


# ============================================================
# Public API: streaming answer
# ============================================================

async def answer_question_stream(
    question: str,
    drug: Optional[str] = None,
    domain: str = "fda",
    top_k: Optional[int] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    start = time.time()

    question = (question or "").strip()

    if not question:
        yield {"type": "token", "content": "Please enter a question."}
        yield {"type": "done", "model": "guardrail", "latency_ms": 0}
        return

    # ------------------------------------------------------------
    # Guardrail 1: basic greetings / identity questions
    # ------------------------------------------------------------
    if is_basic_allowed_question(question):
        try:
            answer = await call_llm_chat(
                system_prompt=APP_IDENTITY_SYSTEM_PROMPT,
                user_prompt=question,
                temperature=get_llm_smalltalk_temperature(),
            )

            yield {
                "type": "token",
                "content": answer,
            }

            yield {
                "type": "done",
                "model": get_model_name(),
                "latency_ms": round((time.time() - start) * 1000, 2),
            }
            return

        except Exception as e:
            logger.exception(f"Small-talk LLM call failed: {e}")

            fallback = (
                "Hello! I am MedLex RAG. Ask me a question about an ingested FDA drug-label document."
            )

            yield {
                "type": "token",
                "content": fallback,
            }

            yield {
                "type": "done",
                "model": "guardrail-fallback",
                "latency_ms": round((time.time() - start) * 1000, 2),
            }
            return

    # ------------------------------------------------------------
    # Guardrail 2: out-of-domain block
    # ------------------------------------------------------------
    if not should_use_rag(question):
        yield {
            "type": "token",
            "content": OUT_OF_DOMAIN_ANSWER,
        }

        yield {
            "type": "done",
            "model": "guardrail",
            "latency_ms": 0,
        }
        return

    # ------------------------------------------------------------
    # RAG retrieval
    # ------------------------------------------------------------
    try:
        chunks = await retrieve_chunks(
            question=question,
            drug=drug,
            domain=domain,
            top_k=top_k,
        )
    except Exception as e:
        logger.exception(f"Retrieval failed: {e}")

        yield {
            "type": "error",
            "message": "Retrieval failed. Please check Pinecone, Neo4j, MySQL, and retriever configuration.",
        }

        yield {
            "type": "done",
            "model": "retrieval-error",
            "latency_ms": round((time.time() - start) * 1000, 2),
        }
        return

    if not chunks:
        yield {
            "type": "token",
            "content": NOT_FOUND_ANSWER,
        }

        yield {
            "type": "done",
            "model": "guardrail",
            "latency_ms": round((time.time() - start) * 1000, 2),
        }
        return

    if not chunks_match_requested_drug(chunks, drug):
        yield {
            "type": "token",
            "content": NOT_FOUND_ANSWER,
        }

        yield {
            "type": "done",
            "model": "guardrail",
            "latency_ms": round((time.time() - start) * 1000, 2),
        }
        return

    context_text, citations = build_context_and_citations(chunks)

    if not context_text.strip():
        yield {
            "type": "token",
            "content": NOT_FOUND_ANSWER,
        }

        yield {
            "type": "done",
            "model": "guardrail",
            "latency_ms": round((time.time() - start) * 1000, 2),
        }
        return

    yield {
        "type": "meta",
        "citations": citations,
        "model": get_model_name(),
    }

    # ------------------------------------------------------------
    # RAG streaming generation
    # ------------------------------------------------------------
    user_prompt = build_rag_user_prompt(question, context_text)

    full_answer = ""

    try:
        async for token in stream_llm_chat(
            system_prompt=RAG_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=get_llm_temperature(),
        ):
            full_answer += token
            yield {
                "type": "token",
                "content": token,
            }

    except Exception as e:
        logger.exception(f"LLM streaming failed: {e}")

        yield {
            "type": "error",
            "message": "LLM streaming failed. Please check Gemini API key, model name, and network connection.",
        }

        yield {
            "type": "done",
            "model": "llm-error",
            "latency_ms": round((time.time() - start) * 1000, 2),
        }
        return

    if not full_answer.strip():
        yield {
            "type": "token",
            "content": NOT_FOUND_ANSWER,
        }

    latency_ms = round((time.time() - start) * 1000, 2)

    logger.info(f"Streamed Q: '{question[:60]}...' → {len(chunks)} chunks, {latency_ms}ms")

    yield {
        "type": "done",
        "model": get_model_name(),
        "latency_ms": latency_ms,
    }