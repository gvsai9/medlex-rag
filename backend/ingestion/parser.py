"""
PDF/Text parser and chunker.
Docling is imported lazily inside the PDF parser to avoid Windows DLL conflicts.
"""

import hashlib
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from loguru import logger

from config import get_settings

settings = get_settings()


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    content: str
    page_num: int
    chunk_index: int
    section_title: str
    doc_title: str
    source_url: str
    domain: str
    entities: List[Dict]
    metadata: Dict


@dataclass
class ParsedDocument:
    doc_id: str
    title: str
    source_url: str
    domain: str
    raw_text: str
    sections: List[Dict]
    chunks: List[Chunk]
    entities: List[Dict]
    metadata: Dict


PII_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
    (r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "[EMAIL]", re.I),
    (r"\b\d{3}[\s.-]?\d{3}[\s.-]?\d{4}\b", "[PHONE]"),
    (r"\b\d{16}\b", "[CC_NUM]"),
]


def redact_pii(text: str) -> str:
    for pattern_args in PII_PATTERNS:
        if len(pattern_args) == 3:
            pattern, replacement, flags = pattern_args
            text = re.sub(pattern, replacement, text, flags=flags)
        else:
            pattern, replacement = pattern_args
            text = re.sub(pattern, replacement, text)
    return text


def extract_entities(text: str) -> List[Dict]:
    """
    Kept for backward compatibility.
    Final KG extraction is done by ingestion/graph_extractor.py using LLM.
    """
    return []


def _split_markdown_sections(text: str) -> List[Dict]:
    """
    Split FDA/OpenFDA text into markdown sections like:
    ## WARNINGS
    content...
    """
    lines = text.splitlines()
    sections = []
    current_title = "Main"
    current_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("## "):
            if current_lines:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_lines).strip(),
                    "page_num": 1,
                })
                current_lines = []

            current_title = stripped.replace("#", "").strip()
        elif stripped.startswith("# "):
            # document title, skip as section content
            continue
        else:
            current_lines.append(line)

    if current_lines:
        sections.append({
            "title": current_title,
            "content": "\n".join(current_lines).strip(),
            "page_num": 1,
        })

    return [s for s in sections if s["content"]]


def semantic_chunk(text: str, chunk_size: int = 512, overlap: int = 51) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        sent_len = len(sent.split())

        if current_len + sent_len > chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)

            overlap_words = chunk_text.split()[-overlap:]
            current_chunk = [" ".join(overlap_words)] if overlap_words else []
            current_len = len(overlap_words)

        current_chunk.append(sent)
        current_len += sent_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return [c.strip() for c in chunks if len(c.strip()) > 40]


def parse_pdf_docling(pdf_path: str) -> Tuple[str, List[Dict]]:
    """
    Use Docling for layout-aware PDF parsing.
    Falls back to PyPDF2 if Docling fails.
    """
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        doc = result.document

        # Prefer markdown export if available
        try:
            markdown = doc.export_to_markdown()
            if markdown and markdown.strip():
                sections = _split_markdown_sections(markdown)
                if not sections:
                    sections = [{"title": "Document", "content": markdown, "page_num": 1}]
                return markdown, sections
        except Exception:
            pass

        sections = []
        full_text_parts = []
        current_section = {
            "title": "Introduction",
            "content": [],
            "page_num": 1,
        }

        for item, _level in doc.iterate_items():
            label = str(getattr(item, "label", "text")).lower()
            text = getattr(item, "text", "") or ""

            page_num = 1
            prov = getattr(item, "prov", None)
            if prov:
                try:
                    page_num = prov[0].page_no
                except Exception:
                    page_num = 1

            if not text.strip():
                continue

            if "title" in label or "section" in label or "heading" in label:
                if current_section["content"]:
                    current_section["content"] = " ".join(current_section["content"])
                    sections.append(current_section)

                current_section = {
                    "title": text.strip(),
                    "content": [],
                    "page_num": page_num,
                }

            elif "table" in label:
                table_text = f"[TABLE] {text.strip()}"
                current_section["content"].append(table_text)
                full_text_parts.append(table_text)

            else:
                current_section["content"].append(text.strip())
                full_text_parts.append(text.strip())

        if current_section["content"]:
            current_section["content"] = " ".join(current_section["content"])
            sections.append(current_section)

        return " ".join(full_text_parts), sections

    except Exception as e:
        logger.warning(f"Docling parsing failed, using PyPDF2 fallback: {e}")
        return parse_pdf_fallback(pdf_path)


def parse_pdf_fallback(pdf_path: str) -> Tuple[str, List[Dict]]:
    try:
        import PyPDF2

        text_parts = []
        sections = []

        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)

            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                text_parts.append(page_text)

                if page_text.strip():
                    sections.append({
                        "title": f"Page {i + 1}",
                        "content": page_text,
                        "page_num": i + 1,
                    })

        return " ".join(text_parts), sections

    except Exception as e:
        logger.error(f"PDF parsing failed: {e}")
        return "", []


def _make_chunks(
    doc_id: str,
    title: str,
    source_url: str,
    domain: str,
    sections: List[Dict],
    base_metadata: Optional[Dict] = None,
) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    chunk_index = 0
    base_metadata = base_metadata or {}

    for section in sections:
        section_text = redact_pii(section.get("content", ""))

        sub_chunks = semantic_chunk(
            section_text,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

        for sub in sub_chunks:
            chunk_id = hashlib.md5(
                f"{doc_id}_{chunk_index}_{sub[:50]}".encode()
            ).hexdigest()

            metadata = {
                **base_metadata,
                "section": section.get("title", ""),
                "page": section.get("page_num", 1),
                "domain": domain,
            }

            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=sub,
                page_num=section.get("page_num", 1),
                chunk_index=chunk_index,
                section_title=section.get("title", ""),
                doc_title=title,
                source_url=source_url,
                domain=domain,
                entities=[],
                metadata=metadata,
            )

            all_chunks.append(chunk)
            chunk_index += 1

    return all_chunks


def ingest_pdf(
    pdf_path: str,
    source_url: str,
    title: str,
    domain: str = "general",
    metadata: Optional[Dict] = None,
) -> ParsedDocument:
    doc_id = hashlib.md5(source_url.encode()).hexdigest()
    logger.info(f"Ingesting PDF {title} (doc_id={doc_id})")

    raw_text, sections = parse_pdf_docling(pdf_path)

    chunks = _make_chunks(
        doc_id=doc_id,
        title=title,
        source_url=source_url,
        domain=domain,
        sections=sections,
        base_metadata=metadata,
    )

    return ParsedDocument(
        doc_id=doc_id,
        title=title,
        source_url=source_url,
        domain=domain,
        raw_text=raw_text,
        sections=sections,
        chunks=chunks,
        entities=[],
        metadata=metadata or {},
    )


def ingest_text(
    text: str,
    source_url: str,
    title: str,
    domain: str = "general",
    metadata: Optional[Dict] = None,
) -> ParsedDocument:
    doc_id = hashlib.md5(source_url.encode()).hexdigest()
    clean_text = redact_pii(text)

    sections = _split_markdown_sections(clean_text)
    if not sections:
        sections = [{"title": "Main", "content": clean_text, "page_num": 1}]

    chunks = _make_chunks(
        doc_id=doc_id,
        title=title,
        source_url=source_url,
        domain=domain,
        sections=sections,
        base_metadata=metadata,
    )

    return ParsedDocument(
        doc_id=doc_id,
        title=title,
        source_url=source_url,
        domain=domain,
        raw_text=clean_text,
        sections=sections,
        chunks=chunks,
        entities=[],
        metadata=metadata or {},
    )