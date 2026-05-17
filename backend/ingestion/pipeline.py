"""
Clean ingestion pipeline.
Saves:
- MySQL state/chunks
- Pinecone vectors
- Neo4j document/chunk graph
- Neo4j LLM knowledge graph
"""

from typing import Dict
from loguru import logger

from state_tracker import get_tracker, DocStatus
from ingestion.parser import ParsedDocument
from ingestion.vector_store import embed_texts, get_vector_store
from ingestion.graph_store import get_graph_store
from ingestion.graph_extractor import extract_graph_from_chunks


async def ingest_parsed_document(parsed: ParsedDocument) -> Dict:
    tracker = get_tracker()
    await tracker.init_db()

    graph_store = get_graph_store()
    await graph_store.connect()

    metadata = parsed.metadata or {}
    drug = str(metadata.get("drug", "")).lower().strip()
    brand = str(metadata.get("brand", "")).strip()

    await tracker.upsert_document(
        doc_id=parsed.doc_id,
        source_url=parsed.source_url,
        title=parsed.title,
        domain=parsed.domain,
        metadata=metadata,
    )

    await tracker.update_status(parsed.doc_id, DocStatus.PROCESSING)

    if not parsed.chunks:
        await tracker.update_status(
            parsed.doc_id,
            DocStatus.FAILED,
            error_message="No chunks created",
        )
        return {
            "doc_id": parsed.doc_id,
            "chunks": 0,
            "entities": 0,
            "relationships": 0,
            "status": "failed",
        }

    # Save chunks to MySQL
    for chunk in parsed.chunks:
        cmeta = chunk.metadata or {}
        await tracker.save_chunk(
            chunk_id=chunk.chunk_id,
            doc_id=parsed.doc_id,
            content=chunk.content,
            page_num=chunk.page_num,
            chunk_index=chunk.chunk_index,
            section_title=chunk.section_title,
            domain=chunk.domain,
            drug=str(cmeta.get("drug", drug)).lower().strip(),
            brand=str(cmeta.get("brand", brand)).strip(),
            metadata=cmeta,
        )

    # Embeddings → Pinecone
    embeddings = await embed_texts([c.content for c in parsed.chunks])
    vector_store = get_vector_store()
    vector_store.add_chunks(parsed.chunks, embeddings)

    for chunk in parsed.chunks:
        await tracker.mark_chunk_embedded(chunk.chunk_id)

    # Neo4j base graph
    await graph_store.index_document(
        doc_id=parsed.doc_id,
        title=parsed.title,
        domain=parsed.domain,
        source_url=parsed.source_url,
        drug=drug,
        brand=brand,
    )

    await graph_store.index_chunks(parsed.chunks)

    # LLM graph extraction
    llm_graph = await extract_graph_from_chunks(parsed.chunks)

    await graph_store.index_llm_graph(
        doc_id=parsed.doc_id,
        chunks=parsed.chunks,
        graph=llm_graph,
    )

    await tracker.update_status(
        parsed.doc_id,
        DocStatus.DONE,
        chunk_count=len(parsed.chunks),
        entity_count=len(llm_graph.get("entities", [])),
    )

    logger.success(
        f"Ingested {parsed.title}: {len(parsed.chunks)} chunks, "
        f"{len(llm_graph.get('entities', []))} graph entities, "
        f"{len(llm_graph.get('relationships', []))} graph relationships"
    )

    return {
        "doc_id": parsed.doc_id,
        "chunks": len(parsed.chunks),
        "entities": len(llm_graph.get("entities", [])),
        "relationships": len(llm_graph.get("relationships", [])),
        "status": "done",
    }