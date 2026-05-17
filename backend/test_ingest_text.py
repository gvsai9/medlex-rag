import asyncio

from ingestion.parser import ingest_text
from ingestion.vector_store import embed_texts, get_vector_store
from ingestion.graph_store import get_graph_store
from state_tracker import get_tracker, DocStatus


async def main():
    text = """
    Paracetamol is commonly used to reduce fever and relieve mild to moderate pain.
    Patients with severe liver disease should consult a doctor before using paracetamol.
    The FDA recommends reading drug labels carefully before taking medication.
    """

    doc = ingest_text(
        text=text,
        source_url="local://paracetamol-test",
        title="Paracetamol Test Document",
        domain="medical",
    )

    print("Doc ID:", doc.doc_id)
    print("Chunks:", len(doc.chunks))
    print("Entities:", len(doc.entities))

    tracker = get_tracker()
    await tracker.init_db()

    await tracker.upsert_document(
        doc_id=doc.doc_id,
        source_url=doc.source_url,
        title=doc.title,
        domain=doc.domain,
        metadata={"test": True},
    )

    for chunk in doc.chunks:
        await tracker.save_chunk(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            content=chunk.content,
            page_num=chunk.page_num,
            chunk_index=chunk.chunk_index,
        )

    embeddings = await embed_texts([c.content for c in doc.chunks])
    print("Embeddings shape:", embeddings.shape)

    vector_store = get_vector_store()
    vector_store.add_chunks(doc.chunks, embeddings)

    for chunk in doc.chunks:
        await tracker.mark_chunk_embedded(chunk.chunk_id)

    graph = get_graph_store()
    await graph.connect()
    await graph.index_document(doc.doc_id, doc.title, doc.domain)
    await graph.index_chunks(doc.chunks)
    await graph.close()

    await tracker.update_status(
        doc_id=doc.doc_id,
        status=DocStatus.DONE,
        chunk_count=len(doc.chunks),
        entity_count=len(doc.entities),
    )

    stats = await tracker.get_stats()
    print("Final stats:", stats)

    await tracker.close()


asyncio.run(main())