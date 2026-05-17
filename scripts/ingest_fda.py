"""
Script: Ingest FDA Drug Label Data
==================================
Fetches free data from OpenFDA API — no API key required.
Ingests drug labels for common drugs as a demo dataset.

Run from backend folder:

  python scripts/ingest_fda.py --limit 2
  python scripts/ingest_fda.py --drugs ibuprofen aspirin metformin --limit 2
"""

import asyncio
import argparse
import sys
from pathlib import Path
import hashlib

# Add backend folder to Python path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import httpx
from loguru import logger

from state_tracker import get_tracker, DocStatus
from ingestion.parser import ingest_text
from ingestion.vector_store import embed_texts, get_vector_store
from ingestion.graph_store import get_graph_store


DEMO_DRUGS = [
    "ibuprofen",
    "aspirin",
    "metformin",
    "lisinopril",
    "atorvastatin",
    "omeprazole",
    "amoxicillin",
    "levothyroxine",
]

FDA_LABEL_FIELDS = [
    "indications_and_usage",
    "warnings",
    "dosage_and_administration",
    "contraindications",
    "adverse_reactions",
    "drug_interactions",
    "warnings_and_cautions",
    "mechanism_of_action",
    "pharmacokinetics",
]


async def ingest_fda_drug(drug_name: str, limit: int, tracker, graph_store):
    """
    Fetch and ingest FDA labels for one drug.
    """
    url = (
        "https://api.fda.gov/drug/label.json"
        f"?search=openfda.generic_name:{drug_name}"
        f"&limit={limit}"
    )

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(url)

            if resp.status_code == 404:
                logger.warning(f"No FDA results for '{drug_name}'")
                return 0

            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        logger.info(f"Found {len(results)} FDA labels for '{drug_name}'")

        ingested = 0

        for item in results:
            brand_names = item.get("openfda", {}).get("brand_name", [drug_name])
            brand = brand_names[0] if brand_names else drug_name

            # Build text from label sections
            text_parts = []

            for field in FDA_LABEL_FIELDS:
                val = item.get(field, [])

                if isinstance(val, list) and val and str(val[0]).strip():
                    section_name = field.upper().replace("_", " ")
                    text_parts.append(f"## {section_name}\n{val[0]}")

            if not text_parts:
                continue

            full_text = (
                f"# FDA Drug Label: {brand}\n\n"
                + "\n\n".join(text_parts)
            )

            source_url = (
                f"https://api.fda.gov/drug/label.json?"
                f"search=openfda.generic_name:{drug_name}&brand={brand}"
            )

            doc_id = hashlib.md5(source_url.encode()).hexdigest()
            title = f"FDA Drug Label: {brand}"

            await tracker.upsert_document(
                doc_id=doc_id,
                source_url=source_url,
                title=title,
                domain="fda",
                metadata={
                    "drug": drug_name,
                    "brand": brand,
                    "source": "openfda",
                },
            )

            await tracker.update_status(doc_id, DocStatus.PROCESSING)

            parsed = ingest_text(
                text=full_text,
                source_url=source_url,
                title=title,
                domain="fda",
            )

            if not parsed.chunks:
                logger.warning(f"No chunks created for {title}")
                await tracker.update_status(
                    doc_id,
                    DocStatus.FAILED,
                    error_message="No chunks created",
                )
                continue

            # Save chunks to MySQL
            for chunk in parsed.chunks:
                await tracker.save_chunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=doc_id,
                    content=chunk.content,
                    page_num=chunk.page_num,
                    chunk_index=chunk.chunk_index,
                )

            # Embed + store in Pinecone
            embeddings = await embed_texts([c.content for c in parsed.chunks])

            vector_store = get_vector_store()
            vector_store.add_chunks(parsed.chunks, embeddings)

            for chunk in parsed.chunks:
                await tracker.mark_chunk_embedded(chunk.chunk_id)

            # Store graph data in Neo4j
            await graph_store.index_document(doc_id, title, "fda")
            await graph_store.index_chunks(parsed.chunks)

            await tracker.update_status(
                doc_id,
                DocStatus.DONE,
                chunk_count=len(parsed.chunks),
                entity_count=len(parsed.entities),
            )

            logger.success(
                f"{title}: {len(parsed.chunks)} chunks, "
                f"{len(parsed.entities)} entities"
            )

            ingested += 1

        return ingested

    except Exception as e:
        logger.error(f"Failed to ingest {drug_name}: {e}")
        return 0


async def main(drugs: list, limit: int):
    logger.info(f"Starting FDA ingestion: {drugs}")

    Path("data/raw").mkdir(parents=True, exist_ok=True)

    tracker = get_tracker()
    await tracker.init_db()

    graph_store = get_graph_store()
    await graph_store.connect()

    total = 0

    for drug in drugs:
        count = await ingest_fda_drug(
            drug_name=drug,
            limit=limit,
            tracker=tracker,
            graph_store=graph_store,
        )

        total += count
        await asyncio.sleep(0.5)

    stats = await tracker.get_stats()

    logger.success(
        f"\n{'=' * 40}\n"
        f"Ingestion complete!\n"
        f"Total labels ingested: {total}\n"
        f"DB stats: {stats}\n"
        f"{'=' * 40}"
    )

    await graph_store.close()
    await tracker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest FDA drug labels")

    parser.add_argument(
        "--drugs",
        nargs="+",
        default=DEMO_DRUGS[:4],
        help="Drug names to ingest",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Labels per drug",
    )

    args = parser.parse_args()

    asyncio.run(main(args.drugs, args.limit))