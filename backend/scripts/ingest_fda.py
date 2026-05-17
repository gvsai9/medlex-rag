"""
Script: Ingest FDA Drug Label Data
==================================
Fetches free data from OpenFDA API — no API key required.

Run from backend:

python scripts/ingest_fda.py --drugs ibuprofen --limit 1
python scripts/ingest_fda.py --drugs ibuprofen aspirin metformin --limit 1
"""

import asyncio
import argparse
import sys
from pathlib import Path
import hashlib
from typing import List

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import httpx
from loguru import logger

from ingestion.parser import ingest_text
from ingestion.pipeline import ingest_parsed_document
from state_tracker import get_tracker
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
    "acetaminophen",
    "naproxen",
]

FDA_LABEL_FIELDS = [
    "indications_and_usage",
    "warnings",
    "boxed_warning",
    "dosage_and_administration",
    "contraindications",
    "adverse_reactions",
    "drug_interactions",
    "warnings_and_cautions",
    "ask_doctor",
    "ask_doctor_or_pharmacist",
    "stop_use",
    "do_not_use",
    "pregnancy_or_breast_feeding",
    "overdosage",
]


def _first_text(value) -> str:
    if isinstance(value, list) and value:
        return "\n".join([str(v) for v in value if str(v).strip()])
    if isinstance(value, str):
        return value
    return ""


async def fetch_fda_labels(drug_name: str, limit: int) -> List[dict]:
    url = (
        "https://api.fda.gov/drug/label.json"
        f"?search=openfda.generic_name:{drug_name}"
        f"&limit={limit}"
    )

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(url)

        if resp.status_code == 404:
            return []

        resp.raise_for_status()
        data = resp.json()

    return data.get("results", [])


async def ingest_fda_drug(drug_name: str, limit: int) -> int:
    drug_name = drug_name.lower().strip()

    try:
        results = await fetch_fda_labels(drug_name, limit)
        logger.info(f"Found {len(results)} FDA labels for '{drug_name}'")

        ingested = 0

        for item in results:
            openfda = item.get("openfda", {}) or {}

            brand_names = openfda.get("brand_name", []) or []
            generic_names = openfda.get("generic_name", []) or []

            brand = brand_names[0] if brand_names else drug_name
            generic = generic_names[0] if generic_names else drug_name

            text_parts = []

            for field in FDA_LABEL_FIELDS:
                section_text = _first_text(item.get(field, []))

                if section_text.strip():
                    section_name = field.upper().replace("_", " ")
                    text_parts.append(f"## {section_name}\n{section_text}")

            if not text_parts:
                logger.warning(f"No usable label sections for {drug_name} / {brand}")
                continue

            full_text = (
                f"# FDA Drug Label: {brand}\n\n"
                f"## DRUG IDENTITY\n"
                f"Generic name: {generic}\n"
                f"Brand name: {brand}\n"
                f"Search drug: {drug_name}\n\n"
                + "\n\n".join(text_parts)
            )

            source_url = (
                "https://api.fda.gov/drug/label.json?"
                f"search=openfda.generic_name:{drug_name}&brand={brand}"
            )

            doc_id = hashlib.md5(source_url.encode()).hexdigest()
            title = f"FDA Drug Label: {brand}"

            metadata = {
                "drug": drug_name,
                "generic": generic,
                "brand": brand,
                "source": "openfda",
                "domain": "fda",
            }

            parsed = ingest_text(
                text=full_text,
                source_url=source_url,
                title=title,
                domain="fda",
                metadata=metadata,
            )

            # Make sure parser doc_id matches our source_url hash
            if parsed.doc_id != doc_id:
                logger.warning("Parsed doc_id mismatch, continuing with parsed doc_id")

            result = await ingest_parsed_document(parsed)

            logger.success(
                f"{title}: {result['chunks']} chunks, "
                f"{result['entities']} graph entities, "
                f"{result['relationships']} graph relationships"
            )

            ingested += 1

        return ingested

    except Exception as e:
        logger.error(f"Failed to ingest {drug_name}: {e}")
        return 0


async def main(drugs: list, limit: int):
    logger.info(f"Starting FDA ingestion: {drugs}")

    Path("data/raw").mkdir(parents=True, exist_ok=True)

    total = 0

    for drug in drugs:
        count = await ingest_fda_drug(drug, limit)
        total += count
        await asyncio.sleep(0.5)

    tracker = get_tracker()
    stats = await tracker.get_stats()

    logger.success(
        f"\n{'=' * 40}\n"
        f"Ingestion complete!\n"
        f"Total labels ingested: {total}\n"
        f"DB stats: {stats}\n"
        f"{'=' * 40}"
    )

    graph_store = get_graph_store()
    await graph_store.close()
    await tracker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest FDA drug labels")

    parser.add_argument(
        "--drugs",
        nargs="+",
        default=["ibuprofen"],
        help="Drug names to ingest",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Labels per drug",
    )

    args = parser.parse_args()

    asyncio.run(main(args.drugs, args.limit))