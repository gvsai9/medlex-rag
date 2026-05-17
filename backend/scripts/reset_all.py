"""
Reset all RAG memory:
- MySQL documents/chunks
- Pinecone namespace
- Neo4j graph

Run from backend:
python scripts/reset_all.py
"""

import asyncio
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from loguru import logger

from state_tracker import get_tracker
from ingestion.vector_store import get_vector_store
from ingestion.graph_store import get_graph_store


async def main():
    logger.warning("Starting full RAG reset...")

    tracker = get_tracker()
    await tracker.init_db()
    await tracker.reset()

    vector_store = get_vector_store()
    vector_store.delete_namespace()

    graph_store = get_graph_store()
    await graph_store.reset()
    await graph_store.close()

    await tracker.close()

    logger.success("Full RAG reset complete")


if __name__ == "__main__":
    asyncio.run(main())