"""
Ask one drug-specific RAG question.

Run:
python tests/test_answer_one_drug.py --drug ibuprofen --question "What are the warnings for ibuprofen?"
"""

import asyncio
import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from retrieval.generator import answer_question
from state_tracker import get_tracker
from ingestion.graph_store import get_graph_store


async def main(drug: str, question: str):
    result = await answer_question(
        question=question,
        domain_filter="fda",
        drug_filter=drug,
        top_k=5,
    )

    print("\nQuestion:")
    print(result["question"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nCitations:")
    for c in result["citations"]:
        print(
            f"[{c['id']}] {c['doc_title']} | page {c['page_num']} "
            f"| section {c['section_title']} | drug={c['drug']} "
            f"| score={c['score']}"
        )

    print("\nModel:", result["model"])
    print("Latency:", result["latency_ms"], "ms")

    tracker = get_tracker()
    await tracker.close()

    graph = get_graph_store()
    await graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug", required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    asyncio.run(main(args.drug, args.question))