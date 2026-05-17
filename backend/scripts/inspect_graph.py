"""
Inspect Neo4j graph for one drug.

Run:
python scripts/inspect_graph.py --drug ibuprofen
"""

import asyncio
import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from ingestion.graph_store import get_graph_store


def display_node(node):
    labels = node.get("labels") or []
    props = node.get("props") or {}

    label = labels[0] if labels else "Node"

    if label == "Document":
        name = props.get("title") or props.get("doc_id")
    elif label == "Chunk":
        section = props.get("section_title") or "Unknown section"
        page = props.get("page_num") or "?"
        chunk_id = props.get("chunk_id") or ""
        name = f"{section} | page {page} | {chunk_id[:8]}"
    else:
        name = props.get("name") or props.get("title") or props.get("chunk_id")

    return f"{label} :: {name}"


async def main(drug: str):
    graph = get_graph_store()
    await graph.connect()

    drug = drug.lower().strip()

    async with graph.driver.session() as session:
        node_result = await session.run("""
            MATCH (d:Drug {name: $drug})
            OPTIONAL MATCH path=(d)-[r*1..2]-(n)
            RETURN DISTINCT labels(n) AS labels, properties(n) AS props
            LIMIT 100
        """, {"drug": drug})

        nodes = []
        async for record in node_result:
            nodes.append({
                "labels": record["labels"],
                "props": record["props"],
            })

        rel_result = await session.run("""
            MATCH (d:Drug {name: $drug})-[r]-(n)
            RETURN
                type(r) AS rel,
                labels(n) AS labels,
                properties(n) AS props
            LIMIT 100
        """, {"drug": drug})

        rels = []
        async for record in rel_result:
            rels.append({
                "rel": record["rel"],
                "labels": record["labels"],
                "props": record["props"],
            })

    print("\n==============================")
    print(f"GRAPH INSPECTION: {drug}")
    print("==============================\n")

    print("Nodes:")
    for node in nodes:
        print("-", display_node(node))

    print("\nRelationships:")
    for rel in rels:
        node_text = display_node({
            "labels": rel.get("labels"),
            "props": rel.get("props"),
        })
        print(f"- Drug -[{rel.get('rel')}]- {node_text}")

    await graph.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug", required=True)
    args = parser.parse_args()

    asyncio.run(main(args.drug))