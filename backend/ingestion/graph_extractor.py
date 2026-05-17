"""
LLM Knowledge Graph Extractor
=============================
Uses Ollama to extract structured medical/FDA graph facts from chunks.

Important:
- LLM output is validated.
- Only allowed node labels and relationship types are accepted.
- Evidence must come from the chunk text.
"""

import json
import re
from typing import List, Dict, Any
from loguru import logger
import httpx

from config import get_settings
from ingestion.parser import Chunk

settings = get_settings()


ALLOWED_ENTITY_TYPES = {
    "Drug",
    "Brand",
    "Section",
    "Warning",
    "Condition",
    "Symptom",
    "Risk",
    "Dosage",
    "Interaction",
    "Population",
    "Instruction",
}

ALLOWED_RELATIONS = {
    "HAS_WARNING",
    "MENTIONS_CONDITION",
    "MENTIONS_SYMPTOM",
    "MENTIONS_RISK",
    "HAS_DOSAGE",
    "INTERACTS_WITH",
    "APPLIES_TO",
    "HAS_INSTRUCTION",
    "PART_OF_SECTION",
    "BRAND_OF",
    "MENTIONS",
}


GRAPH_SYSTEM_PROMPT = """
You are a strict medical knowledge graph extraction engine.

Extract entities and relationships ONLY from the given text.

Allowed entity types:
Drug, Brand, Section, Warning, Condition, Symptom, Risk, Dosage, Interaction, Population, Instruction

Allowed relationships:
HAS_WARNING, MENTIONS_CONDITION, MENTIONS_SYMPTOM, MENTIONS_RISK,
HAS_DOSAGE, INTERACTS_WITH, APPLIES_TO, HAS_INSTRUCTION,
PART_OF_SECTION, BRAND_OF, MENTIONS

Rules:
1. Output ONLY valid JSON.
2. Do not include markdown.
3. Do not invent facts.
4. Use short normalized names.
5. Evidence must be copied or closely paraphrased from the text.
6. If no useful graph facts exist, return {"entities": [], "relationships": []}.

JSON schema:
{
  "entities": [
    {"type": "Drug", "name": "ibuprofen"}
  ],
  "relationships": [
    {
      "source_type": "Drug",
      "source": "ibuprofen",
      "relation": "HAS_WARNING",
      "target_type": "Risk",
      "target": "stomach bleeding",
      "evidence": "may cause severe stomach bleeding"
    }
  ]
}
"""


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Robustly extract JSON object from LLM output.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {"entities": [], "relationships": []}


def _normalize_name(name: str) -> str:
    name = str(name or "").strip()
    name = re.sub(r"\s+", " ", name)
    return name[:200]


def _validate_graph(data: Dict[str, Any]) -> Dict[str, List[Dict]]:
    entities = []
    relationships = []

    seen_entities = set()

    for ent in data.get("entities", []):
        etype = ent.get("type")
        name = _normalize_name(ent.get("name"))

        if etype not in ALLOWED_ENTITY_TYPES:
            continue

        if not name:
            continue

        key = (etype, name.lower())

        if key not in seen_entities:
            seen_entities.add(key)
            entities.append({
                "type": etype,
                "name": name,
            })

    for rel in data.get("relationships", []):
        source_type = rel.get("source_type")
        target_type = rel.get("target_type")
        relation = rel.get("relation")
        source = _normalize_name(rel.get("source"))
        target = _normalize_name(rel.get("target"))
        evidence = _normalize_name(rel.get("evidence"))

        if source_type not in ALLOWED_ENTITY_TYPES:
            continue

        if target_type not in ALLOWED_ENTITY_TYPES:
            continue

        if relation not in ALLOWED_RELATIONS:
            continue

        if not source or not target:
            continue

        relationships.append({
            "source_type": source_type,
            "source": source,
            "relation": relation,
            "target_type": target_type,
            "target": target,
            "evidence": evidence,
        })

        # Ensure relationship endpoints are also entities
        for etype, name in [(source_type, source), (target_type, target)]:
            key = (etype, name.lower())
            if key not in seen_entities:
                seen_entities.add(key)
                entities.append({
                    "type": etype,
                    "name": name,
                })

    return {
        "entities": entities,
        "relationships": relationships,
    }


async def _call_ollama_graph(prompt: str) -> str:
    payload = {
        "model": settings.graph_llm_model,
        "messages": [
            {"role": "system", "content": GRAPH_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0,
        },
    }

    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(
            f"{settings.ollama_base_url}/api/chat",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]


async def extract_graph_from_chunk(chunk: Chunk) -> Dict[str, List[Dict]]:
    """
    Extract graph from one chunk.
    """
    drug = str((chunk.metadata or {}).get("drug", "")).strip()
    brand = str((chunk.metadata or {}).get("brand", "")).strip()
    section = chunk.section_title or str((chunk.metadata or {}).get("section", ""))

    prompt = f"""
Document title: {chunk.doc_title}
Drug: {drug}
Brand: {brand}
Section: {section}

Text:
{chunk.content}

Extract the knowledge graph JSON now.
"""

    try:
        raw = await _call_ollama_graph(prompt)
        data = _extract_json(raw)
        validated = _validate_graph(data)

    except Exception as e:
        logger.warning(f"LLM graph extraction failed for chunk {chunk.chunk_id}: {e}")
        validated = {"entities": [], "relationships": []}

    # Add guaranteed basic entities/relations from metadata
    guaranteed_entities = []

    if drug:
        guaranteed_entities.append({"type": "Drug", "name": drug.lower()})

    if brand:
        guaranteed_entities.append({"type": "Brand", "name": brand})

    if section:
        guaranteed_entities.append({"type": "Section", "name": section})

    existing = {
        (e["type"], e["name"].lower())
        for e in validated["entities"]
    }

    for e in guaranteed_entities:
        key = (e["type"], e["name"].lower())
        if key not in existing:
            validated["entities"].append(e)
            existing.add(key)

    if drug and brand:
        validated["relationships"].append({
            "source_type": "Brand",
            "source": brand,
            "relation": "BRAND_OF",
            "target_type": "Drug",
            "target": drug.lower(),
            "evidence": f"{brand} is a label/brand for {drug}",
        })

    if drug and section:
        validated["relationships"].append({
            "source_type": "Drug",
            "source": drug.lower(),
            "relation": "MENTIONS",
            "target_type": "Section",
            "target": section,
            "evidence": f"Chunk belongs to section {section}",
        })

    return validated


async def extract_graph_from_chunks(chunks: List[Chunk]) -> Dict[str, List[Dict]]:
    """
    Extract graph from multiple chunks.
    """
    all_entities = []
    all_relationships = []
    seen = set()

    for chunk in chunks:
        graph = await extract_graph_from_chunk(chunk)

        for ent in graph["entities"]:
            key = (ent["type"], ent["name"].lower())
            if key not in seen:
                seen.add(key)
                all_entities.append(ent)

        all_relationships.extend(graph["relationships"])

    return {
        "entities": all_entities,
        "relationships": all_relationships,
    }