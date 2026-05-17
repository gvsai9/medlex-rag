"""
Neo4j Graph Store
=================
Stores schema-based medical/FDA knowledge graph.

Core graph:
(Document)-[:HAS_CHUNK]->(Chunk)
(Document)-[:ABOUT_DRUG]->(Drug)
(Document)-[:HAS_BRAND]->(Brand)
(Brand)-[:BRAND_OF]->(Drug)
(Chunk)-[:PART_OF_SECTION]->(Section)
(Chunk)-[:MENTIONS]->(Entity)
Plus LLM extracted relationships.
"""

from typing import List, Dict, Optional
from loguru import logger
from neo4j import AsyncGraphDatabase

from config import get_settings
from ingestion.parser import Chunk

settings = get_settings()


ALLOWED_LABELS = {
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


class Neo4jGraphStore:
    def __init__(self):
        self.driver = None

    async def connect(self):
        if self.driver is not None:
            return

        if not settings.neo4j_uri:
            logger.warning("Neo4j URI not configured")
            return

        self.driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )

        await self.driver.verify_connectivity()
        logger.success("Connected to Neo4j")

        await self.create_constraints()

    async def create_constraints(self):
        if self.driver is None:
            return

        constraints = [
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT drug_name IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT brand_name IF NOT EXISTS FOR (b:Brand) REQUIRE b.name IS UNIQUE",
            "CREATE CONSTRAINT section_name IF NOT EXISTS FOR (s:Section) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT warning_name IF NOT EXISTS FOR (w:Warning) REQUIRE w.name IS UNIQUE",
            "CREATE CONSTRAINT condition_name IF NOT EXISTS FOR (c:Condition) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT symptom_name IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT risk_name IF NOT EXISTS FOR (r:Risk) REQUIRE r.name IS UNIQUE",
            "CREATE CONSTRAINT dosage_name IF NOT EXISTS FOR (d:Dosage) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT interaction_name IF NOT EXISTS FOR (i:Interaction) REQUIRE i.name IS UNIQUE",
            "CREATE CONSTRAINT population_name IF NOT EXISTS FOR (p:Population) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT instruction_name IF NOT EXISTS FOR (i:Instruction) REQUIRE i.name IS UNIQUE",
        ]

        async with self.driver.session(database=settings.neo4j_database or None) as session:
            for c in constraints:
                try:
                    await session.run(c)
                except Exception as e:
                    logger.warning(f"Constraint creation skipped/failed: {e}")

    async def reset(self):
        await self.connect()

        if self.driver is None:
            return

        async with self.driver.session(database=settings.neo4j_database or None) as session:
            await session.run("MATCH (n) DETACH DELETE n")

        logger.warning("Neo4j graph reset complete")

    async def index_document(
        self,
        doc_id: str,
        title: str,
        domain: str,
        source_url: str = "",
        drug: str = "",
        brand: str = "",
    ):
        await self.connect()

        if self.driver is None:
            return

        drug = (drug or "").lower().strip()
        brand = (brand or "").strip()

        async with self.driver.session(database=settings.neo4j_database or None) as session:
            await session.run("""
                MERGE (d:Document {doc_id: $doc_id})
                SET d.title = $title,
                    d.domain = $domain,
                    d.source_url = $source_url
            """, {
                "doc_id": doc_id,
                "title": title,
                "domain": domain,
                "source_url": source_url,
            })

            if drug:
                await session.run("""
                    MERGE (drug:Drug {name: $drug})
                    WITH drug
                    MATCH (d:Document {doc_id: $doc_id})
                    MERGE (d)-[:ABOUT_DRUG]->(drug)
                """, {
                    "doc_id": doc_id,
                    "drug": drug,
                })

            if brand:
                await session.run("""
                    MERGE (brand:Brand {name: $brand})
                    WITH brand
                    MATCH (d:Document {doc_id: $doc_id})
                    MERGE (d)-[:HAS_BRAND]->(brand)
                """, {
                    "doc_id": doc_id,
                    "brand": brand,
                })

            if drug and brand:
                await session.run("""
                    MATCH (brand:Brand {name: $brand})
                    MATCH (drug:Drug {name: $drug})
                    MERGE (brand)-[:BRAND_OF]->(drug)
                """, {
                    "brand": brand,
                    "drug": drug,
                })

    async def index_chunks(self, chunks: List[Chunk]):
        await self.connect()

        if self.driver is None:
            return

        async with self.driver.session(database=settings.neo4j_database or None) as session:
            for chunk in chunks:
                metadata = chunk.metadata or {}
                drug = str(metadata.get("drug", "")).lower().strip()
                brand = str(metadata.get("brand", "")).strip()
                section = chunk.section_title or str(metadata.get("section", ""))

                await session.run("""
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                    SET c.doc_id = $doc_id,
                        c.content = $content,
                        c.page_num = $page_num,
                        c.chunk_index = $chunk_index,
                        c.section_title = $section_title,
                        c.domain = $domain,
                        c.drug = $drug,
                        c.brand = $brand

                    WITH c
                    MATCH (d:Document {doc_id: $doc_id})
                    MERGE (d)-[:HAS_CHUNK]->(c)
                """, {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "content": chunk.content[:4000],
                    "page_num": int(chunk.page_num or 1),
                    "chunk_index": int(chunk.chunk_index or 0),
                    "section_title": section,
                    "domain": chunk.domain,
                    "drug": drug,
                    "brand": brand,
                })

                if section:
                    await session.run("""
                        MERGE (s:Section {name: $section})
                        WITH s
                        MATCH (c:Chunk {chunk_id: $chunk_id})
                        MERGE (c)-[:PART_OF_SECTION]->(s)
                    """, {
                        "chunk_id": chunk.chunk_id,
                        "section": section,
                    })

                if drug:
                    await session.run("""
                        MERGE (drug:Drug {name: $drug})
                        WITH drug
                        MATCH (c:Chunk {chunk_id: $chunk_id})
                        MERGE (c)-[:MENTIONS]->(drug)
                    """, {
                        "chunk_id": chunk.chunk_id,
                        "drug": drug,
                    })

                if brand:
                    await session.run("""
                        MERGE (brand:Brand {name: $brand})
                        WITH brand
                        MATCH (c:Chunk {chunk_id: $chunk_id})
                        MERGE (c)-[:MENTIONS]->(brand)
                    """, {
                        "chunk_id": chunk.chunk_id,
                        "brand": brand,
                    })

        logger.success(f"Indexed {len(chunks)} chunks in Neo4j")

    async def index_llm_graph(
        self,
        doc_id: str,
        chunks: List[Chunk],
        graph: Dict[str, List[Dict]],
    ):
        """
        Insert LLM-extracted graph and connect mentioned entities to chunks.
        """
        await self.connect()

        if self.driver is None:
            return

        async with self.driver.session(database=settings.neo4j_database or None) as session:
            # Create entity nodes
            for ent in graph.get("entities", []):
                label = ent.get("type")
                name = str(ent.get("name", "")).strip()

                if label not in ALLOWED_LABELS or not name:
                    continue

                cypher = f"""
                    MERGE (e:{label} {{name: $name}})
                """
                await session.run(cypher, {"name": name})

            # Create relationships
            for rel in graph.get("relationships", []):
                source_type = rel.get("source_type")
                target_type = rel.get("target_type")
                relation = rel.get("relation")
                source = str(rel.get("source", "")).strip()
                target = str(rel.get("target", "")).strip()
                evidence = str(rel.get("evidence", "")).strip()

                if source_type not in ALLOWED_LABELS:
                    continue
                if target_type not in ALLOWED_LABELS:
                    continue
                if relation not in ALLOWED_RELATIONS:
                    continue
                if not source or not target:
                    continue

                cypher = f"""
                    MERGE (s:{source_type} {{name: $source}})
                    MERGE (t:{target_type} {{name: $target}})
                    MERGE (s)-[r:{relation}]->(t)
                    SET r.evidence = $evidence,
                        r.doc_id = $doc_id
                """

                await session.run(cypher, {
                    "source": source,
                    "target": target,
                    "evidence": evidence[:1000],
                    "doc_id": doc_id,
                })

            # Connect chunk to all entities that appear in its text or metadata
            for chunk in chunks:
                text_lower = chunk.content.lower()
                metadata = chunk.metadata or {}

                for ent in graph.get("entities", []):
                    label = ent.get("type")
                    name = str(ent.get("name", "")).strip()

                    if label not in ALLOWED_LABELS or not name:
                        continue

                    name_lower = name.lower()

                    should_link = (
                        name_lower in text_lower
                        or name_lower == str(metadata.get("drug", "")).lower()
                        or name_lower == str(metadata.get("brand", "")).lower()
                        or name_lower == str(metadata.get("section", "")).lower()
                    )

                    if not should_link:
                        continue

                    cypher = f"""
                        MATCH (c:Chunk {{chunk_id: $chunk_id}})
                        MATCH (e:{label} {{name: $name}})
                        MERGE (c)-[:MENTIONS]->(e)
                    """

                    await session.run(cypher, {
                        "chunk_id": chunk.chunk_id,
                        "name": name,
                    })

        logger.success(
            f"Indexed LLM graph: {len(graph.get('entities', []))} entities, "
            f"{len(graph.get('relationships', []))} relationships"
        )

    async def find_related_chunks(
        self,
        query_terms: List[str],
        drug_filter: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict]:
        await self.connect()

        if self.driver is None:
            return []

        query_terms = [q.lower().strip() for q in query_terms if q and q.strip()]
        drug_filter = (drug_filter or "").lower().strip()

        async with self.driver.session(database=settings.neo4j_database or None) as session:
            if drug_filter:
                result = await session.run("""
                    MATCH (:Drug {name: $drug})<-[:ABOUT_DRUG]-(d:Document)-[:HAS_CHUNK]->(c:Chunk)
                    OPTIONAL MATCH (c)-[:MENTIONS]->(e)
                    WITH c, collect(e.name) AS ents
                    RETURN
                        c.chunk_id AS chunk_id,
                        c.doc_id AS doc_id,
                        c.content AS content,
                        c.page_num AS page_num,
                        c.section_title AS section_title,
                        c.domain AS domain,
                        c.drug AS drug,
                        c.brand AS brand,
                        1.0 + size(ents) * 0.05 AS graph_score
                    ORDER BY graph_score DESC
                    LIMIT $limit
                """, {
                    "drug": drug_filter,
                    "limit": limit,
                })
            elif query_terms:
                result = await session.run("""
                    MATCH (e)
                    WHERE toLower(e.name) IN $terms
                    MATCH (c:Chunk)-[:MENTIONS]->(e)
                    RETURN DISTINCT
                        c.chunk_id AS chunk_id,
                        c.doc_id AS doc_id,
                        c.content AS content,
                        c.page_num AS page_num,
                        c.section_title AS section_title,
                        c.domain AS domain,
                        c.drug AS drug,
                        c.brand AS brand,
                        1.0 AS graph_score
                    LIMIT $limit
                """, {
                    "terms": query_terms,
                    "limit": limit,
                })
            else:
                return []

            rows = []
            async for record in result:
                rows.append(dict(record))

            return rows

    async def inspect_drug(self, drug: str) -> Dict:
        await self.connect()

        drug = drug.lower().strip()

        if self.driver is None:
            return {}

        async with self.driver.session(database=settings.neo4j_database or None) as session:
            node_result = await session.run("""
                MATCH (d:Drug {name: $drug})
                OPTIONAL MATCH path=(d)-[r*1..2]-(n)
                RETURN labels(n) AS labels, n.name AS name
                LIMIT 50
            """, {"drug": drug})

            nodes = []
            async for record in node_result:
                nodes.append({
                    "labels": record["labels"],
                    "name": record["name"],
                })

            rel_result = await session.run("""
                MATCH (d:Drug {name: $drug})-[r]-(n)
                RETURN type(r) AS rel, labels(n) AS labels, n.name AS name
                LIMIT 50
            """, {"drug": drug})

            rels = []
            async for record in rel_result:
                rels.append({
                    "rel": record["rel"],
                    "labels": record["labels"],
                    "name": record["name"],
                })

            return {
                "drug": drug,
                "nodes": nodes,
                "relationships": rels,
            }

    async def close(self):
        if self.driver:
            await self.driver.close()
            self.driver = None


_graph_store: Optional[Neo4jGraphStore] = None


def get_graph_store() -> Neo4jGraphStore:
    global _graph_store

    if _graph_store is None:
        _graph_store = Neo4jGraphStore()

    return _graph_store