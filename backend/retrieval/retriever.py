"""
Hybrid Retriever
================
Combines:
- Pinecone vector retrieval
- BM25 over MySQL chunks
- Neo4j graph retrieval

Supports drug_filter to prevent cross-drug contamination.
"""

import re
import math
from typing import List, Dict, Optional
from collections import defaultdict
from loguru import logger

from rank_bm25 import BM25Okapi

from config import get_settings
from state_tracker import get_tracker
from ingestion.vector_store import embed_query, get_vector_store
from ingestion.graph_store import get_graph_store

settings = get_settings()


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z0-9]+\b", (text or "").lower())


class HybridRetriever:
    def __init__(self):
        self.tracker = get_tracker()
        self.vector_store = get_vector_store()
        self.graph_store = get_graph_store()

        self.bm25 = None
        self.bm25_chunks: List[Dict] = []
        self.bm25_tokens: List[List[str]] = []

    async def build(self):
        """
        Build BM25 from MySQL chunks.
        """
        await self.tracker.init_db()

        self.bm25_chunks = await self.tracker.list_chunks(limit=100000)
        self.bm25_tokens = [tokenize(c.get("content", "")) for c in self.bm25_chunks]

        if self.bm25_tokens:
            self.bm25 = BM25Okapi(self.bm25_tokens)
            logger.info(f"BM25 index built with {len(self.bm25_chunks)} chunks")
        else:
            self.bm25 = None
            logger.warning("No chunks found for BM25")

    async def _bm25_search(
        self,
        query: str,
        top_k: int = 10,
        domain_filter: Optional[str] = None,
        drug_filter: Optional[str] = None,
    ) -> List[Dict]:
        if self.bm25 is None:
            await self.build()

        if self.bm25 is None:
            return []

        q_tokens = tokenize(query)
        scores = self.bm25.get_scores(q_tokens)

        candidates = []

        for idx, score in enumerate(scores):
            chunk = dict(self.bm25_chunks[idx])

            if domain_filter and chunk.get("domain") != domain_filter:
                continue

            if drug_filter:
                chunk_drug = str(chunk.get("drug", "")).lower().strip()
                if chunk_drug != drug_filter.lower().strip():
                    continue

            chunk["bm25_score"] = float(score)
            candidates.append(chunk)

        candidates.sort(key=lambda x: x.get("bm25_score", 0.0), reverse=True)
        return candidates[:top_k]

    async def _vector_search(
        self,
        query: str,
        top_k: int = 10,
        domain_filter: Optional[str] = None,
        drug_filter: Optional[str] = None,
    ) -> List[Dict]:
        q_emb = await embed_query(query)

        results = self.vector_store.search(
            query_embedding=q_emb,
            top_k=top_k,
            domain_filter=domain_filter,
            drug_filter=drug_filter,
        )

        return results

    async def _graph_search(
        self,
        query: str,
        top_k: int = 10,
        drug_filter: Optional[str] = None,
    ) -> List[Dict]:
        terms = tokenize(query)
        important_terms = [
            t for t in terms
            if len(t) > 3 and t not in {"what", "when", "where", "which", "should", "patients", "using"}
        ]

        results = await self.graph_store.find_related_chunks(
            query_terms=important_terms[:8],
            drug_filter=drug_filter,
            limit=top_k,
        )

        return results

    def _normalize_scores(self, items: List[Dict], score_key: str, output_key: str):
        if not items:
            return

        values = [float(i.get(score_key, 0.0)) for i in items]
        min_v = min(values)
        max_v = max(values)

        for item in items:
            val = float(item.get(score_key, 0.0))
            if math.isclose(max_v, min_v):
                item[output_key] = 1.0 if val > 0 else 0.0
            else:
                item[output_key] = (val - min_v) / (max_v - min_v)

    def _merge_results(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
        graph_results: List[Dict],
        top_k: int,
    ) -> List[Dict]:
        merged = {}

        self._normalize_scores(vector_results, "vector_score", "vector_norm")
        self._normalize_scores(bm25_results, "bm25_score", "bm25_norm")
        self._normalize_scores(graph_results, "graph_score", "graph_norm")

        def get_key(item: Dict):
            return item.get("chunk_id") or item.get("id")

        for item in vector_results:
            key = get_key(item)
            if not key:
                continue
            merged[key] = {
                **item,
                "chunk_id": key,
                "vector_norm": item.get("vector_norm", 0.0),
                "bm25_norm": 0.0,
                "graph_norm": 0.0,
            }

        for item in bm25_results:
            key = get_key(item)
            if not key:
                continue

            if key not in merged:
                merged[key] = {
                    **item,
                    "chunk_id": key,
                    "vector_norm": 0.0,
                    "bm25_norm": item.get("bm25_norm", 0.0),
                    "graph_norm": 0.0,
                }
            else:
                merged[key].update({
                    "bm25_score": item.get("bm25_score", 0.0),
                    "bm25_norm": item.get("bm25_norm", 0.0),
                    "content": merged[key].get("content") or item.get("content"),
                    "doc_title": merged[key].get("doc_title") or item.get("doc_title"),
                    "source_url": merged[key].get("source_url") or item.get("source_url"),
                    "page_num": merged[key].get("page_num") or item.get("page_num"),
                    "section_title": merged[key].get("section_title") or item.get("section_title"),
                    "drug": merged[key].get("drug") or item.get("drug"),
                    "brand": merged[key].get("brand") or item.get("brand"),
                    "domain": merged[key].get("domain") or item.get("domain"),
                })

        for item in graph_results:
            key = get_key(item)
            if not key:
                continue

            if key not in merged:
                merged[key] = {
                    **item,
                    "chunk_id": key,
                    "vector_norm": 0.0,
                    "bm25_norm": 0.0,
                    "graph_norm": item.get("graph_norm", 0.0),
                }
            else:
                merged[key].update({
                    "graph_score": item.get("graph_score", 0.0),
                    "graph_norm": item.get("graph_norm", 0.0),
                    "content": merged[key].get("content") or item.get("content"),
                    "page_num": merged[key].get("page_num") or item.get("page_num"),
                    "section_title": merged[key].get("section_title") or item.get("section_title"),
                    "drug": merged[key].get("drug") or item.get("drug"),
                    "brand": merged[key].get("brand") or item.get("brand"),
                    "domain": merged[key].get("domain") or item.get("domain"),
                })

        final = []

        for item in merged.values():
            # Weighted fusion
            item["score"] = (
                0.50 * float(item.get("vector_norm", 0.0))
                + 0.30 * float(item.get("bm25_norm", 0.0))
                + 0.20 * float(item.get("graph_norm", 0.0))
            )
            final.append(item)

        final.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        return final[:top_k]

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        domain_filter: Optional[str] = None,
        drug_filter: Optional[str] = None,
    ) -> List[Dict]:
        top_k = top_k or settings.top_k_retrieval

        vector_results = await self._vector_search(
            query=query,
            top_k=top_k * 3,
            domain_filter=domain_filter,
            drug_filter=drug_filter,
        )

        bm25_results = await self._bm25_search(
            query=query,
            top_k=top_k * 3,
            domain_filter=domain_filter,
            drug_filter=drug_filter,
        )

        graph_results = await self._graph_search(
            query=query,
            top_k=top_k * 3,
            drug_filter=drug_filter,
        )

        final = self._merge_results(
            vector_results=vector_results,
            bm25_results=bm25_results,
            graph_results=graph_results,
            top_k=top_k,
        )

        logger.info(
            f"Retrieved {len(final)} chunks "
            f"(vector={len(vector_results)}, bm25={len(bm25_results)}, graph={len(graph_results)})"
        )

        return final


_retriever: Optional[HybridRetriever] = None


async def get_retriever() -> HybridRetriever:
    global _retriever

    if _retriever is None:
        _retriever = HybridRetriever()
        await _retriever.build()

    return _retriever