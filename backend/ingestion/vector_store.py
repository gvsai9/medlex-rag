"""
Vector Store: Local BGE Embeddings + Pinecone
"""

import asyncio
from typing import List, Dict, Optional
from loguru import logger
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed")

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logger.warning("pinecone package not installed")

from config import get_settings
from ingestion.parser import Chunk

settings = get_settings()

_embedding_model = None


def get_embedding_model():
    global _embedding_model

    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise RuntimeError("sentence-transformers not installed")

    if _embedding_model is None:
        logger.info(f"Loading embedding model: {settings.embedding_model_name}")
        _embedding_model = SentenceTransformer(settings.embedding_model_name)

    return _embedding_model


async def embed_texts(
    texts: List[str],
    task_type: str = "retrieval_document",
) -> np.ndarray:
    if not texts:
        return np.empty((0, settings.embedding_dim), dtype="float32")

    model = get_embedding_model()

    if task_type == "retrieval_query":
        prepared_texts = [
            f"Represent this sentence for searching relevant passages: {text}"
            for text in texts
        ]
    else:
        prepared_texts = texts

    embeddings = await asyncio.to_thread(
        model.encode,
        prepared_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    arr = np.array(embeddings, dtype="float32")

    if arr.shape[1] != settings.embedding_dim:
        raise ValueError(
            f"Embedding dimension mismatch. Got {arr.shape[1]}, expected {settings.embedding_dim}"
        )

    return arr


async def embed_query(query: str) -> np.ndarray:
    arr = await embed_texts([query], task_type="retrieval_query")
    return arr[0]


class PineconeVectorStore:
    def __init__(self):
        self.pc = None
        self.index = None
        self.index_name = settings.pinecone_index_name
        self.namespace = settings.pinecone_namespace
        self._connect()

    def _connect(self):
        if not PINECONE_AVAILABLE:
            logger.warning("Pinecone package not available")
            return

        if not settings.pinecone_api_key:
            logger.warning("Pinecone API key not configured")
            return

        self.pc = Pinecone(api_key=settings.pinecone_api_key)

        existing_indexes = [idx["name"] for idx in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=settings.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.pinecone_cloud,
                    region=settings.pinecone_region,
                ),
            )

        self.index = self.pc.Index(self.index_name)
        logger.success(f"Connected to Pinecone index: {self.index_name}")

    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray):
        if self.index is None:
            logger.warning("Pinecone index not available — skipping")
            return

        vectors = []

        for chunk, embedding in zip(chunks, embeddings):
            metadata = chunk.metadata or {}

            drug = str(metadata.get("drug", "")).lower().strip()
            brand = str(metadata.get("brand", "")).strip()
            source = str(metadata.get("source", "")).strip()

            pinecone_metadata = {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "content": chunk.content[:3000],
                "doc_title": chunk.doc_title,
                "source_url": chunk.source_url,
                "page_num": int(chunk.page_num or 1),
                "section_title": chunk.section_title or "",
                "domain": chunk.domain,
                "drug": drug,
                "brand": brand,
                "source": source,
            }

            vectors.append({
                "id": chunk.chunk_id,
                "values": embedding.tolist(),
                "metadata": pinecone_metadata,
            })

        for i in range(0, len(vectors), 100):
            batch = vectors[i:i + 100]
            self.index.upsert(vectors=batch, namespace=self.namespace)
            logger.debug(f"Upserted Pinecone batch {i // 100 + 1}")

        logger.success(f"Added {len(chunks)} chunks to Pinecone")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        domain_filter: Optional[str] = None,
        drug_filter: Optional[str] = None,
    ) -> List[Dict]:
        if self.index is None:
            return []

        filter_query = {}

        if domain_filter:
            filter_query["domain"] = {"$eq": domain_filter}

        if drug_filter:
            filter_query["drug"] = {"$eq": drug_filter.lower().strip()}

        response = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=filter_query if filter_query else None,
            namespace=self.namespace,
        )

        results = []

        for match in response.get("matches", []):
            metadata = match.get("metadata", {})
            metadata["vector_score"] = float(match.get("score", 0.0))
            results.append(metadata)

        return results

    def delete_namespace(self):
        if self.index is None:
            return

        try:
            self.index.delete(delete_all=True, namespace=self.namespace)
            logger.warning(f"Deleted all Pinecone vectors in namespace: {self.namespace}")
        except Exception as e:
            logger.warning(f"Pinecone namespace delete failed: {e}")

    def total_vectors(self) -> int:
        if self.index is None:
            return 0

        stats = self.index.describe_index_stats()

        if self.namespace:
            ns = stats.get("namespaces", {}).get(self.namespace, {})
            return ns.get("vector_count", 0)

        return stats.get("total_vector_count", 0)


_vector_store: Optional[PineconeVectorStore] = None


def get_vector_store() -> PineconeVectorStore:
    global _vector_store

    if _vector_store is None:
        _vector_store = PineconeVectorStore()

    return _vector_store