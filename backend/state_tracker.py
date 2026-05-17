"""
MySQL-based document state tracker.
Tracks every document through:
queued → processing → done / failed
"""

import aiomysql
import json
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict
from loguru import logger

from config import get_settings

settings = get_settings()


class DocStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class StateTracker:
    def __init__(self):
        self.pool = None

    async def connect(self):
        if self.pool is not None:
            return

        self.pool = await aiomysql.create_pool(
            host=settings.mysql_host,
            port=settings.mysql_port,
            user=settings.mysql_user,
            password=settings.mysql_password,
            db=settings.mysql_database,
            autocommit=True,
            minsize=1,
            maxsize=5,
        )

        logger.success("Connected to MySQL state DB")

    async def init_db(self):
        await self.connect()

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        doc_id VARCHAR(64) PRIMARY KEY,
                        source_url TEXT,
                        title TEXT,
                        domain VARCHAR(100),
                        status VARCHAR(30) DEFAULT 'queued',
                        chunk_count INT DEFAULT 0,
                        entity_count INT DEFAULT 0,
                        error_message TEXT,
                        created_at DATETIME,
                        updated_at DATETIME,
                        metadata JSON
                    )
                """)

                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id VARCHAR(64) PRIMARY KEY,
                        doc_id VARCHAR(64),
                        content LONGTEXT,
                        page_num INT,
                        chunk_index INT,
                        section_title TEXT,
                        domain VARCHAR(100),
                        drug VARCHAR(255),
                        brand VARCHAR(255),
                        embedding_stored TINYINT DEFAULT 0,
                        metadata JSON,
                        FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
                            ON DELETE CASCADE
                    )
                """)

                # Add columns if table already existed from old version
                for sql in [
                    "ALTER TABLE chunks ADD COLUMN section_title TEXT",
                    "ALTER TABLE chunks ADD COLUMN domain VARCHAR(100)",
                    "ALTER TABLE chunks ADD COLUMN drug VARCHAR(255)",
                    "ALTER TABLE chunks ADD COLUMN brand VARCHAR(255)",
                    "ALTER TABLE chunks ADD COLUMN metadata JSON",
                ]:
                    try:
                        await cur.execute(sql)
                    except Exception:
                        pass

                for sql in [
                    "CREATE INDEX idx_doc_status ON documents(status)",
                    "CREATE INDEX idx_doc_domain ON documents(domain)",
                    "CREATE INDEX idx_chunk_drug ON chunks(drug)",
                    "CREATE INDEX idx_chunk_domain ON chunks(domain)",
                ]:
                    try:
                        await cur.execute(sql)
                    except Exception:
                        pass

        logger.info("MySQL DB initialized")

    async def upsert_document(
        self,
        doc_id: str,
        source_url: str,
        title: str,
        domain: str = "general",
        metadata: dict = None,
    ):
        await self.connect()

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO documents
                    (doc_id, source_url, title, domain, status, created_at, updated_at, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    AS new
                    ON DUPLICATE KEY UPDATE
                        source_url = new.source_url,
                        title = new.title,
                        domain = new.domain,
                        status = new.status,
                        updated_at = new.updated_at,
                        metadata = new.metadata
                """, (
                    doc_id,
                    source_url,
                    title,
                    domain,
                    DocStatus.QUEUED.value,
                    now,
                    now,
                    json.dumps(metadata or {}),
                ))

    async def update_status(
        self,
        doc_id: str,
        status: DocStatus,
        chunk_count: int = None,
        entity_count: int = None,
        error_message: str = None,
    ):
        await self.connect()

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        fields = ["status = %s", "updated_at = %s"]
        values = [
            status.value if isinstance(status, DocStatus) else status,
            now,
        ]

        if chunk_count is not None:
            fields.append("chunk_count = %s")
            values.append(chunk_count)

        if entity_count is not None:
            fields.append("entity_count = %s")
            values.append(entity_count)

        if error_message is not None:
            fields.append("error_message = %s")
            values.append(error_message)

        values.append(doc_id)

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"UPDATE documents SET {', '.join(fields)} WHERE doc_id = %s",
                    values,
                )

    async def save_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        content: str,
        page_num: int,
        chunk_index: int,
        section_title: str = "",
        domain: str = "general",
        drug: str = "",
        brand: str = "",
        metadata: dict = None,
    ):
        await self.connect()

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO chunks
                    (chunk_id, doc_id, content, page_num, chunk_index,
                     section_title, domain, drug, brand, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    AS new
                    ON DUPLICATE KEY UPDATE
                        content = new.content,
                        page_num = new.page_num,
                        chunk_index = new.chunk_index,
                        section_title = new.section_title,
                        domain = new.domain,
                        drug = new.drug,
                        brand = new.brand,
                        metadata = new.metadata
                """, (
                    chunk_id,
                    doc_id,
                    content,
                    page_num,
                    chunk_index,
                    section_title,
                    domain,
                    drug,
                    brand,
                    json.dumps(metadata or {}),
                ))

    async def mark_chunk_embedded(self, chunk_id: str):
        await self.connect()

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE chunks SET embedding_stored = 1 WHERE chunk_id = %s",
                    (chunk_id,),
                )

    async def get_document(self, doc_id: str) -> Optional[Dict]:
        await self.connect()

        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    "SELECT * FROM documents WHERE doc_id = %s",
                    (doc_id,),
                )
                return await cur.fetchone()

    async def list_documents(
        self,
        status: Optional[DocStatus] = None,
        domain: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        await self.connect()

        query = "SELECT * FROM documents WHERE 1=1"
        params = []

        if status:
            query += " AND status = %s"
            params.append(status.value if isinstance(status, DocStatus) else status)

        if domain:
            query += " AND domain = %s"
            params.append(domain)

        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)

        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(query, params)
                return await cur.fetchall()

    async def list_chunks(self, limit: int = 100000) -> List[Dict]:
        await self.connect()

        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute("""
                    SELECT
                        c.chunk_id,
                        c.doc_id,
                        c.content,
                        c.page_num,
                        c.chunk_index,
                        c.section_title,
                        c.domain,
                        c.drug,
                        c.brand,
                        c.metadata,
                        d.title AS doc_title,
                        d.source_url
                    FROM chunks c
                    JOIN documents d ON d.doc_id = c.doc_id
                    LIMIT %s
                """, (limit,))

                rows = await cur.fetchall()

                for r in rows:
                    if isinstance(r.get("metadata"), str):
                        try:
                            r["metadata"] = json.loads(r["metadata"])
                        except Exception:
                            r["metadata"] = {}

                return rows

    async def get_stats(self) -> Dict:
        await self.connect()

        stats = {}

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                for status in DocStatus:
                    await cur.execute(
                        "SELECT COUNT(*) FROM documents WHERE status = %s",
                        (status.value,),
                    )
                    row = await cur.fetchone()
                    stats[status.value] = row[0]

                await cur.execute("SELECT COUNT(*) FROM chunks")
                row = await cur.fetchone()
                stats["total_chunks"] = row[0]

        return stats

    async def reset(self):
        await self.connect()

        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SET FOREIGN_KEY_CHECKS = 0")
                await cur.execute("TRUNCATE TABLE chunks")
                await cur.execute("TRUNCATE TABLE documents")
                await cur.execute("SET FOREIGN_KEY_CHECKS = 1")

        logger.warning("MySQL documents/chunks reset complete")

    async def close(self):
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            self.pool = None


_tracker: Optional[StateTracker] = None


def get_tracker() -> StateTracker:
    global _tracker

    if _tracker is None:
        _tracker = StateTracker()

    return _tracker