"""pgvector store — the production path when you already run Postgres.

Why a team picks pgvector over a dedicated vector DB: you keep vectors next to
your relational data, reuse existing backup/HA/permissions, and can filter on
metadata with plain SQL. We create an HNSW index for approximate-NN search at
scale; for small corpora a flat scan is fine and exact.

Uses psycopg (v3). The `<=>` operator is pgvector's cosine distance.
"""
from __future__ import annotations

from ..config import settings
from ..logging_config import get_logger
from ..models import Chunk, ScoredChunk

logger = get_logger(__name__)


class PgVectorStore:
    def __init__(self) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self.conn = psycopg.connect(settings.pg_dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        dim = settings.embedding_dim
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {settings.pg_table} (
                chunk_id  TEXT PRIMARY KEY,
                doc_id    TEXT NOT NULL,
                title     TEXT NOT NULL,
                source    TEXT NOT NULL,
                category  TEXT NOT NULL,
                position  INT  NOT NULL,
                content   TEXT NOT NULL,
                embedding vector({dim}) NOT NULL
            )
            """
        )
        # HNSW = fast approximate search; cosine ops to match normalized vectors.
        self.conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {settings.pg_table}_emb_idx
            ON {settings.pg_table}
            USING hnsw (embedding vector_cosine_ops)
            """
        )
        self.conn.execute(
            f"CREATE INDEX IF NOT EXISTS {settings.pg_table}_cat_idx "
            f"ON {settings.pg_table} (category)"
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        import numpy as np

        rows = [
            (
                c.chunk_id, c.doc_id, c.title, c.source, c.category,
                c.position, c.text, np.array(emb, dtype=np.float32),
            )
            for c, emb in zip(chunks, embeddings)
        ]
        with self.conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {settings.pg_table}
                    (chunk_id, doc_id, title, source, category, position,
                     content, embedding)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding
                """,
                rows,
            )
        logger.info("pgvector upsert", extra={"extra": {"n": len(rows)}})

    def search(self, query_embedding, top_k, category=None) -> list[ScoredChunk]:
        import numpy as np

        vec = np.array(query_embedding, dtype=np.float32)
        params: list = [vec]
        where = ""
        if category:
            where = "WHERE category = %s"
            params.append(category)
        params.append(top_k)
        sql = f"""
            SELECT chunk_id, doc_id, title, source, category, position, content,
                   1 - (embedding <=> %s) AS similarity
            FROM {settings.pg_table}
            {where}
            ORDER BY embedding <=> %s
            LIMIT %s
        """
        # The ORDER BY needs the vector again; insert it before LIMIT.
        params = [vec] + ([category] if category else []) + [vec, top_k]
        rows = self.conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            out.append(
                ScoredChunk(
                    chunk=Chunk(
                        chunk_id=r[0], doc_id=r[1], title=r[2], source=r[3],
                        category=r[4], position=r[5], text=r[6],
                    ),
                    score=float(r[7]),
                    retriever="dense",
                )
            )
        return out

    def count(self) -> int:
        return self.conn.execute(
            f"SELECT COUNT(*) FROM {settings.pg_table}"
        ).fetchone()[0]
