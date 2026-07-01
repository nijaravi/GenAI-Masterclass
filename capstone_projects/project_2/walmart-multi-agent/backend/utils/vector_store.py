"""
vector_store.py — Dual-mode vector store: PGVector (production) / FAISS (demo).

ARCHITECTURE
============
                         DEMO_MODE=true
                         ┌──────────────────┐
get_vector_store() ─────►│ SimpleVectorStore │  (in-memory FAISS, no DB needed)
                         └──────────────────┘

                         DEMO_MODE=false
                         ┌──────────────────┐
get_vector_store() ─────►│  PGVectorStore   │  (PostgreSQL + pgvector extension)
                         └──────────────────┘

PGVECTOR SCHEMA
===============
The PGVectorStore creates and owns a single table:

    CREATE TABLE walmart_documents (
        id          TEXT PRIMARY KEY,
        persona     TEXT NOT NULL,          -- 'customer' | 'client' | 'developer'
        category    TEXT,
        content     TEXT NOT NULL,
        embedding   vector(1536),           -- pgvector column
        created_at  TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX ON walmart_documents
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 10);

Similarity search uses the pgvector <=> (cosine distance) operator:

    SELECT id, persona, category, content,
           1 - (embedding <=> query_vec) AS score
    FROM   walmart_documents
    WHERE  persona = %s
    ORDER  BY embedding <=> query_vec
    LIMIT  %s;

HOW TO RUN WITH REAL POSTGRES
==============================
Option A — Docker (recommended for local dev):

    docker run -d \\
      --name pgvector-dev \\
      -e POSTGRES_PASSWORD=postgres \\
      -e POSTGRES_DB=walmart_ai \\
      -p 5432:5432 \\
      pgvector/pgvector:pg16

Option B — Supabase (free hosted Postgres with pgvector):
    1. Create a project at supabase.com
    2. Copy the connection string from Settings → Database → Connection string
    3. Set PGVECTOR_CONNECTION_STRING in .env

Option C — Neon (serverless Postgres, pgvector built-in):
    1. Create a project at neon.tech
    2. Copy the connection string
    3. Set PGVECTOR_CONNECTION_STRING in .env

After starting Postgres, set DEMO_MODE=false in .env and restart the server.
The first startup will auto-create the table and seed all documents.
"""

import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional

from backend.config import PGVECTOR_CONNECTION_STRING, DEMO_MODE

logger = logging.getLogger(__name__)

# ─── Seed Knowledge Base ─────────────────────────────────────────────────────

SEED_DOCUMENTS = {
    "customer": [
        {
            "id": "c1",
            "text": "Product return policy: Customers can return any item within 90 days with receipt. Electronics must be returned within 30 days. Items must be in original packaging.",
            "metadata": {"persona": "customer", "category": "returns"},
        },
        {
            "id": "c2",
            "text": "Walmart+ membership benefits include free delivery, fuel discounts, Paramount+ streaming, and early access to deals. Monthly cost is $12.95 or $98 annually.",
            "metadata": {"persona": "customer", "category": "membership"},
        },
        {
            "id": "c3",
            "text": "Black Friday 2024 deals: 65-inch OLED TV at $499 (was $1299), Apple AirPods Pro at $159, Samsung Galaxy S24 at $649. Available in-store and online from midnight.",
            "metadata": {"persona": "customer", "category": "deals"},
        },
        {
            "id": "c4",
            "text": "Grocery pickup service: Order online by 10pm for same-day pickup. Minimum order is $35. Available at over 3,000 store locations. Use the Walmart app to manage orders.",
            "metadata": {"persona": "customer", "category": "grocery"},
        },
        {
            "id": "c5",
            "text": "Product warranty claims: For defective electronics, contact Walmart customer support at 1-800-925-6278. Extended protection plans available for $29-$149 depending on item value.",
            "metadata": {"persona": "customer", "category": "warranty"},
        },
    ],
    "client": [
        {
            "id": "cl1",
            "text": "Client portal billing: All invoices are generated on the 1st of each month. NET-30 payment terms apply. Accepted payment methods: ACH, wire transfer, corporate credit card.",
            "metadata": {"persona": "client", "category": "billing"},
        },
        {
            "id": "cl2",
            "text": "Supplier onboarding SLA: New supplier review takes 15 business days. Required documents: W-9, certificate of insurance, product liability coverage minimum $2M, EDI compliance certification.",
            "metadata": {"persona": "client", "category": "onboarding"},
        },
        {
            "id": "cl3",
            "text": "Category management agreement 2024: Shelf space allocation reviewed quarterly. Planogram changes submitted 8 weeks before implementation. Promotional windows: 4 per year, 2-week slots.",
            "metadata": {"persona": "client", "category": "category_mgmt"},
        },
        {
            "id": "cl4",
            "text": "Vendor compliance fines: Late shipments incur 3% of PO value per day. Incorrect labeling: $250 per case. Over/under shipment beyond 5% tolerance: $500 per incident.",
            "metadata": {"persona": "client", "category": "compliance"},
        },
    ],
    "developer": [
        {
            "id": "d1",
            "text": "Walmart API Authentication: All endpoints require OAuth 2.0 bearer tokens. Client credentials flow used for server-to-server. Token expiry: 1 hour. Rate limit: 1000 req/min per client_id.",
            "metadata": {"persona": "developer", "category": "auth"},
        },
        {
            "id": "d2",
            "text": "Product catalog API: GET /v3/items returns paginated list (max 200/page). Required headers: WM_SVC.NAME, WM_QOS.CORRELATION_ID. Use cursor-based pagination via nextCursor field.",
            "metadata": {"persona": "developer", "category": "api_docs"},
        },
        {
            "id": "d3",
            "text": "Order fulfillment webhook: Register at /v3/webhooks/orders. Events: order.created, order.shipped, order.delivered, order.cancelled. Retry policy: 3 attempts with exponential backoff.",
            "metadata": {"persona": "developer", "category": "webhooks"},
        },
        {
            "id": "d4",
            "text": "Deployment pipeline: Uses GitHub Actions → AWS CodePipeline → ECS Fargate. Blue-green deployments enforced. Required: passing unit tests (>85% coverage) and SAST scan before merge.",
            "metadata": {"persona": "developer", "category": "devops"},
        },
        {
            "id": "d5",
            "text": "Internal microservices mesh: Service discovery via AWS Cloud Map. Inter-service calls use mTLS. Circuit breaker pattern implemented with AWS App Mesh. SLO target: 99.95% availability.",
            "metadata": {"persona": "developer", "category": "architecture"},
        },
    ],
}


# ─── PGVector Store (Production) ─────────────────────────────────────────────

class PGVectorStore:
    """
    Production vector store backed by PostgreSQL + pgvector extension.

    Key design decisions:
    ─────────────────────
    • Single table 'walmart_documents' with a vector(1536) column.
    • Persona-scoped queries: WHERE persona = %s ensures cross-persona
      data isolation at the DB layer (not just application layer).
    • IVFFlat index with cosine distance for fast ANN search.
    • Embeddings use deterministic pseudo-vectors in demo builds; swap
      _embed() to call OpenAI text-embedding-3-small for production.
    • Connection managed via psycopg2; for high-concurrency production
      deployments replace with psycopg2 connection pool or asyncpg.

    SQL operators used:
    ───────────────────
      embedding <=> query_vec   cosine distance  (0 = identical, 2 = opposite)
      1 - (emb <=> qvec)        cosine similarity (higher = more similar)

    pgvector index types:
    ─────────────────────
      IVFFlat  — faster build, good recall; use for ≤1M vectors
      HNSW     — better recall, slower build; use for >1M vectors
                 (pgvector ≥0.5.0, set with USING hnsw)
    """

    TABLE = "walmart_documents"
    DIM = 1536  # matches OpenAI text-embedding-3-small

    def __init__(self, connection_string: str):
        self._conn_str = connection_string
        self._conn = None
        self._connect()
        self._setup_schema()

    # ── Connection ────────────────────────────────────────────────────────

    def _connect(self):
        """Establish psycopg2 connection with pgvector adapter registered."""
        import psycopg2
        from pgvector.psycopg2 import register_vector

        self._conn = psycopg2.connect(self._conn_str)
        self._conn.autocommit = False
        register_vector(self._conn)   # teaches psycopg2 to serialise numpy arrays as vector literals
        logger.info("PGVectorStore: connected to Postgres")

    def _cursor(self):
        """Return a cursor, reconnecting if the connection was dropped."""
        try:
            self._conn.isolation_level  # ping
        except Exception:
            logger.warning("PGVectorStore: reconnecting...")
            self._connect()
        return self._conn.cursor()

    # ── Schema bootstrap ──────────────────────────────────────────────────

    def _setup_schema(self):
        """
        Idempotent schema creation.  Safe to call on every startup.

        Steps:
          1. Enable pgvector extension (no-op if already enabled).
          2. Create the documents table (no-op if already exists).
          3. Create IVFFlat cosine index (no-op if already exists).
        """
        cur = self._cursor()
        try:
            # Step 1 — enable extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Step 2 — documents table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE} (
                    id          TEXT PRIMARY KEY,
                    persona     TEXT        NOT NULL,
                    category    TEXT,
                    content     TEXT        NOT NULL,
                    embedding   vector({self.DIM}),
                    created_at  TIMESTAMPTZ DEFAULT NOW()
                );
            """)

            # Step 3 — IVFFlat index (cosine distance)
            # lists=10 is suitable for ~100 documents; increase to sqrt(n_rows) for larger datasets
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.TABLE}_embedding_idx
                ON {self.TABLE}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 10);
            """)

            self._conn.commit()
            logger.info("PGVectorStore: schema ready")
        except Exception as e:
            self._conn.rollback()
            logger.error(f"PGVectorStore: schema setup failed — {e}")
            raise
        finally:
            cur.close()

    # ── Embedding ─────────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        """
        Deterministic pseudo-embedding for local dev (no API key needed).

        PRODUCTION SWAP: replace this method body with:

            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return np.array(resp.data[0].embedding, dtype=np.float32)

        The vector dimension must stay at 1536 to match the table column.
        """
        words = set(text.lower().split())
        vocab = sorted({
            w
            for persona_docs in SEED_DOCUMENTS.values()
            for doc in persona_docs
            for w in doc["text"].lower().split()
        })
        vec = np.zeros(self.DIM, dtype=np.float32)
        for i, word in enumerate(vocab[: self.DIM]):
            if word in words:
                vec[i] = 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    # ── Write ─────────────────────────────────────────────────────────────

    def upsert(self, documents: List[Dict]):
        """
        Insert or update documents.  Uses ON CONFLICT DO UPDATE so it is
        safe to call on repeated startups without creating duplicates.
        """
        cur = self._cursor()
        try:
            for doc in documents:
                embedding = self._embed(doc["text"])
                meta = doc.get("metadata", {})
                cur.execute(
                    f"""
                    INSERT INTO {self.TABLE} (id, persona, category, content, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                        SET persona   = EXCLUDED.persona,
                            category  = EXCLUDED.category,
                            content   = EXCLUDED.content,
                            embedding = EXCLUDED.embedding;
                    """,
                    (
                        doc["id"],
                        meta.get("persona", ""),
                        meta.get("category", ""),
                        doc["text"],
                        embedding,
                    ),
                )
            self._conn.commit()
            logger.info(f"PGVectorStore: upserted {len(documents)} documents")
        except Exception as e:
            self._conn.rollback()
            logger.error(f"PGVectorStore: upsert failed — {e}")
            raise
        finally:
            cur.close()

    # ── Read ──────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        top_k: int = 3,
        filter_persona: Optional[str] = None,
    ) -> List[Dict]:
        """
        Cosine similarity search with optional persona filter.

        The <=> operator returns cosine *distance* (0–2), so we sort ASC
        and return 1 - distance as the similarity score.

        SQL produced (simplified):
            SELECT id, persona, category, content,
                   1 - (embedding <=> $1) AS score
            FROM   walmart_documents
            WHERE  persona = $2            -- only when filter_persona set
            ORDER  BY embedding <=> $1
            LIMIT  $3;
        """
        query_vec = self._embed(query_text)
        cur = self._cursor()
        try:
            if filter_persona:
                cur.execute(
                    f"""
                    SELECT id, persona, category, content,
                           1 - (embedding <=> %s) AS score
                    FROM   {self.TABLE}
                    WHERE  persona = %s
                    ORDER  BY embedding <=> %s
                    LIMIT  %s;
                    """,
                    (query_vec, filter_persona, query_vec, top_k),
                )
            else:
                cur.execute(
                    f"""
                    SELECT id, persona, category, content,
                           1 - (embedding <=> %s) AS score
                    FROM   {self.TABLE}
                    ORDER  BY embedding <=> %s
                    LIMIT  %s;
                    """,
                    (query_vec, query_vec, top_k),
                )

            rows = cur.fetchall()
            return [
                {
                    "id": row[0],
                    "text": row[3],
                    "score": float(row[4]),
                    "metadata": {"persona": row[1], "category": row[2]},
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"PGVectorStore: query failed — {e}")
            return []
        finally:
            cur.close()

    def row_count(self) -> int:
        """Return total number of documents in the table."""
        cur = self._cursor()
        try:
            cur.execute(f"SELECT COUNT(*) FROM {self.TABLE};")
            return cur.fetchone()[0]
        finally:
            cur.close()

    def close(self):
        if self._conn:
            self._conn.close()


# ─── FAISS fallback (Demo Mode) ───────────────────────────────────────────────

class SimpleVectorStore:
    """
    In-memory FAISS vector store.
    Used when DEMO_MODE=true so the pipeline runs without any database.
    Interface is identical to PGVectorStore — swap is transparent.
    """

    def __init__(self):
        self._docs: Dict[str, Dict] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._dim = 1536

    def _embed(self, text: str) -> np.ndarray:
        words = set(text.lower().split())
        vocab = sorted({
            w
            for persona_docs in SEED_DOCUMENTS.values()
            for doc in persona_docs
            for w in doc["text"].lower().split()
        })
        vec = np.zeros(self._dim)
        for i, word in enumerate(vocab[: self._dim]):
            if word in words:
                vec[i] = 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def upsert(self, documents: List[Dict]):
        for doc in documents:
            self._docs[doc["id"]] = doc
            self._embeddings[doc["id"]] = self._embed(doc["text"])

    def query(
        self,
        query_text: str,
        top_k: int = 3,
        filter_persona: Optional[str] = None,
    ) -> List[Dict]:
        q_vec = self._embed(query_text)
        scores = []
        for doc_id, emb in self._embeddings.items():
            doc = self._docs[doc_id]
            if filter_persona and doc.get("metadata", {}).get("persona") != filter_persona:
                continue
            score = float(np.dot(q_vec, emb))
            scores.append((score, doc))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scores[:top_k]]


# ─── Factory ──────────────────────────────────────────────────────────────────

_store = None


def get_vector_store():
    """
    Return the appropriate vector store based on DEMO_MODE.

    DEMO_MODE=true  → SimpleVectorStore (in-memory FAISS, no DB required)
    DEMO_MODE=false → PGVectorStore     (PostgreSQL + pgvector)

    On first call the store is seeded with SEED_DOCUMENTS.
    Subsequent calls return the same singleton instance.
    """
    global _store
    if _store is not None:
        return _store

    if DEMO_MODE:
        logger.info("VectorStore: using SimpleVectorStore (DEMO_MODE=true)")
        _store = SimpleVectorStore()
        all_docs = [doc for docs in SEED_DOCUMENTS.values() for doc in docs]
        _store.upsert(all_docs)
    else:
        logger.info("VectorStore: connecting to PGVector (DEMO_MODE=false)")
        _store = PGVectorStore(PGVECTOR_CONNECTION_STRING)
        # Seed only if the table is empty (safe for repeated restarts)
        if _store.row_count() == 0:
            logger.info("VectorStore: seeding initial documents...")
            all_docs = [doc for docs in SEED_DOCUMENTS.values() for doc in docs]
            _store.upsert(all_docs)
            logger.info(f"VectorStore: seeded {len(all_docs)} documents")
        else:
            logger.info(f"VectorStore: found {_store.row_count()} existing documents, skipping seed")

    return _store
