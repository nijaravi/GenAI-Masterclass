"""
db_check.py — Inspect the PGVector database from the command line.

Usage:
    python db_check.py                    # show table stats + sample rows
    python db_check.py --query "return policy" --persona customer
    python db_check.py --query "billing" --persona client
    python db_check.py --query "OAuth" --persona developer
    python db_check.py --show-index       # show index details

Good for live interview demos to prove the vector search is real SQL.
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from backend.config import PGVECTOR_CONNECTION_STRING, DEMO_MODE


def get_conn():
    import psycopg2
    from pgvector.psycopg2 import register_vector
    conn = psycopg2.connect(PGVECTOR_CONNECTION_STRING)
    register_vector(conn)
    return conn


def cmd_stats(conn):
    cur = conn.cursor()
    print("\n── Table stats ──────────────────────────────────────")
    cur.execute("SELECT persona, COUNT(*) FROM walmart_documents GROUP BY persona ORDER BY persona;")
    for row in cur.fetchall():
        print(f"  {row[0]:<12}  {row[1]} documents")

    cur.execute("SELECT COUNT(*) FROM walmart_documents;")
    total = cur.fetchone()[0]
    print(f"  {'TOTAL':<12}  {total} documents")

    print("\n── Sample rows (id, persona, category) ─────────────")
    cur.execute("SELECT id, persona, category, LEFT(content, 60) FROM walmart_documents ORDER BY persona, id LIMIT 15;")
    for row in cur.fetchall():
        print(f"  {row[0]:<6}  {row[1]:<12}  {row[2]:<16}  {row[3]}...")
    cur.close()


def cmd_query(conn, query_text: str, persona: str, top_k: int = 3):
    """Run a real pgvector cosine similarity query and show the SQL + results."""
    from backend.utils.vector_store import PGVectorStore
    store = PGVectorStore(PGVECTOR_CONNECTION_STRING)
    query_vec = store._embed(query_text)

    print(f"\n── Query: '{query_text}'  |  persona: {persona}  |  top_k: {top_k} ──")
    print("\n  SQL executed:")
    print(f"""
    SELECT id, persona, category, content,
           1 - (embedding <=> $query_vec) AS cosine_similarity
    FROM   walmart_documents
    WHERE  persona = '{persona}'
    ORDER  BY embedding <=> $query_vec
    LIMIT  {top_k};
    """)

    results = store.query(query_text, top_k=top_k, filter_persona=persona)
    print("  Results:")
    for i, doc in enumerate(results, 1):
        score = doc.get("score", 0)
        print(f"\n  [{i}] score={score:.4f}  id={doc['id']}  category={doc['metadata']['category']}")
        print(f"      {doc['text'][:140]}...")

    store.close()


def cmd_show_index(conn):
    cur = conn.cursor()
    print("\n── pgvector Index Info ──────────────────────────────")
    cur.execute("""
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE tablename = 'walmart_documents';
    """)
    for row in cur.fetchall():
        print(f"  {row[0]}")
        print(f"    {row[1]}")

    print("\n── Extension version ────────────────────────────────")
    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
    row = cur.fetchone()
    if row:
        print(f"  pgvector v{row[0]}")
    cur.close()


def main():
    if DEMO_MODE:
        print("⚠️  DEMO_MODE=true — this script requires a real Postgres connection.")
        print("   Set DEMO_MODE=false and PGVECTOR_CONNECTION_STRING in .env, then re-run.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Inspect the PGVector database")
    parser.add_argument("--query", type=str, help="Run a similarity search query")
    parser.add_argument("--persona", type=str, default="customer",
                        choices=["customer", "client", "developer"],
                        help="Persona filter for the query")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--show-index", action="store_true")
    args = parser.parse_args()

    try:
        conn = get_conn()
        print(f"✅ Connected to: {PGVECTOR_CONNECTION_STRING.split('@')[-1]}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nIs Postgres running?  Try:  docker compose up -d")
        sys.exit(1)

    if args.query:
        cmd_query(conn, args.query, args.persona, args.top_k)
    elif args.show_index:
        cmd_show_index(conn)
    else:
        cmd_stats(conn)
        cmd_show_index(conn)

    conn.close()


if __name__ == "__main__":
    main()
