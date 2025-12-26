# chunk_retriever.py
import psycopg2
from psycopg2.extras import RealDictCursor


class ChunksRetriever:
    """
    Dumb data access layer for chunk storage.
    Fetches ground-truth chunks from Postgres.
    """

    def __init__(self, conn):
        self.conn = conn

    # --------------------------------------------------
    # Fetch ONE chunk (used in Step-5 / Step-6)
    # --------------------------------------------------
    def get(self, chunk_id: str) -> dict:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    chunk_id,
                    document_id,
                    chunk_index,
                    page_number,
                    cleaned_text,
                    created_at
                FROM chunks
                WHERE chunk_id = %s
                """,
                (chunk_id,)
            )

            row = cur.fetchone()

        if row is None:
            raise ValueError(f"Chunk not found: {chunk_id}")

        return {
            "chunk_id": str(row["chunk_id"]),
            "document_id": str(row["document_id"]),
            "chunk_index": row["chunk_index"],
            "page_number": row["page_number"],
            "text": row["cleaned_text"],
            "created_at": row["created_at"]
        }

    # --------------------------------------------------
    # Fetch ALL chunks (used for indexing / Step-4 setup)
    # --------------------------------------------------
    def get_all_chunks(self):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    chunk_id,
                    cleaned_text
                FROM chunks
                ORDER BY created_at ASC
                """
            )

            rows = cur.fetchall()

        return [
            {
                "chunk_id": str(r["chunk_id"]),
                "text": r["cleaned_text"]
            }
            for r in rows
        ]
