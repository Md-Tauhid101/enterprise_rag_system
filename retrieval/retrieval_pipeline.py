from retrieval.retrieval_signal import dense_retrieve_text, sparse_retrieve
from retrieval.hybrid_fusion import hybrid_fusion

def retrieval_pipeline(query: str, query_embedding, vector_index, bm25_index, top_k: int = 10):
    dense = dense_retrieve_text(query_embedding=query_embedding, vector_index=vector_index, top_k=top_k)
    sparse = sparse_retrieve(query=query, bm25_index=bm25_index, top_k=top_k)

    hybrid = hybrid_fusion(dense_results=dense, sparse_results=sparse, top_k=top_k)

    return {
        "query": query,
        "retrieval_results": hybrid
    }

if __name__ == "__main__":
    from storage.vector_store import VectorStore
    from storage.bm25_store import BM25Store
    from retrieval.chunks_retriever import ChunksRetriever
    from ingestion.embed_func import embed_text
    from retrieval.retrieval_pipeline import retrieval_pipeline
    from config import DB_CONFIG
    import psycopg2

    # --------------------------------------------------
    # 1. DB connection (READ-ONLY)
    # --------------------------------------------------
    conn = psycopg2.connect(**DB_CONFIG)
    chunk_store = ChunksRetriever(conn)

    # --------------------------------------------------
    # 2. Initialize retrieval stores
    # --------------------------------------------------
    EMBED_DIM = 768
    vector_store = VectorStore(text_dim=EMBED_DIM)
    bm25_store = BM25Store()

    # --------------------------------------------------
    # 3. Load chunks from Postgres
    # --------------------------------------------------
    print("Loading chunks from Postgres...")

    chunks = chunk_store.get_all_chunks()  
    # expected: List[{"chunk_id": str, "text": str}]

    if not chunks:
        raise RuntimeError("No chunks found in database.")

    print(f"Loaded {len(chunks)} chunks")

    # --------------------------------------------------
    # 4. Index chunks (REAL embeddings)
    # --------------------------------------------------
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        text = chunk["text"]

        embedding = embed_text(text)

        vector_store.add_text(
            embedding=embedding,
            chunk_id=chunk_id
        )

        bm25_store.add(
            chunk_id=chunk_id,
            text=text
        )

    print("Indexing complete")

    # --------------------------------------------------
    # 5. Query
    # --------------------------------------------------
    query = "what are the skills?"
    query_embedding = embed_text(query)

    # --------------------------------------------------
    # 6. Run STEP-4
    # --------------------------------------------------
    output = retrieval_pipeline(
        query=query,
        query_embedding=query_embedding,
        vector_index=vector_store,
        bm25_index=bm25_store,
        top_k=5
    )

    # --------------------------------------------------
    # 7. Inspect output (SIGNALS ONLY)
    # --------------------------------------------------
    print("\nSTEP-4 OUTPUT (RETRIEVAL SIGNALS ONLY)")

    for r in output["retrieval_results"]:
        print(
            f"chunk_id={r['chunk_id']} | "
            f"dense={r['dense_score']:.4f} | "
            f"sparse={r['sparse_score']:.4f} | "
            f"fused={r['score']:.4f}"
        )

