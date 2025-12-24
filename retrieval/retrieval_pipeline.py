from retrieval.retrieval_signal import dense_retrieve, sparse_retrieve
from retrieval.hybrid_fusion import hybrid_fusion

def retrieval_pipeline(query: str, query_embedding, vector_index, bm25_index, top_k: int = 10):
    dense = dense_retrieve(query_embedding=query_embedding, vector_index=vector_index, top_k=top_k)
    sparse = sparse_retrieve(query=query, bm25_index=bm25_index, top_k=top_k)

    hybrid = hybrid_fusion(dense_results=dense, sparse_results=sparse, top_k=top_k)

    return {
        "query": query,
        "retrieval_results": hybrid
    }

if __name__ == "__main__":
    import numpy as np

    from storage.vector_store import VectorStore
    from storage.bm25_store import BM25Store

    # -----------------------------
    # 1. Mock corpus (chunks)
    # -----------------------------
    chunks = {
        "c1": "Section 17.4.2 describes the reimbursement policy for employees.",
        "c2": "This document explains leave policy and attendance rules.",
        "c3": "ISO-9001 compliance requirements are listed here.",
        "c4": "Reimbursement applies only to travel expenses.",
        "c5": "Company culture and values are outlined.",
        "c6": "what are the skill sets?"
    }

    EMBED_DIM = 8
    np.random.seed(42)

    # -----------------------------
    # 2. Initialize stores
    # -----------------------------
    vector_store = VectorStore(dim=EMBED_DIM)
    bm25_store = BM25Store()

    # -----------------------------
    # 3. Index data
    # -----------------------------
    for chunk_id, text in chunks.items():
        # fake embedding (ONLY for testing Step-4)
        embedding = np.random.rand(EMBED_DIM)

        vector_store.add(
            embedding=embedding,
            metadata={"chunk_id": chunk_id}
        )

        bm25_store.add(
            chunk_id=chunk_id,
            text=text
        )

    print("Indexed chunks:", vector_store.get_all_chunk_ids())

    # -----------------------------
    # 4. Query
    # -----------------------------
    query = "What does section 17.4.2 say?"

    # fake query embedding
    query_embedding = np.random.rand(EMBED_DIM)

    # -----------------------------
    # 5. Run STEP-4
    # -----------------------------
    output = retrieval_pipeline(
        query=query,
        query_embedding=query_embedding,
        vector_index=vector_store,
        bm25_index=bm25_store,
        top_k=5
    )

    # -----------------------------
    # 6. Inspect output
    # -----------------------------
    print("\nSTEP-4 OUTPUT (RETRIEVAL SIGNALS ONLY)")
    for r in output["retrieval_results"]:
        print(
            f"chunk_id={r['chunk_id']} | "
            f"dense={r['dense_score']:.4f} | "
            f"sparse={r['sparse_score']:.4f} | "
            f"fused={r['score']:.4f}"
        )
