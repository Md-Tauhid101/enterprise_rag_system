from agents.state import QueryState
from retrieval.hybrid_fusion import hybrid_fusion
from retrieval.retrieval_signal import dense_retrieve_text, sparse_retrieve
from storage.vector_store import VectorStore
from storage.bm25_store import BM25Store

TOP_K = 10

def retrieve_node(state: QueryState) -> QueryState:
    query = state["user_query"]
    query_embedding = state["query_embedding"]

    dense_results = dense_retrieve_text(query_embedding=query_embedding, vector_index=VectorStore,top_k=TOP_K)

    sparse_results = sparse_retrieve(query=query, bm25_index=BM25Store, top_k=TOP_K)

    fused = hybrid_fusion(dense_results=dense_results, sparse_results=sparse_results, top_k=TOP_K)

    retrieved_chunks = []
    retrieval_scores = []

    for r in fused:
        retrieved_chunks.append({
            "chunk_id": r["chunk_id"],
            "source": "hybrid"
        })
        retrieval_scores.append(float(r["score"]))    

    return {
        **state,
        "retrieved_chunks": retrieved_chunks,
        "retrieval_scores": retrieval_scores,
        "top_k": TOP_K
    }
