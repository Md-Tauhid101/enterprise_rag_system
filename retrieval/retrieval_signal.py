# retrieval_signal.py
from typing import List, Dict
import numpy as np

def dense_retrieve_text(query_embedding, vector_index, top_k: int = 10) -> List[Dict]:
    if query_embedding is None:
        return []
    
    results = vector_index.search_text(query_embedding, top_k)
    return [
        {
            "chunk_id": r["chunk_id"],
            "dense_score": float(r["score"]),
            "sparse_score": 0.0
        }
        for r in results
    ]

def sparse_retrieve(query: str, bm25_index, top_k: int = 0) -> List[Dict]:
    results = bm25_index.search(query, top_k)
    return [
        {
            "chunk_id": r["chunk_id"],
            "dense_score": 0.0,
            "sparse_score": float(r["score"])
        }
        for r in results
    ]

def dense_retrieve_image(
    query_embedding,
    vector_store,
    top_k: int = 5
):
    """
    Dense retrieval over IMAGE embeddings only.
    Used ONLY for image queries.
    """
    if query_embedding is None:
        return []

    results = vector_store.search_image(
        query_embedding=query_embedding,
        top_k=top_k
    )

    return [
        {
            "chunk_id": r["chunk_id"],
            "dense_score": float(r["score"]),
            "sparse_score": 0.0,
            "modality": "image"
        }
        for r in results
    ]
