# # vector_store.py
import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.id_map = []

    def add(self, embedding: np.ndarray, metadata: dict):
        assert "chunk_id" in metadata, "chunk_id is required"

        vec = embedding.astype("float32")
        vec = vec / np.linalg.norm(vec)
        vec = vec.reshape(1, -1)

        self.index.add(vec)
        self.id_map.append(metadata["chunk_id"])

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if self.index.ntotal == 0:
            return []

        vec = query_embedding.astype("float32")
        vec = vec / np.linalg.norm(vec)
        vec = vec.reshape(1, -1)

        scores, indices = self.index.search(vec, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append({
                "chunk_id": self.id_map[idx],
                "score": float(score)
            })
        return results

    def get_all_chunk_ids(self):
        return set(self.id_map)