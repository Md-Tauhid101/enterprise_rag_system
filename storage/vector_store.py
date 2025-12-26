# vector_store.py
import os 
import json
import faiss
import numpy as np

class VectorStore:
    """
    Persistent multimodal vector store.
    - Text + tables → BGE embeddings (768d)
    - Images → CLIP image embeddings (512d)

    Owns:
    - FAISS indexes
    - id maps
    - disk persistence
    """
    def __init__(self, text_dim: int = 768, image_dim: int = 512, base_path: str = "./vector_store"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

        # paths
        self.text_index_path = os.path.join(base_path, "text.index")
        self.text_map_path = os.path.join(base_path, "text_id_map.json")

        self.image_index_path = os.path.join(base_path, "image.index")
        self.image_map_path = os.path.join(base_path, "image_id_map.json")

        # load or create text index
        if os.path.exists(self.text_index_path):
            self.text_index = faiss.read_index(self.text_index_path)
            with open(self.text_map_path) as f:
                self.text_id_map = json.load(f)
        else:
            self.text_index = faiss.IndexFlatIP(text_dim)
            self.text_id_map = []

        # load or create image index
        if os.path.exists(self.image_index_path):
            self.image_index = faiss.read_index(self.image_index_path)
            with open(self.image_map_path) as f:
                self.image_id_map = json.load(f)
        else:
            self.image_index = faiss.IndexFlatIP(image_dim)
            self.image_id_map = []
    
    # add methods
    def add_text(self, embedding: np.ndarray, chunk_id: str):
        # print("[DEBUG] add_text called for", chunk_id)
        self._validate_embedding(embedding, self.text_index.d)
        vec = self._normalize(embedding)
        self.text_index.add(vec)
        self.text_id_map.append(chunk_id)
    
    def add_image(self, embedding: np.ndarray, chunk_id: str):
        self._validate_embedding(embedding, self.image_index.d)
        vec = self._normalize(embedding)
        self.image_index.add(vec)
        self.image_id_map.append(chunk_id)

    # search methods
    def search_text(self, query_embedding: np.ndarray, top_k: int = 5):
        if self.text_index.ntotal == 0:
            return []
        vec = self._normalize(query_embedding)
        scores, indices = self.text_index.search(vec, top_k)

        return self._format_results(
            scores, indices, self.text_id_map
        )
    
    def search_image(self, query_embedding: np.ndarray, top_k: int = 3):
        if self.image_index.ntotal == 0:
            return []

        vec = self._normalize(query_embedding)
        scores, indices = self.image_index.search(vec, top_k)

        return self._format_results(
            scores, indices, self.image_id_map
        )
    
    # persistant
    def save(self):
        faiss.write_index(self.text_index, self.text_index_path)
        faiss.write_index(self.image_index, self.image_index_path)

        with open(self.text_map_path, "w") as f:
            json.dump(self.text_id_map, f)
        
        with open(self.image_map_path, "w") as f:
            json.dump(self.image_id_map, f)

    # helper functions
    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        vec = vec.astype("float32")
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("Zero-norm embedding")
        return (vec / norm).reshape(1, -1)

    def _validate_embedding(self, vec: np.ndarray, dim: int):
        if vec is None:
            raise ValueError("Embedding is None")
        if vec.shape[-1] != dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {dim}, got {vec.shape[-1]}"
            )

    def _format_results(self, scores, indices, id_map):
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append({
                "chunk_id": id_map[idx],
                "score": float(score)
            })
        return results