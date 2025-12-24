# hybrid_fusion.py
from typing import List, Dict

def hybrid_fusion(dense_results: List[Dict], sparse_results: List[Dict], top_k: int = 10) -> List[Dict]:
    combined = {}

    for r in dense_results + sparse_results:
        cid = r["chunk_id"]
        combined.setdefault(cid, {
            "chunk_id": cid,
            "dense_score": 0.0,
            "sparse_score": 0.0
        })
        combined[cid]["dense_score"] += r["dense_score"]
        combined[cid]["sparse_score"] += r["sparse_score"]

    fused = []
    for v in combined.values():
        fused.append({
            "chunk_id": v["chunk_id"],
            "dense_score": v["dense_score"],
            "sparse_score": v["sparse_score"],
            "score": 0.6 * v["dense_score"] + 0.4 * v["sparse_score"]
        })
    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused[:top_k]
    