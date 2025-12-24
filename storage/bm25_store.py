# bm25_store.py
import math
from collections import Counter, defaultdict
from typing import Dict, List


class BM25Store:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        self.documents = {}          # chunk_id -> token list
        self.doc_len = {}            # chunk_id -> length
        self.df = defaultdict(int)   # term -> document frequency
        self.N = 0
        self.avgdl = 0.0

    # -------------------------
    # Index construction
    # -------------------------
    def add(self, chunk_id: str, text: str):
        tokens = self._tokenize(text)

        if not tokens:
            return

        self.documents[chunk_id] = tokens
        self.doc_len[chunk_id] = len(tokens)

        for term in set(tokens):
            self.df[term] += 1

        self.N += 1
        self.avgdl = sum(self.doc_len.values()) / self.N

    # -------------------------
    # Search
    # -------------------------
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not query or self.N == 0:
            return []

        query_terms = self._tokenize(query)
        scores = {}

        for chunk_id, tokens in self.documents.items():
            tf = Counter(tokens)
            score = 0.0

            for term in query_terms:
                if term not in tf:
                    continue

                idf = self._idf(term)
                freq = tf[term]
                dl = self.doc_len[chunk_id]

                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * (freq * (self.k1 + 1)) / denom

            if score > 0:
                scores[chunk_id] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            {
                "chunk_id": cid,
                "score": float(score)
            }
            for cid, score in ranked
        ]

    # -------------------------
    # Helpers
    # -------------------------
    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def _tokenize(self, text: str) -> List[str]:
        return [
            t for t in text.lower().split()
            if t.isalnum()
        ]
