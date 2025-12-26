# state.py
from typing import TypedDict, Optional, List, Dict

class RetrievedChunk(TypedDict):
    chunk_id: str
    text: str
    source: str

class QueryState(TypedDict):
    user_query: str

    # intent understanding
    intent: Optional[str]
    intent_confidence: Optional[float]
    intent_reason: Optional[str]
    should_refuse: bool

    # rewrite
    rewritten_queries: Optional[Dict[str, object]]
    rewrite_risk: Optional[Dict[str, bool]]

    # retrieval output
    retrieved_chunks: Optional[List[RetrievedChunk]]
    retrieval_scores: Optional[List[float]]
    top_k: Optional[int]

    # validation result
    retrieval_valid: Optional[bool]
    retrieval_failure_reason: Optional[str]

    # query embedding
    query_embedding: Optional[List[float]]

    # answer validation
    answer_text: Optional[str]
    answer_citations: List[str]
    answer_supported: bool