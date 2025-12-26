from agents.state import QueryState

MIN_AVG_SCORE = 0.25
MIN_RECALL = 0.4
MIN_RELEVANT_RATIO = 0.5
MIN_CONTEXT_TOKENS = 150
TOP_N_VALIDATE = 3


def estimate_recall(chunks: list, k: int) -> float:
    # proxy recall: assume retrieved chunks are candidate hits
    return len(chunks)/k if k else 0.0

def relevant_ratio(scores: list) -> float:
    strong = [s for s in scores if s >= MIN_AVG_SCORE]
    return len(strong)/len(scores) if scores else 0.0

def context_sufficient(chunks: list) -> bool:
    total_tokens = sum(len(c["text"].split()) for c in chunks)
    return total_tokens >= MIN_CONTEXT_TOKENS

def has_conflicts(chunks: list) -> bool:
    numbers = []
    for c in chunks:
        for token in c["text"].split():
            if token.isdigit():
                numbers.append(token)
    return len(set(numbers)) > 1


def retrieval_validation_node(state, chunk_loader):
    chunks_meta = state.get("retrieved_chunks", [])
    scores = state.get("retrieval_scores", [])
    k = state.get("top_k", 0)
    query = state["user_query"]

    # Empty retrieval
    if not chunks_meta:
        return {**state, "retrieval_valid": False, "retrieval_failure_reason": "empty_context"}

    # Weak scores
    avg_score = sum(scores) / len(scores)
    if avg_score < MIN_AVG_SCORE:
        return {**state, "retrieval_valid": False, "retrieval_failure_reason": "weak_retrieval"}

    # Recall proxy
    recall = estimate_recall(query, chunks_meta, k)
    if recall < MIN_RECALL:
        return {**state, "retrieval_valid": False, "retrieval_failure_reason": "low_recall"}

    # Partial relevance
    if relevant_ratio(scores) < MIN_RELEVANT_RATIO:
        return {**state, "retrieval_valid": False, "retrieval_failure_reason": "partial_relevance"}

    # Load limited chunk text
    validation_chunks = [
        chunk_loader.get(c["chunk_id"])
        for c in chunks_meta[:TOP_N_VALIDATE]
    ]

    # Context sufficiency
    if not context_sufficient(validation_chunks):
        return {**state, "retrieval_valid": False, "retrieval_failure_reason": "insufficient_context"}

    # Conflicting evidence
    if has_conflicts(validation_chunks):
        return {**state, "retrieval_valid": False, "retrieval_failure_reason": "conflicting_sources"}

    # Safe to answer
    return {**state, "retrieval_valid": True, "retrieval_failure_reason": None}
