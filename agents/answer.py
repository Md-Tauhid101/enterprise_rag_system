# agents/answer.py

from agents.state import QueryState
from utils.llm import get_llm
from retrieval.chunks_retriever import ChunksRetriever
from langchain_core.messages import SystemMessage, HumanMessage

MAX_CONTEXT_CHARS = 4000

def answer_generation_node(state: QueryState, chunk_retriever: ChunksRetriever) -> QueryState:
    retrieved = state.get("retrieved_chunks", [])

    # HARD STOP - no evidence
    if not retrieved:
        return {
            "answer_text": None,
            "answer_citations": [],
            "answer_supported": False
        }
    
    # Load ground-truth text from postgres
    contexts = []
    used_chunk_ids = []

    total_chars = 0
    for item in retrieved:
        chunk = chunk_retriever.get(item["chunk_id"])
        text = chunk["text"]

        if not text:
            continue

        if total_chars + len(text) > MAX_CONTEXT_CHARS:
            break

        contexts.append(f"[{chunk['chunk_id']}] {text}")
        used_chunk_ids.append(chunk["chunk_id"])
        total_chars += len(text)

    if not contexts:
        return {
            **state,
            "answer_text": None,
            "answer_citations": [],
            "answer_supported": False
        }
    
    context_block = "\n\n".join(contexts)

    # LLM
    system_prompt = """
    You are an evidence-bound answer generator.

    RULES:
    - Use ONLY the provided context.
    - Do NOT use external knowledge.
    - DO NOT infer missing details.
    - If the context does not answer the question, respond EXACTLY with:
        "INSUFFICIENT_EVIDENCE"
    - Cite chunk IDs you used.
    """

    user_prompt = f"""
    Question:
    {state["user_query"]}

    Context:
    {context_block}

    Answer format:
    - Answer (1-3 sentences max)
    - Evidence: [chunk_id, chunk_id]
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    llm = get_llm()
    response = llm.invoke(messages).content.strip()

    # Validation gate
    if "INSUFFICIENT_EVIDENCE" in response:
        return {
            **state,
            "answer_text": None,
            "answer_citations": [],
            "answer_supported": False
        }
    
    return {
        **state,
        "answer_text": response,
        "answer_citations": used_chunk_ids,
        "answer_supported": True
    }

# agents/answer.py

# agents/answer.py

if __name__ == "__main__":
    """
    Smoke test for Step-7 Answer Generation
    + Direct Postgres chunk inspection
    """

    import psycopg2
    from psycopg2.extras import RealDictCursor

    from retrieval.chunks_retriever import ChunksRetriever
    from agents.answer import answer_generation_node
    from config import DB_CONFIG

    # --------------------------------------------------
    # 1. DB connection
    # --------------------------------------------------
    conn = psycopg2.connect(**DB_CONFIG)

    def find_chunk_by_keyword(rows, keyword: str):
        for row in rows:
            if keyword.lower() in row["cleaned_text"].lower():
                return row["chunk_id"]
        return None


    # --------------------------------------------------
    # 2. RAW DB SANITY CHECK (CRITICAL)
    # --------------------------------------------------
    print("\n=== RAW CHUNK SAMPLE FROM DB ===")

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT chunk_id, cleaned_text
            FROM chunks
            LIMIT 5;
        """)
        rows = cur.fetchall()

    for row in rows:
        print(f"\nChunk ID: {row['chunk_id']}")
        print(f"Text Preview: {row['cleaned_text'][:200]}")

    # HARD ASSERTION — DO NOT IGNORE
    assert rows, "Chunks table is EMPTY. Ingestion is broken."

    # --------------------------------------------------
    # 3. Initialize retriever
    # --------------------------------------------------
    chunk_retriever = ChunksRetriever(conn)

    # --------------------------------------------------
    # 4. CASE 1 — Supported question
    # --------------------------------------------------
    skill_chunk_id = find_chunk_by_keyword(rows, "skill")
    state_supported = {
        "user_query": "what are the skills have for ai/ml engineer?",
        "retrieved_chunks": (
            [{"chunk_id": skill_chunk_id}]
            if skill_chunk_id
            else []
        )
    }

    result_supported = answer_generation_node(
        state=state_supported,
        chunk_retriever=chunk_retriever
    )

    print("\n=== SUPPORTED QUESTION ===")
    print("Answer supported:", result_supported["answer_supported"])
    print("Answer text:\n", result_supported["answer_text"])
    print("Citations:", result_supported["answer_citations"])

    # --------------------------------------------------
    # 5. CASE 2 — Unsupported question
    # --------------------------------------------------
    skill_chunk_id = find_chunk_by_keyword(rows, "skill")
    state_unsupported = {
        "user_query": "What is the company's remote work policy?",
        "retrieved_chunks": (
            [{"chunk_id": skill_chunk_id}]
            if skill_chunk_id
            else []
        )
    }

    result_unsupported = answer_generation_node(
        state=state_unsupported,
        chunk_retriever=chunk_retriever
    )

    print("\n=== UNSUPPORTED QUESTION ===")
    print("Answer supported:", result_unsupported["answer_supported"])
    print("Answer text:", result_unsupported["answer_text"])
    print("Citations:", result_unsupported["answer_citations"])

    # --------------------------------------------------
    # 6. CASE 3 — No chunks
    # --------------------------------------------------
    state_no_chunks = {
        "user_query": "Explain ISO compliance requirements",
        "retrieved_chunks": []
    }

    result_no_chunks = answer_generation_node(
        state=state_no_chunks,
        chunk_retriever=chunk_retriever
    )

    print("\n=== NO CHUNKS CASE ===")
    print("Answer supported:", result_no_chunks["answer_supported"])
    print("Answer text:", result_no_chunks["answer_text"])

    conn.close()
