from agents.state import QueryState
from ingestion.embed_func import embed_text

def embed_query_node(state: QueryState) -> QueryState:
    query = state["user_query"]
    embedding = embed_text(query)

    return {
        **state,
        "query_embedding": embedding.tolist()
    }