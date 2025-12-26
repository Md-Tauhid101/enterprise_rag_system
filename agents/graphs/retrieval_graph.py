# agents/graphs/retrieval_graph.py
from langgraph.graph import StateGraph
from agents.state import QueryState
from agents.retrieve import retrieve_node
from agents.retrieval_validation import retrieval_validation_node


def build_retrieval_graph(chunk_loader):
    """
    Retrieval + Validation graph.
    Responsibilities:
    - Step 4: retrieve candidate chunks
    - Step 6: validate retrieval safety
    """

    graph = StateGraph(QueryState)

    # Nodes
    graph.add_node("retrieve", retrieve_node)

    # validation needs chunk_loader for limited text inspection
    graph.add_node(
        "validate",
        lambda state: retrieval_validation_node(state, chunk_loader)
    )

    # Edges
    graph.add_edge("retrieve", "validate")

    # Entry + Exit
    graph.set_entry_point("retrieve")
    graph.set_finish_point("validate")

    return graph.compile()
