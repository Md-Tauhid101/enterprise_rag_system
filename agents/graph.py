# graph.py
from langgraph.graph import StateGraph, END
from agents.state import QueryState
from agents.query_understanding.intent_check import intent_check_node
from agents.query_understanding.rewrite import rewrite_node
from utils.refusal import refusal_message

def refuse_node(state: QueryState):
    return {
        "final_answer": refusal_message(state["intent_reason"])
    }

def route_after_intent(state: QueryState) -> str:
    if state["should_refuse"]:
        return "REFUSE"
    return "REWRITE"

def build_query_understanding_graph():
    graph = StateGraph(QueryState)

    # nodes
    graph.add_node("INTENT_CHECK", intent_check_node)
    graph.add_node("REWRITE", rewrite_node)
    graph.add_node("REFUSE", refuse_node)

    # node entry
    graph.set_entry_point("INTENT_CHECK")

    # Conditional routing
    graph.add_conditional_edges(
        "INTENT_CHECK",
        route_after_intent,
        {
            "REWRITE": "REWRITE",
            "REFUSE": "REFUSE"
        }
    )

    # Rewrite -> END
    graph.add_edge("REWRITE", END)

    # Refusal -> END
    graph.add_edge("REFUSE", END)

    return graph.compile()


if __name__ == "__main__":
    from agents.graph import build_query_understanding_graph

    graph = build_query_understanding_graph()

    result = graph.invoke({
        "user_query": "what are the skills required for ai/ml engineer?"
    })
    print(result)
