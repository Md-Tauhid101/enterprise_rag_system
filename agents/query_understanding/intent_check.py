# intent_check.py (NODE-1)
from agents.state import QueryState
from utils.llm import get_llm
from utils.json import extract_json
import json

INTENT_LABELS = [
    "factual",
    "analytical",
    "multi_hop",
    "unanswerable",
    "unknown"
]

def intent_check_node(state: QueryState) -> QueryState:
    query = state["user_query"]
    llm = get_llm()

    system_prompt = """
        You are an intent classifier for a retrieval system.

        Allowed labels:
        - factual
        - analytical
        - multi_hop
        - unanswerable

        You MUST respond with ONLY valid JSON.
        No markdown.
        No explanation.
        No extra text.

        JSON schema:
        {
        "intent": "factual | analytical | multi_hop | unanswerable",
        "confidence": number,
        "reason": string
        }
    """



    user_prompt = f"""
    Query: "{query}"

    Decide intent.
    Explain briefly.
    """

    response = llm.invoke(system_prompt + user_prompt)
    # print("RAW LLM OUTPUT:\n", response.content)

    try:
        parsed = extract_json(response.content)
    except Exception:
        # if model can’t follow instructions → safest option
        return {
            **state,
            "intent": "unknown",
            "intent_confidence": 0.0,
            "intent_reason": "Intent classification failed (invalid model output)",
            "should_refuse": False,
        }
    intent = parsed.get("intent")
    if intent not in INTENT_LABELS:
        intent = "unanswerable"

    should_refuse = intent == "unanswerable"

    return {
        **state,
        "intent": intent,
        "intent_confidence": parsed.get("confidence", 0.5),
        "intent_reason": parsed.get("reason", ""),
        "should_refuse": should_refuse
    }

