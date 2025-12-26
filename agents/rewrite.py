# rewrite.py
from agents.state import QueryState
from utils.llm import get_llm
import json

def rewrite_node(state: QueryState) -> QueryState:
    # refusal status check
    if state["should_refuse"]:
        return state
    
    query = state["user_query"]
    intent = state["intent"]
    llm = get_llm()

    # default outputs
    rewrites = {
        "original": query,
        "keyword_expansion": None,
        "hyde": None,
        "sub_questions": None
    }

    risk_flags = {
        "recall_boost": False,
        "precision_risk": False
    }

    # if intent is factual -> minimal expansion
    if intent == "factual":
        prompt = f"""
        Expand this query with ONLY essential keywords.
        Do NOT add new facts.
        Do NOT broaden scope.

        Query: "{query}"

        Return JSON:
        {{"expanded_query": "..."}}
        """
        response = llm.invoke(prompt)
        try:
            data = json.loads(response.content)
            rewrites["keyword_expansion"] = data["expanded_query"]
            risk_flags["recall_boost"] = True
        except Exception:
            pass

    # If intent is analytical -> expansion + HyDE
    elif intent == "analytical":
        prompt = f"""
        Generate:
        1. A keyword-expanded query
        2. A hypothetical answer (HyDE)

        Rules:
        - HyDE may fabricate but must stay on-topic
        - Do NOT add external facts

        Query: "{query}"

        Return JSON:
        {{
            "expanded_query": "...",
            "hyde": "..."
        }}
        """

        response = llm.invoke(prompt)
        try:
            data = json.loads(response.content)
            rewrites["keyword_expansion"] = data["expanded_query"]
            rewrites["hyde"] = data["hyde"]
            risk_flags["recall_boost"] = True
            risk_flags["precision_risk"] = True
        except Exception:
            pass
    
    # if intent is multi-hop -> decomposition
    elif intent == "multi_hop":
        prompt = f"""
        Decompose this query into minimal sub-questions.

        Rules:
        - Each sub-question must be answerable independently
        - Do NOT introduce new facts

        Query: "{query}"

        Return JSON:
        {{"sub_questions": ["...", "..."]}}
        """
        response = llm.invoke(prompt)
        try:
            data = json.loads(response.content)
            rewrites["sub_questions"] = data["sub_questions"]
            risk_flags["recall_boost"] = True
            risk_flags["precision_risk"] = True
        except Exception:
            pass
    elif intent == "unknown":
        # Conservative fallback: no LLM call
        rewrites["keyword_expansion"] = query
        rewrites["hyde"] = None
        rewrites["sub_questions"] = None

        risk_flags["recall_boost"] = False
        risk_flags["precision_risk"] = True

    # UNANSWERABLE should never reach here
    return{
        **state,
        "rewritten_queries": rewrites,
        "rewrite_risk": risk_flags
    }