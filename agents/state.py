# state.py
from typing import TypedDict, Optional, List, Dict

class QueryState(TypedDict):
    user_query: str

    # from node 1
    intent: Optional[str]
    intent_confidence: Optional[float]
    intent_reason: Optional[str]
    should_refuse: bool

    # node - 2
    rewritten_queries: Optional[Dict[str, object]]
    rewrite_risk: Optional[Dict[str, bool]]

