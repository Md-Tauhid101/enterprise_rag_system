# refusal.py
def refusal_message(reason: str) -> str:
    """
    Standard refusal response.
    Never adds facts.
    Never suggests retrieval.
    """
    return (
        "I can't answer question using the available documents.\n"
        f"Reason: {reason}"
    )