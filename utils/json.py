import json
import re

def extract_json(text: str):
    """
    Extract first JSON object from text.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found")
    return json.loads(match.group())
