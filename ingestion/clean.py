import re

def clean_text(text: str) -> str:
    if not text:
        return ""

    # Remove page numbers like "Page 3 of 12"
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)

    # Fix hyphenated line breaks
    text = re.sub(r"-\n", "", text)

    # Normalize newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove excessive spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()
