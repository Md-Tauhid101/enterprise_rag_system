# ingest.py
from embed_func import embed_text, embed_image, embed_table
from clean import clean_text

from typing import List
from PIL import Image

TEXT_ELEMENT_TYPES = {
    "Text",
    "Title",
    "NarrativeText",
    "Header",
    "Footer",
    "SectionHeader",
    "ListItem"
}

def ingest_documents(docs: List):
    records = []

    for doc in docs:
        elt_type = doc.metadata.get("element_type")

        # -------- TEXT --------
        if elt_type in TEXT_ELEMENT_TYPES:
            cleaned_doc = clean_text(doc.page_content)
            if not cleaned_doc:
                continue
            vec = embed_text(cleaned_doc)

        # -------- TABLE --------
        elif elt_type == "Table":
            vec = embed_table(doc.page_content)
            if vec is None:
                continue

        # -------- IMAGE --------
        elif elt_type == "Image":
            image_path = doc.metadata["image_path"]
            pil_image = Image.open(image_path).convert("RGB")
            vec = embed_image(pil_image)

        else:
            continue

        records.append({
            "embedding": vec,
            "document": doc
        })

    return records


if __name__ == "__main__":
    from load import load_documents

    FILES = ["./data/raw/doc_pdf.pdf", "./data/raw/doc_pdf_img.pdf", "./data/raw/doc_docx.docx", "./data/raw/doc_ppt.pptx", "./data/raw/scaned_pdf.pdf"]

    # ---------- STEP 1: LOAD ----------
    docs = load_documents(FILES)
    print(f"[LOAD] Total elements loaded: {len(docs)}")

    # quick sanity: element distribution
    from collections import Counter
    element_counts = Counter(d.metadata["element_type"] for d in docs)
    print("[LOAD] Element types:", dict(element_counts))

    # ---------- STEP 2: INGEST ----------
    records = ingest_documents(docs)
    print(f"[INGEST] Total embeddings created: {len(records)}")

    # ---------- STEP 3: VERIFY PAIRS ----------
    for r in records[:5]:
        doc = r["document"]
        emb = r["embedding"]

        print({
            "element_type": doc.metadata["element_type"],
            "embedding_dim": emb.shape[0],
            "text_preview": doc.page_content[:60]
        })
