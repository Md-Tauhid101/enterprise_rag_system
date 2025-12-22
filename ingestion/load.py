# load.py
import os
import uuid
from unstructured.partition.auto import partition
from langchain_core.documents import Document
from typing import List

IMAGE_DIR = "./data/images"
os.makedirs(IMAGE_DIR, exist_ok=True)

TEXT_ELEMENT_TYPES = {
    "Text",
    "Title",
    "NarrativeText",
    "Header",
    "Footer",
    "SectionHeader",
    "ListItem"
}

def load_documents(file_paths: List):
    """
    Load files into normalized Document objects WITHOUT embedding.
    This function is modality-aware but model-agnostic.
    """
    docs = []
    for file_path in file_paths:
        elements = partition(filename=file_path)
        file_extention = os.path.splitext(file_path)[-1].lower()

        # generate id for doc
        doc_id = uuid.uuid4().hex

        for idx, el in enumerate(elements):
            raw_type = type(el).__name__ # it gives structural element type like text, image, table etc
            element_type = (
                "Text" if raw_type in {
                    "Text", "NarrativeText", "Title", "Header", "Footer", "ListItem"
                }
                else raw_type
            )

            text = getattr(el, "text", None)  # if text in el it will return it's value otherwise default value is None

            # Image element
            if element_type == "Image" and hasattr(el, "image") and el.image is not None:
                image_id = f"{uuid.uuid4().hex}.png"
                image_path = os.path.join(IMAGE_DIR, image_id)

                # save image to disk
                el.image.save(image_path)

                doc = Document(
                    page_content="", # images have no text
                    metadata = {
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}_{idx}",
                        "source": file_path,
                        "file_ext": file_extention,
                        "element_type": "Image",
                        "image_path": image_path
                    }
                )
                docs.append(doc)
                continue
            
            # text/table element
            if not text or not text.strip():
                continue
            
            raw_meta = el.metadata.to_dict()
            extra_metadata = {
                "page_number": raw_meta.get("page_number"),
                "language": raw_meta.get("language"),
                "raw_element_type": raw_type
            }

            doc = Document(
                page_content=text or "",
                metadata = {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_{idx}",   
                    "source": file_path,
                    "file_ext": file_extention,
                    "element_type": element_type,
                    **extra_metadata
                }
            )
            docs.append(doc)
    return docs


if __name__ == "__main__":
    docs = load_documents(["./data/raw/doc_pdf.pdf", "./data/raw/doc_pdf_img.pdf", "./data/raw/doc_docx.docx", "./data/raw/doc_ppt.pptx", "./data/raw/scaned_pdf.pdf"])
    # docs = load_documents(["./data/raw/doc_pdf_img.pdf"])

    print(f"Total documents loaded: {len(docs)}\n")
    print(f"✅============={type(docs)}\n")

    # sanity check first 5 elements
    for d in docs[-10:]:
        print({
            "element_type": d.metadata.get("element_type"),
            "file": d.metadata.get("source"),
            "text_preview": d.page_content[:80]
        })

    from collections import Counter

    print("\nElement distribution:")
    print(Counter(d.metadata["element_type"] for d in docs))

    print("\nImage paths:")
    for d in docs:
        if d.metadata["element_type"] == "Image":
            print(d.metadata["image_path"])
            break
