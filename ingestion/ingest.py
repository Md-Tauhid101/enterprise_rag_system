# ingest.py
from ingestion.embed_func import embed_text, embed_image, embed_table
from ingestion.clean import clean_text

from typing import List
from PIL import Image

import os
import hashlib
import psycopg2

from storage.postgres import PostgresStore
from storage.vector_store import VectorStore

# DB connection
from config import DB_CONFIG
conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

TEXT_ELEMENT_TYPES = {
    "Text",
    "Title",
    "NarrativeText",
    "Header",
    "Footer",
    "SectionHeader",
    "ListItem"
}

def prepare_chunks(docs: List):
    prepared = []

    for doc in docs:
        elt_type = doc.metadata.get("element_type")

        if elt_type in TEXT_ELEMENT_TYPES:
            cleaned = clean_text(doc.page_content)
            if not cleaned:
                continue
        elif elt_type == "Table":
            cleaned = doc.page_content
            if not cleaned:
                continue
        elif elt_type == "Image":
            cleaned = None
        else:
            continue

        prepared.append({
            "element_type": elt_type,
            "raw_text": doc.page_content,
            "cleaned_text": cleaned,
            "page_number": doc.metadata.get("page_number"),
            "image_path": doc.metadata.get("image_path")
        })
    return prepared


def ingest_pipeline(docs, source_path, source_type, raw_file_bytes):
    pg = PostgresStore(DB_CONFIG)
    vs = VectorStore(dim=768)

    try:
        # Prepare chunk payloads
        chunks = prepare_chunks(docs=docs)

        # Insert document metadata
        checksum = hashlib.sha256(raw_file_bytes).hexdigest()
        document_id = pg.insert_document(
            source_path=source_path,
            source_type=source_type,
            checksum=checksum
        )
        # Insert chunk metadata before embedding(no embeddings)
        chunk_ids = pg.insert_chunks(document_id=document_id, chunks=chunks)

        # COMMIT TO DATABASE
        pg.commit()

        # Now embeddings are allowed
        for chunk, chunk_id in zip(chunks, chunk_ids):
            if chunk["element_type"] in TEXT_ELEMENT_TYPES:
                vec = embed_text(chunk["cleaned_text"])
            elif chunk["element_type"] == "Table":
                vec = embed_table(chunk["cleaned_text"])
                if vec is None:
                    continue
            elif chunk["element_type"] == "Image":
                img = Image.open(chunk["image_path"]).convert("RGB")
                vec = embed_image(img)
            else: continue

            vs.add(vec, metadata = {"chunk_id": str(chunk_id)})
    except Exception:
        pg.rollback()
        raise
    finally:
        pg.close()
