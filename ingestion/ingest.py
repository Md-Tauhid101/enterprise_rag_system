# ingest.py
from ingestion.embed_func import embed_text, embed_image
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

def prepare_chunks(docs):
    """
    Normalize raw document elements into ingestion-ready chunks.

    Rules:
    - If image_path exists → Image chunk
    - Else if page_content is text → Text chunk
    - Otherwise → drop
    """

    prepared = []

    for doc in docs:
        text = doc.page_content
        meta = doc.metadata or {}

        image_path = meta.get("image_path")
        page_number = meta.get("page_number")

        # --- IMAGE CHUNK ---
        if image_path and os.path.exists(image_path):
            prepared.append({
                "element_type": "Image",
                "cleaned_text": None,
                "page_number": page_number,
                "image_path": image_path
            })
            continue

        # --- TEXT CHUNK ---
        if isinstance(text, str):
            cleaned = clean_text(text)
            if not cleaned:
                continue

            prepared.append({
                "element_type": "Text",
                "raw_text": doc.page_content,
                "cleaned_text": cleaned,
                "page_number": page_number,
                "image_path": None
            })

    return prepared



def ingest_pipeline(docs, source_path, source_type, raw_file_bytes, vector_store):
    pg = PostgresStore(DB_CONFIG)

    embedded_text = 0
    embedded_images = 0
 
    try:
        # Prepare chunk payloads
        chunks = prepare_chunks(docs=docs)
        # print("[DEBUG] prepare_chunks count:", len(chunks))
        # print("[DEBUG] sample chunk:", chunks[0] if chunks else None)

        if not chunks:
            raise RuntimeError("No valid chunks produced")
        
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

        # Embed + Store
        for chunk, chunk_id in zip(chunks, chunk_ids):
        # for i, (chunk, chunk_id) in enumerate(zip(chunks, chunk_ids)):
            # print(f"[DEBUG] Loop {i} | type={chunk['element_type']}")

            # ---- IMAGE ----
            if chunk["element_type"] == "Image":
                img = Image.open(chunk["image_path"]).convert("RGB")
                vec = embed_image(img)
                vector_store.add_image(vec, str(chunk_id))
                embedded_images += 1

            # ---- TEXT (includes tables) ----
            else:
                # print("[DEBUG] Calling embed_text")
                vec = embed_text(chunk["cleaned_text"])
                # print("[DEBUG] embed_text returned", type(vec), vec.shape)
                vector_store.add_text(vec, str(chunk_id))
                embedded_text += 1

        print(f"[VERIFY] Embedded text chunks: {embedded_text}")
        print(f"[VERIFY] Embedded image chunks: {embedded_images}")
        print(f"[VERIFY] FAISS text vectors: {vector_store.text_index.ntotal}")
        print(f"[VERIFY] FAISS image vectors: {vector_store.image_index.ntotal}")
        if embedded_text == 0 and embedded_images == 0:
            raise RuntimeError("Ingestion failed: no embeddings created")
        assert vector_store.text_index.ntotal > 0 or vector_store.image_index.ntotal > 0, \
        "FAISS EMPTY — embeddings never added"

    except Exception:
        pg.rollback()
        raise
    finally:
        pg.close()
