from ingestion.load import load_documents
from ingestion.ingest import ingest_pipeline
from typing import List
import os
from dotenv import load_dotenv
load_dotenv()

from storage.vector_store import VectorStore

vs = VectorStore()
def run_ingestion(file_paths):
    print("🚀 Starting ingestion pipeline...\n")

    for file_path in file_paths:
        print(f"📄 Processing file: {file_path}")

        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue

        # 1️⃣ Read raw bytes (document-level)
        with open(file_path, "rb") as f:
            raw_file_bytes = f.read()

        # 2️⃣ Load + chunk document
        docs = load_documents([file_path])
        print(f"   ➜ Extracted {len(docs)} chunks")

        if not docs:
            print("   ⚠️ No valid chunks found, skipping file\n")
            continue

        # 3️⃣ Derive metadata
        source_path = file_path
        source_type = file_path.split(".")[-1].lower()

        # 4️⃣ Ingest pipeline
        ingest_pipeline(
            docs=docs,
            source_path=source_path,
            source_type=source_type,
            raw_file_bytes=raw_file_bytes,
            vector_store=vs
        )

        print(f"   ✅ Successfully ingested: {file_path}\n")
    vs.save()

    print("🎉 Ingestion pipeline completed for all files.")

if __name__ == "__main__":
    FILES_TO_INGEST = ["./data/raw/doc_pdf.pdf", "./data/raw/doc_pdf_img.pdf", "./data/raw/doc_docx.docx", "./data/raw/doc_ppt.pptx", "./data/raw/scaned_pdf.pdf", "./data/raw/img.png"]

    run_ingestion(FILES_TO_INGEST)