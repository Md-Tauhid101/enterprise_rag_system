# postgres.py
import psycopg2
import uuid
import hashlib

class PostgresStore:
    def __init__(self, db_config):
        self.conn = psycopg2.connect(**db_config)
        self.cursor = self.conn.cursor()

    
    ### DOCUMENT
    def insert_document(self, source_path, source_type, checksum, version=1):
        document_id = uuid.uuid4()

        self.cursor.execute("""
            INSERT INTO documents (
                document_id, source_path, source_type, checksum, version             
            )
            VALUES (%s, %s, %s, %s, %s)
        """, (
            str(document_id),
            source_path,
            source_type,
            checksum,
            version
        ))
        return document_id
    
    ### CHUNKS
    def insert_chunks(self, document_id, chunks):
        chunk_ids = []

        for idx, chunk in enumerate(chunks):
            chunk_id = uuid.uuid4()
            chunk_ids.append(chunk_id)

            chunk_hash = hashlib.sha256(
                chunk["cleaned_text"].encode("utf-8")
            ).hexdigest()

            self.cursor.execute("""
                INSERT INTO chunks (
                    chunk_id,
                    document_id,
                    chunk_index,
                    raw_text,
                    cleaned_text,
                    chunk_hash
                )
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                str(chunk_id),
                str(document_id),
                idx,
                chunk["raw_text"],
                chunk["cleaned_text"],
                chunk_hash
            ))
        return chunk_ids
    
    ### TRANSACTIONS
    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()
    
    def close(self):
        self.cursor.close()
        self.conn.close()