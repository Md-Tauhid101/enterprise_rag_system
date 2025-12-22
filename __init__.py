import psycopg2

conn = psycopg2.connect(
    host = "localhost",
    database = "enterprise_rag_system_DB",
    user = "postgres",
    password = "Tauhid@2003",
    port = 5432
)
cursor = conn.cursor()

