import sqlite3

conn = sqlite3.connect(":memory:")

conn.execute("""
CREATE VIRTUAL TABLE documents
USING fts5(title, body)
""")

conn.executemany(
    "INSERT INTO documents VALUES (?, ?)",
    [
        ("SQLite", "SQLite is a lightweight embedded database."),
        ("FastAPI", "FastAPI is an asynchronous Python framework."),
        ("Anacostia", "Anacostia is built around SQLite and FastAPI.")
    ]
)

rows = conn.execute("""
SELECT title
FROM documents
WHERE documents MATCH ?
""", ("SQLite",))

for row in rows:
    print(row)