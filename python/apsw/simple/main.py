import apsw
import os
import time


# 1) Define the schema
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY,
    name TEXT
);
"""


# 2) Create two databases
for path in ("db_a.sqlite", "db_b.sqlite"):
    if os.path.exists(path):
        os.remove(path)

conn_a = apsw.Connection("db_a.sqlite")
conn_b = apsw.Connection("db_b.sqlite")

conn_a.execute(SCHEMA_SQL)
conn_b.execute(SCHEMA_SQL)

# 3) Start a Session on DB A (the source)
# Start a session that tracks changes to the "runs" table
session = apsw.Session(conn_a, "main")

# Only replicate the tables we care about
session.table_filter(lambda name: name == "runs")

# 4) Make changes in DB A
conn_a.execute("INSERT INTO runs (run_id, name) VALUES (?, ?)", (1, "first run"))
conn_a.execute("INSERT INTO runs (run_id, name) VALUES (?, ?)", (2, "second run"))

# 5) Export a changeset
changeset = session.changeset()

print(f"Changeset size: {len(changeset)} bytes")

time.sleep(10)  # Ensure a time gap for demonstration purposes

# 6) Apply the changeset to DB B (the target)
def conflict_handler(reason, change):
    # Simple policy: ignore conflicts
    return apsw.SQLITE_CHANGESET_OMIT

apsw.Changeset.apply(
    changeset,
    conn_b,
    filter=lambda name: name == "runs",
    conflict=conflict_handler,
)

# 7) Verify replication
rows = list(conn_b.execute("SELECT run_id, name FROM runs ORDER BY run_id"))
print(rows)
