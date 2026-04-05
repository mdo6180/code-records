import sqlite3
import time



source_con = sqlite3.connect("source.sqlite")
source_cur = source_con.cursor()
source_cur.execute("CREATE TABLE IF NOT EXISTS test(id INTEGER PRIMARY KEY, name TEXT)")
source_cur.execute("INSERT INTO test(name) VALUES('Alice')")
source_cur.execute("INSERT INTO test(name) VALUES('Bob')")
source_con.commit()
source_cur.close()
source_con.close()


target_con = sqlite3.connect("target.sqlite")
target_cur = target_con.cursor()
target_cur.execute("CREATE TABLE IF NOT EXISTS test(id INTEGER PRIMARY KEY, name TEXT)")
target_con.commit()
target_cur.close()
target_con.close()


time.sleep(5)  # Simulate some delay before the import


def import_data(target_db: str, source_db: str):
    conn = sqlite3.connect(target_db)
    try:
        cur = conn.cursor()
        cur.execute("ATTACH DATABASE ? AS incoming", (source_db,))  
        cur.execute("BEGIN TRANSACTION")

        # Append test tables
        cur.execute("""
            INSERT OR IGNORE INTO main.test
            SELECT * FROM incoming.test
        """)

        conn.commit()
        cur.execute("DETACH DATABASE incoming")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

import_data("target.sqlite", "source.sqlite")

