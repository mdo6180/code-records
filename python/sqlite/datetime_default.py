import datetime
import sqlite3
import time



now = datetime.datetime.now()

con = sqlite3.connect(":memory:", detect_types=sqlite3.PARSE_DECLTYPES)
cur = con.cursor()
cur.execute("CREATE TABLE runs(run_id INTEGER PRIMARY KEY AUTOINCREMENT, start_time DATETIME, end_time DATETIME DEFAULT NULL)")
cur.execute("INSERT INTO runs(start_time, end_time) VALUES(?, ?)", (now, now, ))
cur.execute("SELECT * FROM runs")
print(cur.fetchall())

time.sleep(1)

# update end_time to now
now_now = datetime.datetime.now()
cur.execute("UPDATE runs SET end_time = ? WHERE run_id = 1", (now_now,))
cur.execute("SELECT * FROM runs")
print(cur.fetchall())

time.sleep(1)

cur.execute("INSERT INTO runs(start_time) VALUES(?)", (now,))
cur.execute("SELECT * FROM runs")
print(cur.fetchall())

cur.execute("SELECT run_id FROM runs WHERE end_time IS NULL")
print(cur.fetchone()[0])

cur.close()
con.close()