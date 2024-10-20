import datetime
import sqlite3
import time



def adapt_datetime(ts: datetime.datetime):
    return ts.isoformat()

def convert_datetime(s: bytes):
    return datetime.datetime.fromisoformat(s.decode())

sqlite3.register_adapter(datetime.datetime, adapt_datetime)
sqlite3.register_converter("datetime", convert_datetime)



now = datetime.datetime.now()

con = sqlite3.connect(":memory:", detect_types=sqlite3.PARSE_DECLTYPES)
cur = con.cursor()
cur.execute("CREATE TABLE runs(run_id INTEGER PRIMARY KEY AUTOINCREMENT, start_time datetime, end_time datetime DEFAULT NULL)")
cur.execute("INSERT INTO runs(start_time) VALUES(?)", (now,))
cur.execute("SELECT * FROM runs")
print(cur.fetchall())

time.sleep(1)

# update end_time to now
now_now = datetime.datetime.now()
cur.execute("UPDATE runs SET end_time = ? WHERE run_id = 1", (now_now,))
cur.execute("SELECT * FROM runs")
print(cur.fetchall())

cur.close()
con.close()