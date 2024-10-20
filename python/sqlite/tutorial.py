import sqlite3



class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

def adapt_point(point: Point):
    return f"{point.x};{point.y}"

# Note Converter functions are always passed a bytes object, no matter the underlying SQLite data type.
def convert_point(s: bytes):
    x, y = list(map(float, s.split(b";")))
    return Point(x, y)



# Register the adapter and converter
sqlite3.register_adapter(Point, adapt_point)
sqlite3.register_converter("point", convert_point)

# 1) Parse using declared types
# Note: if detect_types is set to sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES (i.e. both PARSE_DECLTYPES and PARSE_COLNAMES),
# column names take precedence over declared types.
p = Point(4.0, -3.2)
con = sqlite3.connect(":memory:", detect_types=sqlite3.PARSE_DECLTYPES)
cur = con.execute("CREATE TABLE test(p point)")

cur.execute("INSERT INTO test(p) VALUES(?)", (p,))
cur.execute("SELECT p FROM test")
print("with declared types:", cur.fetchone()[0])
cur.close()
con.close()

# 2) Parse using column names
con = sqlite3.connect(":memory:", detect_types=sqlite3.PARSE_COLNAMES)
cur = con.execute("CREATE TABLE test(p)")

cur.execute("INSERT INTO test(p) VALUES(?)", (p,))
cur.execute('SELECT p AS "p [point]" FROM test')
print("with column names:", cur.fetchone()[0])
cur.close()
con.close()