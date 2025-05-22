from time import sleep


"""
'calls',        SortKey.CALLS,      call count
'cumulative'    SortKey.CUMULATIVE, cumulative time
'cumtime',      N/A,                cumulative time
'file',         N/A,                file name
'filename',     SortKey.FILENAME,   file name
'module',       N/A,                file name
'ncalls',       N/A,                call count
'pcalls',       SortKey.PCALLS,     primitive call count
'line',         SortKey.LINE,       line number
'name',         SortKey.NAME,       function name
'nfl',          SortKey.NFL,        name/file/line
'stdname',      SortKey.STDNAME,    standard name
'time',         SortKey.TIME,       internal time
'tottime',      N/A,                internal time
"""


def long_func():
    print("functioned entered")
    sleep(3)

    a = []
    for i in range(100000):
        a.append(i*i)


long_func()

# How to profile:
# run the following command to profile code: 
# python -m cProfile -o main.prof main.py

# run the following command to visualize main.prof:
# pip install snakeviz
# snakeviz main.prof

