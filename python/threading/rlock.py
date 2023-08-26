from time import sleep
from threading import Thread, Lock, RLock
from random import random

# reporting function
def report(lock, identifier):
    # acquire the lock
    with lock:
        print(f'>thread {identifier} done')

# work function
def task(lock, identifier, value):
    # acquire the lock
    with lock:
        print(f'>thread {identifier} sleeping for {value}')
        sleep(value)
        # report
        report(lock, identifier)

# create a shared reentrant lock
lock = RLock()

# using a regular lock will cause a deadlock
# lock = Lock()

# start a few threads that attempt to execute the same critical section
for i in range(10):
    # start a thread
    Thread(target=task, args=(lock, i, random())).start()
# wait for all threads to finish...
