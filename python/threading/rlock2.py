import threading
import time


# if we change the RLock to a regular Lock, we will encounter a deadlock because thread t1 cannot reacquire the Lock
lock = threading.RLock()

def worker():
  lock.acquire()
  try:
    print("Working...")
    lock.acquire() # Acquire again
  finally:
    lock.release()
    lock.release() # Release twice

t1 = threading.Thread(target=worker)
t2 = threading.Thread(target=worker)
t1.start()
t2.start()

time.sleep(5)

t1.join()
t2.join()
