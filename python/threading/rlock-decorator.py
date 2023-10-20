import threading
import time


# if we change the RLock to a regular Lock, we will encounter a deadlock because thread t1 cannot reacquire the Lock
# in this example, when we are creating decorators that acquire the same lock, 
# we should use a reentrant lock if the decorators are going to be stacked on top of each other.
lock = threading.RLock()


def synchronized1(func):
  def wrapper(*args, **kwargs):
    with lock:
      print("running synchronized1")
      return func(*args, **kwargs)
  return wrapper


def synchronized2(func):
  def wrapper(*args, **kwargs):
    with lock:
      print("running synchronized2")
      return func(*args, **kwargs)
  return wrapper


@synchronized1
@synchronized2
def worker():
  pass


t1 = threading.Thread(target=worker)
t2 = threading.Thread(target=worker)
t1.start()
t2.start()

time.sleep(5)

t1.join()
t2.join()
