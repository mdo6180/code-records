import threading
import time
import random

def child_thread(lock, event):
    print(f"Thread {threading.current_thread().name} is waiting to acquire the lock.")
    with lock:
        print(f"Thread {threading.current_thread().name} acquired the lock.")
        sleep_time = random.uniform(0.5, 2.0)  # Random sleep time between 0.5 and 2.0 seconds
        print(f"Thread {threading.current_thread().name} is sleeping for {sleep_time:.2f} seconds.")
        time.sleep(sleep_time)
        event.set()  # Signal that child thread has acquired the lock
    print(f"Thread {threading.current_thread().name} is done.")

# Number of child threads
NUM_THREADS = 3

# Create a lock
lock = threading.Lock()

# Create an event
event = threading.Event()

# Create and start the child threads
child_threads = []
for i in range(NUM_THREADS):
    child = threading.Thread(target=child_thread, args=(lock, event))
    child_threads.append(child)
    child.start()

# Wait for all child threads to acquire the lock
for _ in range(NUM_THREADS):
    event.wait()

print("Main thread is waiting to acquire the lock.")
with lock:
    print("Main thread acquired the lock.")

# Wait for all child threads to finish
for child in child_threads:
    child.join()

print("Both child and main threads have finished.")
