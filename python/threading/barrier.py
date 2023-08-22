import threading
import time
import random

# Define the number of threads
NUM_THREADS = 3

# Define a function that will run in each thread
def worker(barrier):
    sleep_time = random.uniform(0.5, 2.0)  # Random sleep time between 0.5 and 2.0 seconds
    print(f"Thread {threading.current_thread().name} is sleeping for {sleep_time:.2f} seconds.")
    time.sleep(sleep_time)
    print(f"Thread {threading.current_thread().name} is waiting at the barrier.")
    barrier.wait()  # Wait for all threads to reach the barrier
    print(f"Thread {threading.current_thread().name} passed the barrier.")

# Create a barrier that waits for NUM_THREADS threads
# Note: number of parties in the barrier must be equal to the number of threads
barrier = threading.Barrier(NUM_THREADS)

# Create and start the threads
threads = []
for i in range(NUM_THREADS):
    thread = threading.Thread(target=worker, args=(barrier,))
    threads.append(thread)
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()

print("All threads have passed the barrier.")
