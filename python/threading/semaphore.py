import threading
import time

# Create a counting semaphore

# if value of semaphore is less than the number of child threads (suppose we set the value to 3 i.e., we set num_connections = 3), 
# threads 4 and 5 will have to wait for threads 1, 2, or 3 to release the semaphore before it can acquire the semaphore and start its execution.

# if value of semaphore is equal to the number of child threads, then all threads can acquire the semaphore and execute all at once.
 
# num_connections = 3  # Number of available connections
num_connections = 5
semaphore = threading.Semaphore(value=num_connections)

# Simulated function that uses a shared resource (database connection)
def use_database_connection(thread_id):
    with semaphore:
        print(f"Thread {thread_id} acquired a database connection.")
        time.sleep(2)  # Simulate using the connection
        print(f"Thread {thread_id} released the database connection.")

# Create multiple threads that use the shared resource (database connection)
threads = []
for i in range(5):
    thread = threading.Thread(target=use_database_connection, args=(i+1,))
    threads.append(thread)
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()

print("All threads have finished.")
