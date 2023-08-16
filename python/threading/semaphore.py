import threading
import time

# Create a counting semaphore
num_connections = 3  # Number of available connections
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
