import threading
import time

# Create an Event object
event = threading.Event()

# Function to wait for the event to be set
# because the flag is initially set to False, the thread automatically blocks until set() is called.
def wait_for_event():
    print("Waiting for event to be set...")
    event.wait()  # Wait until the event is set
    print("Event is set!")

# Function to clear the event
def set_event():
    time.sleep(2)
    
    event.set()  # Set the event
    print("Setting the event!")

# Create threads for each function
wait_thread = threading.Thread(target=wait_for_event)
set_thread = threading.Thread(target=set_event)

# Start threads
wait_thread.start()

# Wait for a while before setting the event
time.sleep(1)
set_thread.start()

# Wait for threads to finish
wait_thread.join()
set_thread.join()

print("All threads have finished.")
