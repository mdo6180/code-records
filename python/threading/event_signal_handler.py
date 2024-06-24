import threading
import time
import signal


# Create an Event object
shutdown_event = threading.Event()


# Note: the internal flag is initially set to False, thus shutdown_event.is_set() returns False.
def wait_for_event():
    while not shutdown_event.is_set():
        print("running")
        time.sleep(1)


# Create and start child thread
wait_thread = threading.Thread(target=wait_for_event)
wait_thread.start()


def _kill(sig, frame):
    print("\nCTRL+C Caught!; Killing main thread and child threads...")
    shutdown_event.set()
    wait_thread.join()
    print("All threads have finished.")

signal.signal(signal.SIGINT, _kill)


