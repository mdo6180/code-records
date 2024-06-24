import threading
import time
import signal


# Create an Event object
pause_event = threading.Event()
shutdown_event = threading.Event()


# Note: the internal flag is initially set to False, thus shutdown_event.is_set() returns False.
def wait_for_event():
    pause_event.set()

    while not shutdown_event.is_set():
        if pause_event.is_set():
            print("running...")
            time.sleep(1)
        else:
            print("paused for max 2 seconds...")
            pause_event.wait(timeout=2)
            pause_event.set()


# Create and start child thread
wait_thread = threading.Thread(target=wait_for_event)
wait_thread.start()


def _shutdown(sig, frame):
    print("\nCTRL+C Caught!; Killing main thread and child threads...")
    shutdown_event.set()
    wait_thread.join()
    print("All threads have finished.")

def _pause(sig, frame):
    print("\nCTRL+C Caught!; pausing child threads...")
    pause_event.clear()
    signal.signal(signal.SIGINT, _shutdown)

signal.signal(signal.SIGINT, _pause)


