import threading, time

def main():
    threading.Thread(target=bloop).start()

def bloop():
    time.sleep(0.5)
    threading.Thread(target=print, args=("new thread running",)).start()

main()

# Note: if we comment the following lines out, we will get a RuntimeError: can't create new thread at interpreter shutdown.
# apparently, this is due to a bug in python 3.12 https://github.com/python/cpython/issues/113964
for thread in threading.enumerate():
    if thread.daemon or thread is threading.current_thread():
        continue
    thread.join()