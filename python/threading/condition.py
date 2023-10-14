import threading

num_threads = 5
condition = threading.Condition()
active_threads = 0

def worker(thread_id: int):
    global active_threads
    
    with condition:
        active_threads += 1
        if active_threads == num_threads:
            condition.notify()
        else:
            condition.wait()
            
    with lock:
        # critical section
	print(f"thread {i} using lock")
        
lock = threading.Lock() 

threads = []
for i in range(num_threads - 1):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()
    
with condition:
    if active_threads < num_threads - 1:
        condition.wait()
        
with lock:
    # Main thread critical section
    print("parent using lock")
    
for t in threads:
    t.join()
