import time
import random
import threading


class Queue:
    def __init__(self):
        self.queue = []
    
    def front(self):
        if len(self.queue) > 0:
            return self.queue[0]

    def enque(self, element):
        self.queue.append(element)

    def deque(self):
        self.queue.pop(0)

    def __len__(self):
        return len(self.queue)

    def __str__(self):
        return str(self.queue)


def f(b, batch, queue, results):

    # following line simulates GPU operation 
    # (sending data to GPU, inference/activation maps extraction, sending data back to CPU)
    time.sleep(random.randint(1, 5))

    print("{} woke at: {}".format(threading.current_thread().getName(), time.ctime()))

    b.wait()
    
    while len(queue) > 0:
        if queue.front() == threading.current_thread().getName():

            batchID = threading.current_thread().getName()[-1]

            # following line simulates appending resulting activation maps recieved from threads into resulting numpy array
            results.append(batchID)

            queue.deque()
            break

    print("{} passed the barrier at: {}".format(threading.current_thread().getName(), time.ctime()))


def main_function():
    queue = Queue()
    results = []

    barrier = threading.Barrier(3)
    threads = []
    for i in range(3):
        # replace batch variable with numpy array recieved from dataloader
        batch = "batch " + str(i)

        threadID = "thread-{}".format(i)

        t = threading.Thread(target=f, args=(barrier, batch, queue, results))
        t.setName(threadID)
        queue.enque(threadID)
        threads.append(t)

        t.start()

    print(queue)

    for thread in threads:
        thread.join()

    # following line will be replaced with converting results into numpy array 
    # and saving the results as .npy file
    print(results)


if __name__ == "__main__":
    main_function()

