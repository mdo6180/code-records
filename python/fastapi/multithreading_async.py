from fastapi import FastAPI
import threading
import time


class Node(threading.Thread):
    def __init__(self, name):
        super().__init__(name=name)
        self.exit_event = threading.Event()
    
    def exit(self):
        print(f"Exiting {self.name}")
        self.exit_event.set()

    def run(self):
        print(f"Starting {self.name}")
        while self.exit_event.is_set() is False:
            print(f"hello from {self.name}")
            time.sleep(1)
        print(f"Exited {self.name}")


class AppServer(FastAPI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node = Node("Node1")

        @self.post("/start")
        async def create_pipeline():
            print("Handling /start request")
            self.node.start()
            return "pipeline started"
        
        @self.post("/stop")
        async def stop():
            print("Handling /stop request")
            print(f"Thread is_alive: {self.node.is_alive()}")
            if not self.node.is_alive():
                return "pipeline not running"
            self.node.exit()
            print("Called exit()")
            self.node.join()
            print("Called join()")
            return "pipeline stopped"


app = AppServer()