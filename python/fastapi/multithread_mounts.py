import uvicorn
from fastapi import FastAPI
from threading import Thread, Event
import time
from typing import List



class Node(Thread):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.event = Event()
    
    def run(self):
        while self.event.is_set() is False:
            print(f"Node thread running {self.name}...")
            time.sleep(1)
    


class NodeApp(FastAPI):
    def __init__(self, name: str, node: Node, host: str, port: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.node = node
        self.host = host
        self.port = port

        @self.get("/")
        async def root():
            return {"message": f"Hello from {self.name}!"}

        config = uvicorn.Config(self, host=self.host, port=self.port)
        self.server = uvicorn.Server(config)

    def run(self):
        self.node.start()

    def stop(self):
        print(f"Stopping {self.name}...")
        self.node.event.set()
        self.node.join()



class GraphApp(FastAPI):
    def __init__(self, name: str, host: str, port: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.host = host
        self.port = port

        self.nodeapps: List[NodeApp] = []

        @self.get("/")
        async def root():
            return {"message": f"Hello from {self.name}!"}
        
        @self.get("/stop/{node_name}")
        async def stop(node_name: str):
            for nodeapp in self.nodeapps:
                if nodeapp.name == node_name:
                    nodeapp.stop()
                    self.nodeapps.remove(nodeapp)

            return {"message": f"Stopped {node_name}..."}

        config = uvicorn.Config(self, host=self.host, port=self.port)
        self.server = uvicorn.Server(config)

    def run(self):
        self.thread = Thread(target=self.server.run)
        self.thread.start()

        for i in range(1, 4):
            nodeapp = NodeApp(name=f"node{i}", node=Node(name=f"node{i}"), host="localhost", port=8000)
            nodeapp.node.start()
            self.nodeapps.append(nodeapp)
            self.mount(f"/node{i}", app=nodeapp)
    
    def stop(self):
        for nodeapp in self.nodeapps:
            nodeapp.stop()

        self.server.should_exit = True
        self.thread.join()



app = GraphApp(name="graph", host="localhost", port=8000)
app.run()

while True:
    try:
        time.sleep(1)
        print("Main thread running...")
    except KeyboardInterrupt:
        print("Stopping...")
        app.stop()
        break