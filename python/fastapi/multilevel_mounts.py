from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn
import httpx
import threading
import signal
import time
import asyncio



class NodeApp(FastAPI):
    def __init__(self, name: str, host: str, port: int, *args, **kwargs):

        # lifespan context manager for spinning up and down the Service
        @asynccontextmanager
        async def lifespan(app: NodeApp):
            print(f"Opening client for {app.name}")

            yield

            print(f"Closing client for {app.name}")
            await app.client.aclose()
            
        super().__init__(lifespan=lifespan, *args, **kwargs)
        self.name = name
        self.host = host
        self.port = port
        self.client = httpx.AsyncClient()

        @self.get("/")
        async def root():
            return {"message": f"Hello from {self.name}!"}
    
    def run(self):
        config = uvicorn.Config(self, host=self.host, port=self.port)
        server = uvicorn.Server(config)
        fastapi_thread = threading.Thread(target=server.run)

        original_sigint_handler = signal.getsignal(signal.SIGINT)

        def _kill_webserver(sig, frame):
            print(f"\nCTRL+C Caught!; Killing {self.name} Webservice...")
            server.should_exit = True
            fastapi_thread.join()
            print(f"Webservice {self.name} Killed...")

            # register the original default kill handler once the pipeline is killed
            signal.signal(signal.SIGINT, original_sigint_handler)
        
        # register the kill handler for the webserver
        signal.signal(signal.SIGINT, _kill_webserver)
        fastapi_thread.start()



class Node(threading.Thread):
    def __init__(self, name: str, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.app = NodeApp(name="node1", host="localhost", port=8000)
        self.shutdown = False
        self.lock = threading.Lock()
    
    async def run_loop(self):
        while not self.shutdown:
            await asyncio.sleep(1)
            print(f"Node {self.name} is running...", flush=True)
    
    def stop(self):
        with self.lock:
            self.shutdown = True

    def run(self):
        asyncio.run(self.run_loop())


if __name__ == "__main__":
    """
    node1 = NodeApp(name="node1", host="localhost", port=8000)
    node1.run()

    while True:
        time.sleep(1)
        print("Main thread is running...")
    """

    node = Node(name="node1")
    node.start()
    time.sleep(5)
    node.stop()
    node.join()