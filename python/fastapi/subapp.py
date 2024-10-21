from fastapi import FastAPI
from contextlib import asynccontextmanager
from httpx import AsyncClient
from starlette.routing import Mount



class Node1(FastAPI):
    def __init__(self):
        super().__init__()
        self.client: AsyncClient = None

        @self.get("/node1_info")
        def node1_info():
            return "node1 info"
        
    def start_client(self):
        self.client = AsyncClient()
    
    async def stop_client(self):
        await self.client.aclose()
        


class Node2(FastAPI):
    def __init__(self):
        super().__init__()
        self.client: AsyncClient = None

        @self.get("/node2_info")
        def node2_info():
            return "node2 info"
        
        @self.get("/send_request")
        async def send_request():
            response = await self.client.get("http://localhost:8000/node1/node1_info")
            return response.json()
        
    def start_client(self):
        self.client = AsyncClient()
    
    async def stop_client(self):
        await self.client.aclose()



class Node3(FastAPI):
    def __init__(self):
        @asynccontextmanager
        async def lifespan(app: Node3):
            print("Opening client for node3")
            app.client = AsyncClient()

            for route in app.routes:
                if isinstance(route, Mount):
                    print(f"Opening client for subapp {route.path}")
                    route.app.start_client()

            yield

            for route in app.routes:
                if isinstance(route, Mount):
                    print(f"Closing client for subapp {route.path}")
                    await route.app.stop_client()

            await app.client.aclose()
            print("Closing client for node3")

        super().__init__(lifespan=lifespan)
        self.client = None

        @self.get("/node3_info")
        def node3_info():
            return "node3 info"

        @self.get("/")
        def root():
            return "root"



node1 = Node1()
node2 = Node2()
node3 = Node3()

node3.mount("/node1", node1)
node3.mount("/node2", node2)
