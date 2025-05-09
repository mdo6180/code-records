import uvicorn
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware


host = "localhost"
port = 8001

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    except asyncio.CancelledError:
        # Swallow it or log cleanly
        print("Lifespan cancelled during shutdown.")

app = FastAPI(lifespan=lifespan)

origins = [
    "https://127.0.0.1:8000", 
    "https://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Note: visit 
@app.get("/", response_class=HTMLResponse)
async def read_leaf():
    return "<h1>Hello from leaf</h1>"


@app.get("/library_data")
async def library_data():
    return {"phrase": "hello world"}

config = uvicorn.Config(
    app=app, 
    host=host, 
    port=port, 
<<<<<<< HEAD:python/fastapi/https/basic_example/leaf.py
    ssl_keyfile="../private_root.key", 
    ssl_certfile="../certificate_root.pem"
=======
    ssl_keyfile="private_leaf.key", 
    ssl_certfile="certificate_leaf.pem"
>>>>>>> 6eae4d54eb937ed47ba3715f97b05fef9eb2cffc:python/fastapi/https/two_certs/leaf.py
)
server = uvicorn.Server(config)

if __name__ == "__main__":
    try:
        server.run()
    except KeyboardInterrupt:
        # Swallow it or log cleanly
        print("KeyboardInterrupt: Lifespan cancelled during shutdown.")