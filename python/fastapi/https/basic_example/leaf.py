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



@app.get("/", response_class=HTMLResponse)
async def read_leaf():
    return "<h1>Hello from leaf</h1>"

config = uvicorn.Config(
    app=app, 
    host=host, 
    port=port, 
    ssl_keyfile="../private_leaf.key", 
    ssl_certfile="../certificate_leaf.pem"
)
server = uvicorn.Server(config)

if __name__ == "__main__":
    server.run()