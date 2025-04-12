# command to run to generate self-signed cert and key
# openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout private_leaf.key -out certificate_leaf.pem -config openssl.cnf

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



@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>FastAPI HTTPS Example</title>
        </head>
        <body>
            <h1>FastAPI HTTPS Example</h1>
            <p>This is a simple leaf FastAPI application running with HTTPS.</p>
        </body>
    </html>
    """

config = uvicorn.Config(
    app=app, 
    host=host, 
    port=port, 
    ssl_keyfile="private_leaf.key", 
    ssl_certfile="certificate_leaf.pem"
)
server = uvicorn.Server(config)

if __name__ == "__main__":
    server.run()