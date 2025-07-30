from pathlib import Path
import os
import uvicorn
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx



host = "localhost"
port = 8000

# Dynamically find mkcert's local CA
mkcert_ca = Path(os.popen("mkcert -CAROOT").read().strip()) / "rootCA.pem"
mkcert_ca = str(mkcert_ca)


subapp = FastAPI()
@subapp.get("/", response_class=HTMLResponse)
async def read_root():
    return "<h1>Hello from root subapp</h1>"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    except asyncio.CancelledError:
        # Swallow it or log cleanly
        print("Lifespan cancelled during shutdown.")

app = FastAPI(lifespan=lifespan)
app.mount("/subapp", subapp)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://localhost:8000", 
        "https://127.0.0.1:8000",
        "http://localhost:8000", 
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>FastAPI HTTPS Example</title>
            <script src="https://unpkg.com/htmx.org@2.0.4"></script>
            <!-- this meta tag is used to configure htmx to allow for requests from a different origin, this is only because we're using htmx version 2 -->
            <meta name="htmx-config" content='{"selfRequestsOnly": false}' />
        </head>
        <body>
            <h1 id="header">Hello from the Root FastAPI Application!</h1>
            <div id=data_div></div>
            <button hx-get="https://localhost:8000/subapp" hx-trigger="click" hx-target="#data_div" hx-swap="innerHTML">Click here to see root subapp</button>
            <button hx-get="https://localhost:8001/" hx-trigger="click" hx-target="#data_div" hx-swap="innerHTML">Click here to see leaf main app</button>
            <button hx-get="https://localhost:8001/subapp" hx-trigger="click" hx-target="#data_div" hx-swap="innerHTML">Click to see leaf subapp</button>
        </body>
    </html>
    """

config = uvicorn.Config(
    app=app, 
    host=host, 
    port=port, 
    ssl_ca_certs=mkcert_ca,
    ssl_keyfile="private_root.key", 
    ssl_certfile="certificate_root.pem",
)
server = uvicorn.Server(config)

if __name__ == "__main__":
    try:
        server.run()
    except KeyboardInterrupt:
        server.should_exit = True
        print("Server stopped.")