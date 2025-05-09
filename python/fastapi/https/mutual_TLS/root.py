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

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    except asyncio.CancelledError:
        # Swallow it or log cleanly
        print("Lifespan cancelled during shutdown.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://localhost:8000", "https://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dynamically find mkcert's local CA
mkcert_ca = Path(os.popen("mkcert -CAROOT").read().strip()) / "rootCA.pem"
mkcert_ca = str(mkcert_ca)
print(mkcert_ca)


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
            <button hx-get="https://localhost:8001/" hx-trigger="click" hx-target="#header" hx-swap="outerHTML">Click here to see leaf</button>
            <button hx-get="https://localhost:8000/get_data" hx-trigger="click" hx-target="#data_div" hx-swap="innerHTML">Click to get data from leaf</button>
        </body>
    </html>
    """


@app.get("/get_data", response_class=HTMLResponse)
async def get_data():
    async with httpx.AsyncClient(verify=mkcert_ca, cert=("certificate_root.pem", "private_root.key")) as client:
        response = await client.get("https://localhost:8001/library_data")
        response = response.json()
        return f"<div>{response['phrase']}</div>"

config = uvicorn.Config(
    app=app, 
    host=host, 
    port=port, 
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