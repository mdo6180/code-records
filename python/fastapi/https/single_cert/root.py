import uvicorn
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse



host = "127.0.0.1"
port = 8000

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
            <script src="https://unpkg.com/htmx.org@2.0.4"></script>
            <!-- this meta tag is used to configure htmx to allow for requests from a different origin, this is only because we're using htmx version 2 -->
            <meta name="htmx-config" content='{"selfRequestsOnly": false}' />
        </head>
        <body>
            <h1 id="header">Hello from the Root FastAPI Application!</h1>
            <button hx-get="https://localhost:8001/" hx-trigger="click" hx-target="#header" hx-swap="outerHTML">Click here to see leaf</button>
        </body>
    </html>
    """

config = uvicorn.Config(
    app=app, 
    host=host, 
    port=port, 
    ssl_keyfile="../private_root.key", 
    ssl_certfile="../certificate_root.pem"
)
server = uvicorn.Server(config)

if __name__ == "__main__":
    try:
        server.run()
    except KeyboardInterrupt:
        # Swallow it or log cleanly
        print("KeyboardInterrupt: Lifespan cancelled during shutdown.")