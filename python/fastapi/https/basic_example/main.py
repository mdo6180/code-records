import uvicorn
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import httpx



# note: we can't access the route through 
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
            <button hx-get="https://localhost:8000/get_data" hx-trigger="click" hx-target="#data_div" hx-swap="innerHTML">Click to get data from leaf</button>
        </body>
    </html>
    """


@app.get("/get_data", response_class=HTMLResponse)
async def get_data():
    return """<div>hello there</div>"""


config = uvicorn.Config(
    app=app, 
    host=host, 
    port=port, 
    ssl_keyfile="private.key", 
    ssl_certfile="certificate.pem"
)
server = uvicorn.Server(config)

if __name__ == "__main__":
    server.run()