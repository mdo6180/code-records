import uvicorn
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi import Request
import json
from fastapi.staticfiles import StaticFiles



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
app.mount("/app", StaticFiles(directory="static"), name="app")



@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>FastAPI HTTPS Example</title>
            <script src="https://unpkg.com/htmx.org@2.0.4"></script>
            <!-- this meta tag is used to configure htmx to allow for requests from a different origin, this is only because we're using htmx version 2 -->
            <meta name="htmx-config" content='{"selfRequestsOnly": false}' />

            <script src="https://unpkg.com/htmx-ext-sse/dist/sse.js"></script>
        </head>
        <body>
            <div hx-ext="sse" sse-connect="https://localhost:8000/event_stream" sse-swap="ChangeColor">
                <h1 id="title" style="color:blue;">Server Sent Events</h1>
            </div>
            <script src="/app/main.js"></script>
        </body>
    </html>
    """


@app.get("/event_stream")
async def event_stream(request: Request):
    async def change_color_stream():
        text_color = "blue"
        while True:
            try:
                await asyncio.sleep(2)

                if text_color == "blue":
                    text_color = "red"
                else:
                    text_color = "blue"
                
                data = {
                    "color": text_color,
                    "source": "source1",
                    "target": "target1"
                }
                    
                yield "event: ChangeColor\n" 
                yield f"data: {json.dumps(data)}\n\n"
            
            except asyncio.CancelledError:
                print("Event stream cancelled.")
                break

    return StreamingResponse(change_color_stream(), media_type="text/event-stream")


config = uvicorn.Config(
    app=app, 
    host=host, 
    port=port, 
    ssl_keyfile="private.key", 
    ssl_certfile="certificate.pem"
)
server = uvicorn.Server(config)


# Note: to run this example, run the following command in the terminal:
# python main.py
if __name__ == "__main__":
    try:
        server.run()
    except KeyboardInterrupt:
        # Swallow it or log cleanly
        print("KeyboardInterrupt: Lifespan cancelled during shutdown.")