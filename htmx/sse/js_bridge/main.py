from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi import Request
import asyncio
import json


app = FastAPI()
app.mount("/static", StaticFiles(directory="../../static"), name="static")
app.mount("/app", StaticFiles(directory="static"), name="app")



@app.get("/", response_class=HTMLResponse)
def home():
    return """<html>
        <head>
            <meta charset="UTF-8">
            <title>Anacostia Console</title>
            
            <!-- non-minified Htmx -->
            <script src="/static/js/htmx.js" type="text/javascript"></script>

            <!-- htmx SSE extension -->
            <script src="/static/js/sse.js"></script>
        </head>
        <body>
            <div hx-ext="sse" sse-connect="/event_stream" sse-swap="ChangeColor">
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
                yield "event: close\n"
                break

    return StreamingResponse(change_color_stream(), media_type="text/event-stream")