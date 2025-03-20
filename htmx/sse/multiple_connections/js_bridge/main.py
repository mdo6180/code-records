from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi import Request
import asyncio
import json


app = FastAPI()
app.mount("/static", StaticFiles(directory="../../../static"), name="static")
app.mount("/app", StaticFiles(directory="static"), name="app")



@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>Anacostia Console</title>
            
            <!-- non-minified Htmx -->
            <script src="/static/js/htmx.js" type="text/javascript"></script>

            <!-- htmx SSE extension -->
            <script src="/static/js/sse.js"></script>
        </head>
        <body hx-ext="sse" sse-connect="/event_stream">
            <div sse-swap="ChangeColor">
                <h1 id="title" style="color:blue;">Server Sent Events</h1>
            </div>

            <div sse-swap="NewElement" hx-swap="beforeend"></div>

            <script src="/app/main.js"></script>
        </body>
    </html>
    """


def format_html_for_sse(html_content: str) -> str:
    # Split the HTML content into lines
    lines = html_content.split('\n')

    # Prefix each line with 'data: ' and join them back into a single string with newlines
    formatted_content = "\n".join(f"data: {line}" for line in lines if line.strip()) + "\n\n"

    return formatted_content


i = 0

@app.get("/event_stream")
async def event_stream(request: Request):
    async def change_color_stream():
        global i
        text_color = "blue"
        while True:
            try:
                await asyncio.sleep(2)

                if text_color == "blue":
                    text_color = "red"
                else:
                    text_color = "blue"
                
                data = {
                    "color": text_color
                }
                    
                # ChangeColor event is used for js bridge
                yield "event: ChangeColor\n" 
                yield f"data: {json.dumps(data)}\n\n"

                # NewElement event is swapped directly into the second div in the page
                yield "event: NewElement\n"
                yield format_html_for_sse(
                    f"""
                    <div style="color:{text_color};">Content ({i}).</div>
                    <div style="color:{text_color};">More content to swap into your HTML page ({i}).</div>
                    """
                )

                i += 1

            except asyncio.CancelledError:
                yield "event: close\n"
                break

    return StreamingResponse(change_color_stream(), media_type="text/event-stream")