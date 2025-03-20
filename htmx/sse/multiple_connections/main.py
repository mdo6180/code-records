from typing import Union

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi import Request
from fastapi.responses import StreamingResponse

import asyncio


def homepage():
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
            <div>Event 0</div>
            <div hx-ext="sse" sse-connect="/event" sse-swap="Event0" hx-swap="beforeend"></div>
            
            <div>Event 1</div>
            <div hx-ext="sse" sse-connect="/event" sse-swap="Event1" hx-swap="beforeend"></div>
        </body>
    </html>
    """


def chat_data(i: int, connection: int):
    if connection == 0:
        return f"""
        <div style="color:blue;">Connection 0 ({i}).</div>
        <div style="color:blue;">More content to swap into your HTML page ({i}).</div>
        """
    else:
        return f"""
        <div style="color:red;">Connection 1 ({i}).</div>
        <div style="color:red;">More content to swap into your HTML page ({i}).</div>
        """


def format_html_for_sse(html_content: str) -> str:
    # Split the HTML content into lines
    lines = html_content.split('\n')

    # Prefix each line with 'data: ' and join them back into a single string with newlines
    formatted_content = "\n".join(f"data: {line}" for line in lines if line.strip()) + "\n\n"

    return formatted_content


app = FastAPI()
app.mount("/static", StaticFiles(directory="../../static"), name="static")


@app.get("/", response_class=HTMLResponse)
def chat_page():
    return homepage()

@app.get("/event")
async def chat(connection: int = 0):
    async def chatroom():
        i = 0

        while True:
            try:
                await asyncio.sleep(1)
                
                print("update")
                
                yield "event: Event0\n" 
                yield format_html_for_sse(chat_data(i, connection=0))
                
                yield "event: Event1\n" 
                yield format_html_for_sse(chat_data(i, connection=1))

                i += 1
            
            except asyncio.CancelledError:
                yield "event: close\n"
                break

    return StreamingResponse(chatroom(), media_type="text/event-stream")