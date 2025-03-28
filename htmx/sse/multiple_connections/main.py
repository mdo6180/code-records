from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse

import asyncio


# Note: in the case where we need to display the same value of the variable i in both the Event0 and the Event1 div,
# we can either set an sse connection on the body (or some parent div) and then have one div listen to Event0 and another div listen to Event1;
# or we can have one div listen to Event0 and another div listen to Event1 but execute an SSE callback

# Note: when we call the http://localhost:8000/next endpoint, both divs will update with variable i incrementing by 2.
# this can be avoided by keeping the SSE connection open (i.e., not swapping out the body)


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
        <body hx-ext="sse" sse-connect="/event">
            <div>Event 0</div>
            <div sse-swap="Event0" hx-swap="beforeend"></div>
            
            <div>Event 1</div>
            <!--
            <div hx-get="/callback" hx-trigger="sse:Event1" hx-swap="beforeend"></div>
            -->
            <div sse-swap="Event1" hx-swap="beforeend"></div>
        </body>
    </html>
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
async def chat_page():
    return homepage()


@app.get("/next", response_class=HTMLResponse)
async def next():
    return homepage()


i = 0


@app.get("/callback", response_class=HTMLResponse)
async def callback():
    global i
    return f"""
    <div style="color:red;">Connection 1 ({i}).</div>
    <div style="color:red;">More content to swap into your HTML page ({i}).</div>
    """


@app.get("/event")
async def chat():
    async def chatroom():
        print("connection opened")

        global i

        while True:
            try:
                await asyncio.sleep(1)
                
                print("update")
                
                yield "event: Event0\n" 
                yield format_html_for_sse(
                    f"""
                    <div style="color:blue;">Connection 0 ({i}).</div>
                    <div style="color:blue;">More content to swap into your HTML page ({i}).</div>
                    """
                )
                
                yield "event: Event1\n" 
                yield format_html_for_sse(
                    f"""
                    <div style="color:red;">Connection 1 ({i}).</div>
                    <div style="color:red;">More content to swap into your HTML page ({i}).</div>
                    """
                )

                i += 1
            
            except asyncio.CancelledError:
                yield "event: close\n"
                print("connection closed")
                return

    return StreamingResponse(chatroom(), media_type="text/event-stream")