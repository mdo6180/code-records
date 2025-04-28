from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import asyncio
from queue import Queue



app = FastAPI()
app.mount("/static", StaticFiles(directory="../static"), name="static")
app.mount("/css", StaticFiles(directory="./css"), name="css")

notification_queue = Queue()


@app.get("/notification_delete", response_class=HTMLResponse)
async def notification_delete():
    return ""

def send_notification(type: str, title: str, message: str):
    global notification_queue

    if type not in ["success", "warning", "error", "info"]:
        raise ValueError(f"type '{type}' is not a valid type for a notification")

    notification_snippet = f"""
    <div class="notification {type}" hx-get="/notification_delete" hx-trigger="load delay:4s" hx-target="this" hx-swap="delete">
        <div class="notification-content">
            <div class="notification-title">{title}</div>
            <div class="notification-message">{message}</div>
        </div>
        <button class="close-button" hx-get="/notification_delete" hx-trigger="click" hx-target="closest .notification" hx-swap="delete">✕</button>
    </div>
    """

    notification_queue.put_nowait(notification_snippet)


@app.get("/", response_class=HTMLResponse)
async def home():
    return f"""
    <!DOCTYPE html>
    <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Popup Notification</title>
            
            <!-- non-minified Htmx -->
            <script src="/static/js/htmx.js" type="text/javascript"></script>
            <script hx-preserve="true">
                htmx.config.allowNestedOobSwaps = false;
                htmx.config.useTemplateFragments = true;
            </script>

            <!-- htmx head support -->
            <script hx-preserve="true" src="https://unpkg.com/htmx-ext-head-support@2.0.1/head-support.js"></script>

            <!-- htmx SSE extension -->
            <script src="/static/js/sse.js"></script>

            <link rel="stylesheet" href="/css/styles.css">
        </head>
        <body hx-ext="head-support, sse">
            <div class="notification-container" id="notification-container" sse-connect="/notification_events" sse-swap="NotificationEvent" hx-swap="afterbegin"></div>
        </body>
    </html> 
    """

def format_html_for_sse(html_content: str) -> str:
    # Split the HTML content into lines
    lines = html_content.split('\n')

    # Prefix each line with 'data: ' and join them back into a single string with newlines
    formatted_content = "\n".join(f"data: {line}" for line in lines if line.strip()) + "\n\n"

    return formatted_content

@app.get("/notification_events")
async def notification_events(request: Request):
    global notification_queue
    for _ in range(20):
        send_notification("success", title="Success", message="Successful!")

    async def event_stream():
        while await request.is_disconnected() is False:
            try:
                if notification_queue.qsize() > 0:
                    data = notification_queue.get()
                    yield "event: NotificationEvent\n" 
                    yield format_html_for_sse(data)
                    await asyncio.sleep(0.1)
            
            except asyncio.CancelledError:
                yield "event: close\n"
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")