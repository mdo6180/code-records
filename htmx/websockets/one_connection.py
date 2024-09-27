from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json



app = FastAPI()
app.mount("/static", StaticFiles(directory="../static"), name="static")

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Chat</title>

        <!-- non-minified Htmx -->
        <script src="/static/js/htmx.js" type="text/javascript"></script>

        <!-- htmx websockets extension -->
        <script src="https://unpkg.com/htmx.org@1.9.12/dist/ext/ws.js"></script>
    </head>
    <body>
        <h1>WebSocket Chat</h1>

        <div hx-ext="ws" ws-connect="/chatroom">
            <div id="chat_room"></div>
            <form id="form" ws-send>
                <input name="chat_message">
                <button>Send</button>
            </form>
        </div>
    </body>
</html>
"""


@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/chatroom")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            data = json.loads(data)

            # this will swap in the new content using out-of-band swaps
            await websocket.send_text(
                f"""
                <div id="chat_room" hx-swap-oob="beforeend">
                    <div>Message text was: {data["chat_message"]}</div>
                </div>
                """
            )
    except WebSocketDisconnect:
        pass
