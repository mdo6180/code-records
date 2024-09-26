from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import time
import json



app = FastAPI()
app.mount("/static", StaticFiles(directory="../static"), name="static")



class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


@app.get("/", response_class=HTMLResponse)
async def get():
    client_id = int(time.time())

    return f"""
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
            <h2>Your ID: {client_id}</h2>
            <div hx-ext="ws" ws-connect="/ws/{client_id}">
                <div id="chat_room"></div>
                <form id="form" ws-send>
                    <input name="chat_message">
                    <button>Send</button>
                </form>
            </div>
        </body>
    </html>
    """


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            data = json.loads(data)

            await manager.send_personal_message(
                f"""
                <div id="chat_room" hx-swap-oob="beforeend">
                    <div>You wrote: {data["chat_message"]}</div>
                </div>
                """, 
                websocket
            )

            await manager.broadcast(
                f"""
                <div id="chat_room" hx-swap-oob="beforeend">
                    <div>Client #{client_id} says: {data["chat_message"]}</div>
                </div>
                """
            )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(
            f"""
            <div id="chat_room" hx-swap-oob="beforeend">
                <div>Client #{client_id} left the chat</div>
            </div>
            """
        )
        print(f"Client #{client_id} left the chat")