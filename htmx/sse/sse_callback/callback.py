from typing import Union

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi import Request
from fastapi.responses import StreamingResponse

import asyncio

from chat_callback import homepage, chat_data



app = FastAPI()
app.mount("/static", StaticFiles(directory="../../static"), name="static")


@app.get("/", response_class=HTMLResponse)
def chat_page():
    return homepage()

n = 0
@app.get("/chat-row", response_class=HTMLResponse)
def chat_row():
    global n
    html_str = chat_data(n)
    n += 1
    return html_str

@app.get("/event-source")
async def chat(request: Request):
    async def chatroom():
        while True:
            try:
                await asyncio.sleep(1)
                print("update")
                yield "event: EventName\n" 
                yield "data:\n\n"
            
            except asyncio.CancelledError:
                yield "event: close\n"
                break

    return StreamingResponse(chatroom(), media_type="text/event-stream")

