from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse

import asyncio

from hyperscript_component import homepage



app = FastAPI()
app.mount("/static", StaticFiles(directory="../../static"), name="static")


@app.get("/", response_class=HTMLResponse)
def chat_page():
    return homepage()

@app.get("/event-source")
async def chat(request: Request):
    print("connected")
    async def chatroom():
        i = 0

        while True:
            try:
                await asyncio.sleep(1)

                print("event sent")
                if i % 2 == 0:
                    yield "event: HelloEvent\n" 
                    yield "data: hello\n\n"
                else:
                    yield "event: GoodbyeEvent\n" 
                    yield "data: goodbye\n\n"

                i += 1
            
            except asyncio.CancelledError:
                yield "event: close\n"
                break

    return StreamingResponse(chatroom(), media_type="text/event-stream")
