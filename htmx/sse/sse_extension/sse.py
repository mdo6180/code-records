from typing import Union

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi import Request
from fastapi.responses import StreamingResponse

import asyncio

from chat_sse import homepage, chat_data



app = FastAPI()
app.mount("/static", StaticFiles(directory="../../static"), name="static")


@app.get("/", response_class=HTMLResponse)
def chat_page():
    return homepage()

def format_html_for_sse(html_content: str) -> str:
    # Split the HTML content into lines
    lines = html_content.split('\n')

    # Prefix each line with 'data: ' and join them back into a single string with newlines
    formatted_content = "\n".join(f"data: {line}" for line in lines if line.strip()) + "\n\n"

    return formatted_content

@app.get("/event-source")
async def chat(request: Request):
    async def chatroom():
        i = 0

        while True:
            try:
                await asyncio.sleep(1)
                print("update")
                yield "event: EventName\n" 
                yield format_html_for_sse(chat_data(i))
                i += 1
            
            except asyncio.CancelledError:
                yield "event: close\n"
                break

    return StreamingResponse(chatroom(), media_type="text/event-stream")