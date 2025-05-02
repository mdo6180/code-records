from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio



# Create a FastAPI instance
app = FastAPI()
app.mount("/static", StaticFiles(directory="../../static"), name="static")

origins = [
    "http://127.0.0.1:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def chat_data(i: int):
    return f"""
    <div>Content to swap into your HTML page ({i}).</div>
    <div>More content to swap into your HTML page ({i}).</div>
    """


def format_html_for_sse(html_content: str) -> str:
    # Split the HTML content into lines
    lines = html_content.split('\n')

    # Prefix each line with 'data: ' and join them back into a single string with newlines
    formatted_content = "\n".join(f"data: {line}" for line in lines if line.strip()) + "\n\n"

    return formatted_content


# Define a route for the root URL
@app.get("/snippet", status_code=200, response_class=HTMLResponse)
async def read_root():
    return """
    <p>Snippet From Leaf Service</p>
    """

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


if __name__ == "__main__":
    config = uvicorn.Config("leaf_service:app", host="127.0.0.1", port=8001, log_level="info")
    server = uvicorn.Server(config)
    server.run()