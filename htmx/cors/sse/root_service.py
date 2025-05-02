from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn



# Create a FastAPI instance
app = FastAPI()
app.mount("/static", StaticFiles(directory="../../static"), name="static")

# Define a route for the root URL
@app.get("/", status_code=200, response_class=HTMLResponse)
async def read_root():
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
            <div hx-ext="sse" sse-connect="http://127.0.0.1:8001/event-source" sse-swap="EventName" hx-swap="beforeend"></div>
        </body>
    </html>
    """


if __name__ == "__main__":
    config = uvicorn.Config("root_service:app", host="127.0.0.1", port=8000, log_level="info")
    server = uvicorn.Server(config)
    server.run()
