from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn



# Create a FastAPI instance
app = FastAPI()
app.mount("/static", StaticFiles(directory="../static"), name="static")

# Define a route for the root URL
@app.get("/", status_code=200, response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Root Service</title>
        <script src="/static/js/htmx.js" type="text/javascript"></script>
        </head>
        <body>
            <h1>Root Service</h1>
            <div id="snippet"></div>
            <button hx-get="http://192.168.100.2:8002/snippet" hx-target="#snippet" hx-swap="innerHTML" hx-trigger="click">Get Snippet</button>
        </body>
    </html>
    """


if __name__ == "__main__":
    config = uvicorn.Config("root_service:app", host="127.0.0.1", port=8000, log_level="info")
    server = uvicorn.Server(config)
    server.run()
