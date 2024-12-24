from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import time


newline = "\n"
app = FastAPI()
app.mount("/static", StaticFiles(directory="../static"), name="static")
app.mount("/img", StaticFiles(directory="./img"), name="img")
app.mount("/css", StaticFiles(directory="./css"), name="css")


@app.get("/", response_class=HTMLResponse)
async def home():
    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>HTMX Indicator</title>

            <!-- non-minified Htmx -->
            <script src="/static/js/htmx.js" type="text/javascript"></script>

            <link rel="stylesheet" href="/css/styles.css">
	</head>
        <body>
            <div>
                <button hx-post="/example" hx-indicator="#spinner" hx-target="this" hx-swap="outerHTML" hx-trigger="click">
                    Post It!
                </button>
                <img  id="spinner" class="htmx-indicator" src="/img/spinner2.gif"/>
            </div>
        </body>
    </html>
    """


@app.post("/example", response_class=HTMLResponse)
async def example():
    time.sleep(2)
    return "<div>Example Posted</div>"