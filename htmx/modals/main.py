from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


newline = "\n"

app = FastAPI()
app.mount("/static", StaticFiles(directory="../static"), name="static")
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

            <!-- CSS for Modal -->
            <link rel="stylesheet" href="/css/modal.css">
        </head>
        <body>
            <button hx-get="/modal/open" hx-target="#modal-container">Open Modal</button>
            <div id="modal-container"></div>
        </body>
    </html>
    """


@app.get("/modal/{action}", response_class=HTMLResponse)
async def modal(action: str):
    if action == "open":
        return f"""
        <div class="modal-overlay" hx-get="/modal/close" hx-trigger="click target:.modal-overlay" hx-target="#modal-container">
            <div class="modal-content">
                Hello!
            </div>
        </div>
        """
    elif action == "close":
        return ""
