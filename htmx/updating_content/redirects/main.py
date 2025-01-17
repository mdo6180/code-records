from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.routing import Mount


newline = "\n"


app = FastAPI()
app.mount("/static", StaticFiles(directory="../../static"), name="static")

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

            <!-- When we redirect back to the home page, we can also remove dependencies -->
        </head>
        <body>
            <button hx-get="/redirect/?path=/app2" hx-swap="none" hx-trigger="click" hx-target="this">Click to go to /app2 home page</button>
        </body>
    </html>
    """

@app.get("/redirect/")
async def redirect(path: str, response: Response):
    response.headers["HX-Redirect"] = path



app2 = FastAPI()
app2.mount("/css", StaticFiles(directory="./css"), name="css")

@app2.get("/", response_class=HTMLResponse)
async def home2():
    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>HTMX Indicator</title>

            <!-- non-minified Htmx -->
            <script src="/static/js/htmx.js" type="text/javascript"></script>

            <!-- When we do the redirect, we can also add new dependencies -->
            <link rel="stylesheet" href="/app2/css/app2.css">
        </head>
        <body>
            <button hx-get="/redirect/?path=/" hx-swap="none" hx-trigger="click" hx-target="this">Click to go back home</button>
        </body>
    </html>
    """

app.mount("/app2", app2)