from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from admin import router


app = FastAPI()
app.mount("/static", StaticFiles(directory="../static"), name="static")
app.include_router(router, tags=["AFSIM Control"])


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>HTMX Indicator</title>
            
            <!-- non-minified Htmx -->
            <script src="/static/js/htmx.js" type="text/javascript"></script>
        </head>
        <body>
            <div id="top">
                <button hx-get="/hello" hx-trigger="click" hx-target="#top" hx-swap="afterbegin">click me</button>
            </div>
        </body>
    </html>
    """