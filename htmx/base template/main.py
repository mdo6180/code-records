from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

html = str   # alias of the str type for syntax highlighting using the Python Inline Source Syntax Highlighting extension by Sam Willis in VSCode.


newline = "\n"
app = FastAPI()
app.mount("/static", StaticFiles(directory="../static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home():
    home_html: html = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>HTMX Indicator</title>

            <!-- non-minified Htmx -->
            <script src="/static/js/htmx.js" type="text/javascript"></script>

            <!-- Add more dependencies here -->
        </head>
        <body>
        </body>
    </html>
    """
    return home_html