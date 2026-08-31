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
            <script
                src="https://cdn.jsdelivr.net/npm/htmx.org@4.0.0/dist/htmx.min.js"
                integrity="sha384-BvJpBiO8Kh31EqtJe5DRIeWrHWnCGkwytKs9NKFi86Hhw96dEqdEMzZDeK9iEGTc" 
                crossorigin="anonymous">
            </script>

            <!-- Add more dependencies here -->
        </head>
        <body>
        </body>
    </html>
    """
    return home_html