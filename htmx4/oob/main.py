from fastapi import FastAPI
from fastapi.responses import HTMLResponse

html = str   # alias of the str type for syntax highlighting using the Python Inline Source Syntax Highlighting extension by Sam Willis in VSCode.


newline = "\n"
app = FastAPI()


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
            <div id="oob-target"></div>
            <div id="target"></div>
            <button hx-get="/target1" hx-trigger="click" hx-target="#target" hx-swap="innerHTML">Click 1!</button>
            <button hx-get="/target2" hx-trigger="click" hx-target="#target" hx-swap="innerHTML">Click 2!</button>
        </body>
    </html>
    """
    return home_html


@app.get("/target1", response_class=HTMLResponse)
async def target1():
    target1_html: html = f"""
    <hx-partial hx-target="#oob-target" hx-swap="beforeend">
        <div>Target 1 OOB Content</div>
    </hx-partial>

    <div>
        <h1>Target 1</h1>
        <p>This is the content for Target 1.</p>
    </div>
    """
    return target1_html


@app.get("/target2", response_class=HTMLResponse)
async def target2():
    target2_html: html = f"""
    <hx-partial hx-target="#oob-target" hx-swap="beforeend">
        <div>Target 2 OOB Content</div>
    </hx-partial>

    <div>
        <h1>Target 2</h1>
        <p>This is the content for Target 2.</p>
    </div>
    """
    return target2_html