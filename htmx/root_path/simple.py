from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

html = str   # alias of the str type for syntax highlighting using the Python Inline Source Syntax Highlighting extension by Sam Willis in VSCode.


newline = "\n"

# set the home page URL to http://localhost:8000/htmx/
app = FastAPI()
app.mount("/static", StaticFiles(directory="../static"), name="static")


def html_template(base_url: str = ""):
    home_html: html = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>HTMX</title>

            <!-- <base href="{base_url}"> -->
    
            <!-- non-minified Htmx -->
            <script src="/static/js/htmx.js" type="text/javascript"></script>

            <!-- Add more dependencies here -->
        </head>
        <body>
            <button hx-get="display" hx-target="#display" hx-swap="outerHTML">Get display</button>
            <div id="display"></div>
        </body>
    </html>
    """
    return home_html

@app.get("/", response_class=HTMLResponse)
async def home():
    return html_template("http://localhost:8000/htmx/")

@app.get("/display", response_class=HTMLResponse)
async def display():
    display_html: html = f"""
    <div id="display">
        <h1>HTMX is working!</h1>
    </div>
    """
    return display_html