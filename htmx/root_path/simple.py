from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

html = str   # alias of the str type for syntax highlighting using the Python Inline Source Syntax Highlighting extension by Sam Willis in VSCode.


newline = "\n"

# set the home page URL to http://localhost:8000/htmx/
app = FastAPI(root_path="/ged-edap-modelsec/test-container-min-5/anacostia")
app.mount("/js", StaticFiles(directory="js"), name="js")
app.mount("/css", StaticFiles(directory="css"), name="css")

def html_template() -> str:
    base_url = "http://anacostia.local/ged-edap-modelsec/test-container-min-5/anacostia/"

    home_html: html = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>HTMX</title>

            <!-- non-minified Htmx -->
            <script src="js/htmx.js" type="text/javascript"></script>

            <!-- Add more dependencies here -->
            <link rel="stylesheet" href="css/styles.css">
            <script src="js/settings.js" type="text/javascript"></script>
        </head>
        <body>
            <button id="display-btn" hx-get="/display" hx-target="#display" hx-swap="outerHTML">Get display</button>
            <div id="display"></div>
        </body>
    </html>
    """
    return home_html

@app.get("/", response_class=HTMLResponse)
async def home():
    return html_template()

@app.get("/display", response_class=HTMLResponse)
async def display():
    display_html: html = f"""
    <div id="display">
        <h1>HTMX is working!</h1>
    </div>
    """
    return display_html