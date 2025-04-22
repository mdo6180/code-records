from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


newline = "\n"
app = FastAPI()
app.mount("/static", StaticFiles(directory="./static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home():
    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>Multiple JS Files</title>

            <!-- Add more dependencies here -->

        </head>
        <body>
            <button id="clicky">Click here</button>

            <script type="module" src="/static/main.js"></script>
        </body>
    </html> 
    """