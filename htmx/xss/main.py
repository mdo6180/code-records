from html import escape

import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/htmx.org@2.0.7"></script>
    </head>

    <body>

        <h1>HTMX XSS Demo</h1>

        <h2>Unsafe</h2>

        <form hx-post="/unsafe" hx-target="#unsafe_result" hx-swap="innerHTML">
            <input type="text" name="message" style="width:400px">
            <button>Submit</button>
        </form>

        <div id="unsafe_result" style="border:1px solid black;padding:10px;margin-bottom:40px;"></div>

        <h2>Safe</h2>

        <form hx-post="/safe" hx-target="#safe_result" hx-swap="innerHTML">
            <input type="text" name="message" style="width:400px">
            <button>Submit</button>
        </form>

        <div id="safe_result" style="border:1px solid black;padding:10px;"></div>

    </body>
    </html>
    """


@app.post("/unsafe", response_class=HTMLResponse)
def unsafe(message: str = Form(...)):
    return f"""
    <p>User entered:</p>

    <div style="padding:10px;border:1px solid red">
        {message}
    </div>
    """


@app.post("/safe", response_class=HTMLResponse)
def safe(message: str = Form(...)):
    return f"""
    <p>User entered:</p>

    <div style="padding:10px;border:1px solid green">
        {escape(message)}
    </div>
    """
