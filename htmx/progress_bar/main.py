from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import random


newline = "\n"
app = FastAPI()
app.mount("/static", StaticFiles(directory="../static"), name="static")
app.mount("/css", StaticFiles(directory="./css"), name="css")


percentage = 0


@app.get("/", response_class=HTMLResponse)
async def home():
    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>HTMX Progress Bar</title>

            <!-- non-minified Htmx -->
            <script src="/static/js/htmx.js" type="text/javascript"></script>

            <!-- htmx class tools extension -->
            <script src="https://unpkg.com/htmx-ext-class-tools@2.0.1/class-tools.js"></script>

            <link rel="stylesheet" href="/css/styles.css">
        </head>
        <body>
            <div id="outer_div">
                <h3>Start Progress</h3>
                <button class="btn primary" hx-post="/start" hx-trigger="click" hx-target="#outer_div" hx-swap="outerHTML">Start Job</button>
            </div>
        </body>
    </html>
    """

@app.post("/start", response_class=HTMLResponse)
async def start():
    global percentage
    if percentage >= 100:
        percentage = 0
    
    return f"""
    <div id="outer_div" hx-get="/finished" hx-trigger="done" hx-swap="outerHTML" hx-target="this">
        <h3 role="status" id="pblabel" tabindex="-1" autofocus>Running</h3>

        <div 
            hx-get="/progress" 
            hx-trigger="every 600ms" 
            hx-target="this" 
            hx-swap="innerHTML">

            <div>Progress: {percentage}%</div>
            <div class="progress" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{percentage}" aria-labelledby="pblabel">
                <div id="pb" class="progress-bar" style="width:{percentage}%">
            </div>
        </div>
    </div>
    """

@app.get("/finished", response_class=HTMLResponse)
async def job():
    global percentage
    return f"""
    <div id="outer_div" hx-get="/finished" hx-trigger="done" hx-swap="outerHTML" hx-target="this">
        <h3 role="status" id="pblabel" tabindex="-1" autofocus>Complete</h3>

        <div 
            hx-get="/progress" 
            hx-trigger="none" 
            hx-target="this" 
            hx-swap="innerHTML">

            <div>Progress: {percentage}%</div>
            <div class="progress" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{percentage}" aria-labelledby="pblabel">
                <div id="pb" class="progress-bar" style="width:{percentage}%"></div>
            </div>
        </div>

        <button id="restart-btn" class="btn primary" classes="add show:600ms" hx-post="/start" hx-trigger="click" hx-target="#outer_div" hx-swap="outerHTML">
            Restart Job
        </button>
    </div>
    """

@app.get("/progress", response_class=HTMLResponse)
async def job_progress(response: Response):
    global percentage
    if percentage >= 100:
        percentage = 100
        response.headers["HX-Trigger"] = "done"
    else:
        percentage += random.randint(0, 20)

    return f"""
    <div>Progress: {percentage}%</div>
    <div class="progress" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{percentage}" aria-labelledby="pblabel">
        <div id="pb" class="progress-bar" style="width:{percentage}%"></div>
    </div> 
    """