from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


newline = "\n"
app = FastAPI()
app.mount("/static", StaticFiles(directory="../../static"), name="static")
app.mount("/scripts", StaticFiles(directory="./scripts"), name="scripts")
app.mount("/css", StaticFiles(directory="./css"), name="css")


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

            <!-- Add more dependencies here -->
            <link rel="stylesheet" type="text/css" href="/css/slide_bar.css">
        </head>
        <body>
            <div id="target_div">Slide bar value: 50</div>
            <div id="submitted_value_div"></div>
            <div class="slidecontainer">
                <form>
                    <input type="range" style="width: 100%;" min="1" max="100" value="50" class="slider" id="myRange" name="slide_bar">
                    <button type="submit" hx-post="/submit" hx-trigger="click" hx-target="#submitted_value_div" hx-swap="innerHTML">Send</button>
                </form>
            </div>
            <script src="/scripts/slide_bar.js" type="text/javascript"></script>
        </body>
    </html>
    """

@app.post("/submit", response_class=HTMLResponse)
async def submit(slide_bar: int = Form(...)):
    print(slide_bar)
    return f"Submitted slide bar value: {slide_bar}"