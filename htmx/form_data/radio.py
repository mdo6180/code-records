from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


app = FastAPI()
app.mount("/static", StaticFiles(directory="../static"), name="static")



@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>htmx POST request</title>

            <!-- non-minified Htmx -->
            <script src="/static/js/htmx.js" type="text/javascript"></script>
        </head>
        <body>
            <div id="target_div">
                <form>
                    <p>Radio buttons:</p>
                    <label class="radio">
                        <input type="radio" name="option" value="Option 1" checked>
                        Option 1
                    </label><br>
                    <label class="radio">
                        <input type="radio" name="option" value="Option 2" >
                        Option 2
                    </label><br>
                    <br>
                    <button type="submit" hx-post="/hx_post" hx-trigger="click" hx-target="#target_div" hx-swap="innerHTML">Send</button>
                </form>
            </div>
        </body>
    </html>
    """


@app.post("/hx_post", response_class=HTMLResponse)
async def post(option: str = Form(...)):
    return f"""
    <div>
        <div>
            option selected: {option}
        </div>
    </div>
    """