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
                    <p id="p_div">checkbox buttons:</p>
                    <label class="checkbox">
                        <input type="checkbox" name="option" value="Option 1" checked hx-post="/checkbox/?option=1" hx-trigger="click" hx-target="#p_div" hx-swap="innerHTML">
                        Option 1
                    </label><br>
                    <label class="checkbox">
                        <input type="checkbox" name="option" value="Option 2" checked hx-post="/checkbox/?option=2" hx-trigger="click" hx-target="#p_div" hx-swap="innerHTML">
                        Option 2
                    </label><br>
                </form>
            </div>
        </body>
    </html>
    """


@app.post("/checkbox/", response_class=HTMLResponse)
async def post(option: int):
    return f"""
    <div>
        <div>
            option selected: {option}
        </div>
    </div>
    """