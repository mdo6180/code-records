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
                    <label for="message">Message:</label>
                        <input type="text" name="message" placeholder="Type a message">
                    </label><br>
                    <br>
                    <button type="submit" hx-post="/hx_post" hx-trigger="click" hx-target="#target_div" hx-swap="innerHTML">Send</button>
                </form>
            </div>
        </body>
    </html>
    """


@app.post("/hx_post", response_class=HTMLResponse)
async def post(message: str = Form(...)):
    return f"""
    <div>
        <div>
            message received: {message}
        </div>
    </div>
    """