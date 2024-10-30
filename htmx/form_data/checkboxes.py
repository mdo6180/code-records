from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from typing import List, Tuple
from fastapi.staticfiles import StaticFiles


newline = "\n"
app = FastAPI()
app.mount("/static", StaticFiles(directory="../static"), name="static")


@app.get("/", response_class=HTMLResponse)
def home():
    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>Anacostia Console</title>

            <!-- non-minified Htmx -->
            <script src="/static/js/htmx.js" type="text/javascript"></script>
	</head>
        <body>
            <form>
                <input type="checkbox" id="vehicle1" name="checkboxes" value="vehicle1:Bike">
                <label for="vehicle1"> I have a bike</label><br>
                <input type="checkbox" id="vehicle2" name="checkboxes" value="vehicle2:Car">
                <label for="vehicle2"> I have a car</label><br>
                <input type="checkbox" id="vehicle3" name="checkboxes" value="vehicle3:Boat">
                <label for="vehicle3"> I have a boat</label><br>
                <button type="submit" hx-post="/form_input" hx-trigger="click" hx-target="#response_div" hx-swap="innerHTML">Submit</button>
            </form>

            <div id="response_div">
            </div>
        </body>
    </html>
    """


@app.post("/form_input", response_class=HTMLResponse)
def checkboxes_function(checkboxes: List[str] = Form(default=[])):
    
    checks: List[List[str]] = [value.split(":") for value in checkboxes]
    checks: List[Tuple[str]] = [(value[0], value[1]) for value in checks]
    
    return f"""
        {
            newline.join(
                [ f"<div>{key} : {value}</div>"for key, value in checks ]
            )
        }
    """