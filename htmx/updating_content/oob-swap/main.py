from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles



app = FastAPI()
app.mount("/static", StaticFiles(directory="../../static"), name="static")



@app.get("/", response_class=HTMLResponse)
def home():
    return '''<html>
        <head>
            <meta charset="UTF-8">
            <title>Anacostia Console</title>
            
            <!-- non-minified Htmx -->
            <script src="/static/js/htmx.js" type="text/javascript"></script>
        </head>
        <body>
            <div id="oob">Not Clicked</div>
            <button hx-get="/button" hx-swap="outerHTML" hx-target="this" hx-trigger="click">Click Me</button>
        </body>
    </html>
    '''


@app.get("/button", response_class=HTMLResponse)
def button():
    return f'''
        <!-- 
        Note: setting hx-swap-oob="true" will work as well because true and outerHTML is equivalent.
        Thus, <div id="oob" hx-swap-oob="outerHTML">Clicked</div> or <div id="oob" hx-swap-oob="true">Clicked</div> will also work here
        -->
        <div id="oob" hx-swap-oob="outerHTML:#oob">Clicked</div>
        <button hx-get="/button" hx-swap="outerHTML" hx-target="this" hx-trigger="click">Clicked</button>
    '''


@app.get("/content", response_class=HTMLResponse)
def content():
    return "some text"