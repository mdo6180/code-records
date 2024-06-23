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
            <div id="node1-div" hx-get="/content/node1" hx-trigger="click from:#node1-btn" hx-swap-oob="true"></div>
            <div id="node2-div" hx-get="/content/node2" hx-trigger="click from:#node2-btn" hx-swap-oob="true"></div>
            <button id="node1-btn" hx-get="/button/node1" hx-trigger="click" hx-swap-oob="true">Click node1</button>
            <button id="node2-btn" hx-get="/button/node2" hx-trigger="click" hx-swap-oob="true">Click node2</button>
        </body>
    </html>
    '''


@app.get("/button/{node_clicked}", response_class=HTMLResponse)
def button(node_clicked: str):
    if node_clicked == "node1":
        return """
        <button id="node1-btn" hx-get="/button/node1" hx-trigger="click" hx-swap-oob="true">node1 Clicked</button>
        <button id="node2-btn" hx-get="/button/node2" hx-trigger="click" hx-swap-oob="true">Click node2</button>
        """
    else:
        return """
        <button id="node1-btn" hx-get="/button/node1" hx-trigger="click" hx-swap-oob="true">Click node1</button>
        <button id="node2-btn" hx-get="/button/node2" hx-trigger="click" hx-swap-oob="true">node2 Clicked</button>
        """



@app.get("/content/{node_clicked}", response_class=HTMLResponse)
def content(node_clicked: str):
    if node_clicked == "node1":
        return """
        <div id="node1-div" hx-get="/content/node1" hx-trigger="click from:#node1-btn" hx-swap-oob="true">Clicked node1</div>
        <div id="node2-div" hx-get="/content/node2" hx-trigger="click from:#node2-btn" hx-swap-oob="true"></div>
        """
    else:
        return """
        <div id="node1-div" hx-get="/content/node1" hx-trigger="click from:#node1-btn" hx-swap-oob="true"></div>
        <div id="node2-div" hx-get="/content/node2" hx-trigger="click from:#node2-btn" hx-swap-oob="true">Clicked node2</div>
        """