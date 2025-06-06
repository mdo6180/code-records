from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


newline = "\n"

app1 = FastAPI()
app1.mount("/static", StaticFiles(directory="./static1"), name="static")

app2 = FastAPI()
app2.mount("/static", StaticFiles(directory="./static2"), name="static")


app = FastAPI()
app.mount("/app1", app1)
app.mount("/app2", app2)

@app.get("/", response_class=HTMLResponse)
async def home():
    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>Multiple StaticFiles</title>

            <link rel="stylesheet" href="/app1/static/style.css">
            <link rel="stylesheet" href="/app2/static/style.css">
        </head>
        <body>
            <div id="app1">App 1</div>
            <div id="app2">App 2</div>
        </body>
    </html> 
    """