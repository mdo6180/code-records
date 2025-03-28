from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles



newline = "\n"
app = FastAPI()
app.mount("/static", StaticFiles(directory="./static"), name="static")



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
            <link rel="stylesheet" href="/static/table.css">
        </head>
        <body>
            <button hx-get="/quotes" hx-trigger="click" hx-target="#quotes" hx-swap="innerHTML">Get Quotes</button>
            <div id="quotes"></div>

            <button hx-get="/table" hx-trigger="click" hx-target="#table" hx-swap="innerHTML">Get Table</button>
            <div id="table"></div>
        </body>
    </html>
    """


@app.get("/quotes", response_class=HTMLResponse)
async def quotes():
    div_str = "<div>hello there</div>"
    return f"""
    <div>
        "{div_str}"
    </div>
    """


@app.get("/table", response_class=HTMLResponse)
async def table():
    return f"""
    "<table>
        <tr>
            <th>Company</th>
            <th>Contact</th>
            <th>Country</th>
        </tr>
        <tr>
            <td>Alfreds Futterkiste</td>
            <td>Maria Anders</td>
            <td>Germany</td>
        </tr>
        <tr>
            <td>Centro comercial Moctezuma</td>
            <td>Francisco Chang</td>
            <td>Mexico</td>
        </tr>
    </table>"
    """