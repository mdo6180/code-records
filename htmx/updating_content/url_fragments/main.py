from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles



app = FastAPI()
app.mount("/static", StaticFiles(directory="../../static"), name="static")


newline = "\n"

@app.get("/", response_class=HTMLResponse)
def home():
    return f'''<html>
        <head>
            <meta charset="UTF-8">
            <title>Anacostia Console</title>
            
            <!-- non-minified Htmx -->
            <script src="/static/js/htmx.js" type="text/javascript"></script>
            
            <!-- implement smooth scrolling -->
            <style>
                html {{ scroll-behavior: smooth; }}
            </style>
        </head>
        <body>
            <form>
                <label for="row">Enter row number:</label>
                    <input type="number" name="row" placeholder="Type a row number">
                </label>
                <button type="submit" hx-post="/hx_post" hx-trigger="click" hx-swap="none">Jump to Row</button>
                <br>
            </form>
            <!-- navigate to row 500 by typing this url into the browser: http://127.0.0.1:8000/#row500 -->
            <table>
                <tr>
                    <th>Info</th>
                    <th>Row #</th>
                </tr>
                {
                    newline.join(
                        [
                            f"""
                            <tr id="row{i}">
                                <td>info {i}</td>
                                <td> {i}</td>
                            </tr>
                            """ for i in range(1000)
                        ])
                }
            </table>
        </body>
    </html>
    '''


@app.post("/hx_post", response_class=HTMLResponse)
async def post(response: Response, row: int = Form(...)):
    response.headers["HX-Redirect"] = f"http://127.0.0.1:8000/#row{str(row)}"