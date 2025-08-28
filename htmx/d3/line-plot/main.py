import csv
import json

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


newline = "\n"
app = FastAPI()
app.mount("/js", StaticFiles(directory="./js"), name="js")


@app.get("/", response_class=HTMLResponse)
async def home():
    
    # replace this code with code to read in your own data
    with open("line_plot.csv", "r") as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]
        data_json = json.dumps(data)

    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>HTMX Indicator</title>

            <!-- non-minified Htmx -->
            <script src="https://cdn.jsdelivr.net/npm/htmx.org@2.0.6/dist/htmx.js" integrity="sha384-ksKjJrwjL5VxqAkAZAVOPXvMkwAykMaNYegdixAESVr+KqLkKE8XBDoZuwyWVUDv" crossorigin="anonymous"></script>

            <!-- Add more dependencies here -->
        </head>
        <body>
            <!-- Load d3.js -->
            <script src="https://d3js.org/d3.v6.js"></script>

            <!-- Create a div where the graph will take place -->
            <div id="my_dataviz"></div>

            <script src="/js/line_chart.js" data-graph='{data_json}'></script>
        </body>
    </html>
    """
