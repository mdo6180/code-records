"""
Note: merge algorithm is as follows:
- Elements that exist in the current head as exact textual matches will be left in place
- Elements that do not exist in the current head will be added at the end of the head tag
- Elements that exist in the current head, but not in the new head will be removed from the head

Note: in order to enable the merge algorithm, the hx-head="merge" attribute must be added to the new head tag

Note: If you place hx-head="re-eval" on a head element, it will be re-added (removed and appended) to the head tag on every request, even if it already exists. 
This can be useful to execute a script on every htmx request. This can also be useful for ensuring certain elements are placed at the bottom of the head tag.
In this demo, the main.css file is re-evaluated on every request in order to keep main.css at the bottom of the head tag.

Note: The hx-preserve="true" attribute is used to prevent the element from being removed from the head tag during the merge algorithm.

Note: when running this demo, open the browser console to see how the head tag changes.

Note: for more information, see https://htmx.org/extensions/head-support/
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


newline = "\n"
app = FastAPI()
app.mount("/static", StaticFiles(directory="../static"), name="static")
app.mount("/css", StaticFiles(directory="./css"), name="css")


@app.get("/", response_class=HTMLResponse)
async def home():
    return f"""
    <!DOCTYPE html>
    <html>
        <head hx-head="merge">
            <meta hx-preserve="true" charset="UTF-8">
            <title hx-preserve="true">HTMX Indicator</title>
            <script hx-preserve="true" src="/static/js/htmx.js" type="text/javascript"></script>
            <script hx-preserve="true" src="https://unpkg.com/htmx-ext-head-support@2.0.1/head-support.js"></script>
            <link hx-head="re-eval" rel="stylesheet" href="/css/main.css">
        </head>
        <body hx-ext="head-support">
            <h1 id="header">No styling</h1>
            <button hx-get="/style1" hx-trigger="click" hx-target="#header" hx-swap="outerHTML">Click to see /style1 styling</button>
            <button hx-get="/style2" hx-trigger="click" hx-target="#header" hx-swap="outerHTML">Click to see /style2 styling</button>
            <button hx-get="/no_styling" hx-trigger="click" hx-target="#header" hx-swap="outerHTML">Click to remove all styling</button>
        </body>
    </html>
    """

@app.get("/no_styling", response_class=HTMLResponse)
async def no_styling():
    return f"""
    <head hx-head="merge">
        <meta hx-preserve="true" charset="UTF-8">
        <title hx-preserve="true">HTMX Indicator</title>
        <script hx-preserve="true" src="/static/js/htmx.js" type="text/javascript"></script>
        <script hx-preserve="true" src="https://unpkg.com/htmx-ext-head-support@2.0.1/head-support.js"></script>
        <link hx-head="re-eval" rel="stylesheet" href="/css/main.css">
    </head> 
    <h1 id="header">No styling</h1>
    """

@app.get("/style1", response_class=HTMLResponse)
async def home1():
    return f"""
    <head hx-head="merge">
        <meta hx-preserve="true" charset="UTF-8">
        <title hx-preserve="true">HTMX Indicator</title>
        <script hx-preserve="true" src="/static/js/htmx.js" type="text/javascript"></script>
        <script hx-preserve="true" src="https://unpkg.com/htmx-ext-head-support@2.0.1/head-support.js"></script>
        <link rel="stylesheet" href="/css/button_style1.css">
        <link hx-head="re-eval" rel="stylesheet" href="/css/main.css">
    </head>
    <h1 id="header">/style1 styling</h1>
    """


@app.get("/style2", response_class=HTMLResponse)
async def home2():
    return f"""
    <head hx-head="merge">
        <meta hx-preserve="true" charset="UTF-8">
        <title hx-preserve="true">HTMX Indicator</title>
        <script hx-preserve="true" src="/static/js/htmx.js" type="text/javascript"></script>
        <script hx-preserve="true" src="https://unpkg.com/htmx-ext-head-support@2.0.1/head-support.js"></script>
        <link rel="stylesheet" href="/css/button_style2.css">
        <link hx-head="re-eval" rel="stylesheet" href="/css/main.css">
    </head>
    <h1 id="header">/style2 styling</h1>
    """
