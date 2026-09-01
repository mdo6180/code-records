from fastapi import FastAPI
from fastapi.responses import HTMLResponse

html = str   # alias of the str type for syntax highlighting using the Python Inline Source Syntax Highlighting extension by Sam Willis in VSCode.


newline = "\n"
app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def home():
    '''
    Instead of placing the hx-confirm on each button (like below), 
    '''
    buttons_html: html = f"""
        <button hx-delete="/account" hx-confirm="Are you sure?">
            Delete My Account
        </button>
        <button hx-put="/account" hx-confirm="Are you sure?">
            Update My Account
        </button>
    """

    '''
    We can hoist this attribute to a parent element using the :inherited modifier on the attribute:
    '''
    buttons_html: html = f"""
        <div hx-confirm:inherited="Are you sure?">
            <button hx-delete="/account">
                Delete My Account
            </button>
            <button hx-put="/account">
                Update My Account
            </button>
        </div>
    """

    home_html: html = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>HTMX Indicator</title>

            <!-- non-minified Htmx -->
            <script
                src="https://cdn.jsdelivr.net/npm/htmx.org@4.0.0/dist/htmx.min.js"
                integrity="sha384-BvJpBiO8Kh31EqtJe5DRIeWrHWnCGkwytKs9NKFi86Hhw96dEqdEMzZDeK9iEGTc" 
                crossorigin="anonymous">
            </script>

            <!-- Add more dependencies here -->
        </head>
        <body>
            {buttons_html}
        </body>
    </html>
    """
    return home_html


@app.delete("/account", response_class=HTMLResponse)
async def delete_account():
    delete_html: html = f"""
    <div>
        <h1>Account Deleted</h1>
        <p>Your account has been successfully deleted.</p>
    </div>
    """
    return delete_html


@app.put("/account", response_class=HTMLResponse)
async def update_account():
    update_html: html = f"""
    <div>
        <h1>Account Updated</h1>
        <p>Your account has been successfully updated.</p>
    </div>
    """
    return update_html