from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles



app = FastAPI()
app.mount("/static", StaticFiles(directory="../static"), name="static")
app.mount("/css", StaticFiles(directory="./css"), name="css")


@app.get("/notification/{type}", response_class=HTMLResponse)
async def notification(type: str):

    if type == "success":
        title = "Success!"
        message = "Your action was completed successfully."
    elif type == "warning":
        title = "Warning!"
        message = "This action might cause issues."
    elif type == "error":
        title = "Error!"
        message = "Something went wrong. Please try again."
    elif type == "info":
        title = "Info"
        message = "This is an informational message."
    else:
        return ""

    return f"""
    <div class="notification {type}" hx-get="/notification/delete" hx-trigger="load delay:10s" hx-target="this" hx-swap="delete">
        <div class="notification-content">
            <div class="notification-title">{title}</div>
            <div class="notification-message">{message}</div>
        </div>
        <button class="close-button" hx-get="/notification/delete" hx-trigger="click" hx-target="closest .notification" hx-swap="delete">✕</button>
    </div>
    """


@app.get("/", response_class=HTMLResponse)
async def home():
    return f"""
    <!DOCTYPE html>
    <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Popup Notification</title>
            
            <!-- non-minified Htmx -->
            <script src="/static/js/htmx.js" type="text/javascript"></script>

            <link rel="stylesheet" href="/css/styles.css">
        </head>
        <body>
            <div class="notification-container" id="notification-container"></div>

            <h2>Click a button to show a notification</h2>

            <div class="button-group">
                <button class="type-button success-btn" hx-get="/notification/success" hx-trigger="click" hx-target="#notification-container" hx-swap="afterbegin">Success</button>
                <button class="type-button warning-btn" hx-get="/notification/warning" hx-trigger="click" hx-target="#notification-container" hx-swap="afterbegin">Warning</button>
                <button class="type-button error-btn" hx-get="/notification/error" hx-trigger="click" hx-target="#notification-container" hx-swap="afterbegin">Error</button>
                <button class="type-button info-btn" hx-get="/notification/info" hx-trigger="click" hx-target="#notification-container" hx-swap="afterbegin">Info</button>
            </div>
        </body>
    </html> 
    """