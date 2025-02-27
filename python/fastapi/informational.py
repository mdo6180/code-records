from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {
                margin: 100px;
            }

            .info-icon-container {
                position: relative;
                display: inline-block;
            }

            .info-icon {
                font-size: 18px;
                width: 24px;
                height: 24px;
                border: 2px solid #007bff; /* Blue color common for info icons */
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #007bff;
                font-family: Arial, sans-serif;
                font-weight: bold;
                cursor: help; /* Indicates clickable info */
                background-color: transparent;
                transition: background-color 0.2s; /* Smooth hover transition */
            }

            .info-icon:hover {
                background-color: #e6f0ff; /* Light blue hover effect */
            }

            .tutorial-tooltip {
                display: none;
                position: absolute;
                background-color: #ffffff;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 4px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                top: calc(100% + 5px); /* Small gap below icon */
                left: 50%;
                transform: translateX(-50%);
                width: 200px; /* Fixed width for tutorial text */
                font-family: Arial, sans-serif;
                font-size: 14px;
                color: #333;
                z-index: 10;
            }

            .info-icon-container:hover .tutorial-tooltip {
                display: block;
            }

            /* Optional: Add a small triangle pointer */
            .tutorial-tooltip::before {
                content: '';
                position: absolute;
                top: -6px;
                left: 50%;
                transform: translateX(-50%);
                border-left: 6px solid transparent;
                border-right: 6px solid transparent;
                border-bottom: 6px solid #fff;
            }
        </style>
    </head>
    <body>
        <div class="info-icon-container">
            <div class="info-icon">i</div>
            <div class="tutorial-tooltip">
                Welcome! Click here to start your guided tour of the website.
            </div>
        </div>
    </body>
    </html>
    """