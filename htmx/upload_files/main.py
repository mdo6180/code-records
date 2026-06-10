from pathlib import Path
import hashlib

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

html = str      # alias of the str type for syntax highlighting using the Python Inline Source Syntax Highlighting extension by Sam Willis in VSCode.


UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

newline = "\n"
app = FastAPI()
app.mount("/static", StaticFiles(directory="../static"), name="static")
app.mount("/js", StaticFiles(directory="./js"), name="js")


@app.get("/", response_class=HTMLResponse)
async def home():
    home_html: html = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="UTF-8">
            <title>HTMX Indicator</title>

            <!-- non-minified Htmx -->
            <script src="/static/js/htmx.js" type="text/javascript"></script>

            <!-- Add more dependencies here -->
            <script src="/js/upload.js" type="text/javascript"></script>
        </head>
        <body>
            <form id='form' hx-post='/upload' hx-encoding='multipart/form-data' hx-target="#upload-result" hx-swap="innerHTML">
                <input type='file' name='file'>
                <button>
                    Upload
                </button>
                <progress id='progress' value='0' max='100'></progress>
            </form>
            <div id="upload-result"></div>
        </body>
    </html>
    """
    return home_html


@app.post("/upload", response_class=HTMLResponse)
async def create_upload_file(file: UploadFile = File(...)):
    destination = UPLOAD_DIR / file.filename
    sha256 = hashlib.sha256()       # Optional: Create a SHA-256 hash object

    with open(destination, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            f.write(chunk)
            sha256.update(chunk)    # Optional: Update the hash with the chunk data

    await file.close()

    file_hash = sha256.hexdigest()  # Optional: Get the hexadecimal representation of the hash

    return f"""
    <p>Uploaded file: <strong>{file.filename}</strong></p>
    <p>SHA-256: <strong>{file_hash}</strong></p>
    """