from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import shutil
import os
from pathlib import Path
import httpx


# Directory to store received files
STORAGE_DIR = "downloaded_files/"
if os.path.exists(STORAGE_DIR) is False:
    os.makedirs(STORAGE_DIR)
else:
    if os.path.exists(f"./{STORAGE_DIR}sample_file.txt"):
        os.unlink(f"./{STORAGE_DIR}sample_file.txt")


app = FastAPI()


@app.post("/upload-file")
async def receive_file(file: UploadFile):
    print(f"Received file: {file.filename}")
    try:
        # Save the received file
        file_path = os.path.join(STORAGE_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"File saved to: {file_path}")
    
    except Exception as e:
        print(f"An error occurred while receiving: {str(e)}")


@app.get("/download-file/{filepath:path}")
async def download_file(filepath: str):
    """
    Endpoint to download a file.
    """

    try:

        # Validate the file exists
        if os.path.exists(filepath) is False:
            raise HTTPException(status_code=404, detail=f"File not found: {filepath}")
        
        # Return the file as a response
        print(f"Downloading file: {filepath}")
        return FileResponse(
            path=filepath,
            media_type="application/octet-stream"
        )
        
    except HTTPException as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")