from fastapi import FastAPI, UploadFile, HTTPException, Request, Header
from fastapi.responses import FileResponse, JSONResponse
import shutil
import os
from pathlib import Path
import httpx
from typing import Optional
import asyncio


# Directory to store received files
STORAGE_DIR = "downloaded_files/"
if os.path.exists(STORAGE_DIR) is False:
    os.makedirs(STORAGE_DIR)
else:
    if os.path.exists(f"./{STORAGE_DIR}sample_file.txt"):
        os.unlink(f"./{STORAGE_DIR}sample_file.txt")
    
    if os.path.exists(f"./{STORAGE_DIR}dir1/dir1.txt"):
        os.unlink(f"./{STORAGE_DIR}dir1/dir1.txt")
    


app = FastAPI()


@app.post("/upload-file")
async def receive_file(file: UploadFile):
    print(f"Received file: {file.filename}")
    try:
        folder_path = os.path.join(STORAGE_DIR, os.path.dirname(file.filename))
        if os.path.exists(folder_path) is False:
            os.makedirs(folder_path)

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


@app.post("/upload_stream")
async def upload_stream(
    request: Request,
    x_filename: Optional[str] = Header(None)
):
    try:
        # Use the provided filename or generate one
        if not x_filename:
            x_filename = f"uploaded_file_{asyncio.get_event_loop().time()}"
        
        # Create the file path
        file_path = os.path.join(STORAGE_DIR, x_filename)
        
        # Stream the request body directly to a file
        content_length = request.headers.get("content-length")
        if content_length:
            total_size = int(content_length)
            bytes_received = 0
        else:
            total_size = None
            bytes_received = 0
        
    # Open the file and write chunks as they arrive
        with open(file_path, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
                bytes_received += len(chunk)
                # Optional: Add progress logging here
                if total_size:
                    progress = bytes_received / total_size * 100
                    print(f"Received: {bytes_received/1024/1024:.2f}MB / {total_size/1024/1024:.2f}MB ({progress:.1f}%)")
        
        return JSONResponse(
            content={
                "filename": x_filename,
                "status": "File received and saved successfully",
                "bytes_received": bytes_received,
                "stored_path": str(file_path)
            },
            status_code=200
        )
    
    except Exception as e:
        return JSONResponse(
            content={"error": f"An error occurred while receiving: {str(e)}"},
            status_code=500
        )