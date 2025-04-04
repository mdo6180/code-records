# sender_streaming.py - Script to send large files with chunked streaming
import httpx
import asyncio
from pathlib import Path
import sys
import os

# Configuration for the receiving server
RECEIVING_SERVER_URL = "http://localhost:8000/receive-file-stream"
ROOT_DIR = "files"          # Directory to store files

# Size of chunks to read and send (4MB)
CHUNK_SIZE = 4 * 1024 * 1024

async def send_file_streaming(file_path):
    file_path = str(Path(file_path).absolute())     # /Users/minhquando/Desktop/code-records/python/fastapi/files_examples/files/dir1/dir1.txt
    folder_path = str(Path(ROOT_DIR).absolute())    # /Users/minhquando/Desktop/code-records/python/fastapi/files_examples/files

    # Check if file exists
    if os.path.exists(file_path) is False:
        print(f"Error: File not found - {file_path}")
        return
    
    filename = file_path.removeprefix(folder_path)  # /dir1/dir1.txt
    filename = filename.lstrip("/")                # dir1/dir1.txt
    
    try:
        filesize = os.path.getsize(file_path)

        print(f"Preparing to upload: {filename} ({filesize/1024/1024:.2f} MB)")
        
        # Set up headers with file metadata
        headers = {
            "X-Filename": filename,
            "Content-Type": "application/octet-stream",
            "Content-Length": str(filesize)
        }

        async def file_generator():
            """Generator function that yields chunks of the file"""
            with open(file_path, "rb") as f:
                while chunk := f.read(CHUNK_SIZE):
                    yield chunk
                    # Optional: Add progress reporting
                    print(f"Sent chunk: {len(chunk)/1024/1024:.2f} MB")
        
        # Send the file using streaming upload
        async with httpx.AsyncClient() as client:
            response = await client.post(
                RECEIVING_SERVER_URL,
                headers=headers,
                content=file_generator(),
                timeout=None  # Disable timeout for large uploads
            )
        
        # Print the response
        if response.status_code == 200:
            print(f"Success: File {filename} sent successfully")
            response_data = response.json()
            print(f"remote storage path: {response_data['stored_path']}")
            return True
        else:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error: An exception occurred while sending the file: {str(e)}")
        return False
    


if __name__ == "__main__":
    file_path = "./files/dir1/dir1.txt"  # Replace with the path to your file
    success = asyncio.run(send_file_streaming(file_path))