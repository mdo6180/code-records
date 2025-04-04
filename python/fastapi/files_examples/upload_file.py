# sender.py - A script to send files to a FastAPI receiver
import httpx
import asyncio
from pathlib import Path
import os


root_dir = "files"

# Function to send a file to the receiving server
async def upload_file(file_path: str):

    file_path = str(Path(file_path).absolute())     # /Users/minhquando/Desktop/code-records/python/fastapi/files_examples/files/dir1/dir1.txt
    folder_path = str(Path(root_dir).absolute())    # /Users/minhquando/Desktop/code-records/python/fastapi/files_examples/files

    # Check if file exists
    if os.path.exists(file_path) is False:
        print(f"Error: File not found - {file_path}")
        return
    
    filename = file_path.removeprefix(folder_path)  # /dir1/dir1.txt
    filename = filename.lstrip("/")                # dir1/dir1.txt
    
    try:
        # Get the filename from the path
        print(f"Sending file: {filename}")
        
        # Send the file to the receiving server
        async with httpx.AsyncClient() as client:
            with open(file_path, "rb") as f:
                files = {"file": (filename, f)}
                response = await client.post("http://localhost:8000/upload-file", files=files)
                print(response.status_code)
            
    except Exception as e:
        print(f"Error: An exception occurred while sending the file: {str(e)}")


# Run the function if script is executed directly
if __name__ == "__main__":
    file_path = "./files/dir1/dir1.txt"  # Replace with the path to your file
    asyncio.run(upload_file(file_path))