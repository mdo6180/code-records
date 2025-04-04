# sender.py - A script to send files to a FastAPI receiver
import httpx
import asyncio
from pathlib import Path
import sys


# Function to send a file to the receiving server
async def upload_file(file_path):
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        print(f"Error: File not found - {file_path}")
        return
    
    try:
        # Get the filename from the path
        filename = file_path.name
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
    file_path = "./files/sample_file.txt"  # Replace with the path to your file
    asyncio.run(upload_file(file_path))