import httpx
import asyncio
from pathlib import Path
import sys

# Configuration
SENDER_SERVER_URL = "http://localhost:8000/download-file/"

remote_filepath = "files/sample_file.txt"  # Replace with the path to your file
local_filepath = "downloaded_files/sample_file.txt"  # Replace with the path to save the downloaded file
url = f"{SENDER_SERVER_URL}{remote_filepath}"



async def download_file(url):
    """
    Download a file from the given URL.
    """
    try:
        async with httpx.AsyncClient() as client:

            # Stream the response to handle large files efficiently
            async with client.stream("GET", url) as response:
                if response.status_code != 200:
                    print(f"Error: Server returned status code {response.status_code}")
                    print(f"Response: {await response.text()}")
                    return False
                
                else:
                    print(f"Downloading file from {url}...")

                    # Create the file and write the content chunk by chunk
                    with open(local_filepath, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            f.write(chunk)
                    
                    print(f"File downloaded successfully: {local_filepath}")
                    return True

    except Exception as e:
        print(f"Error: An exception occurred while downloading the file: {str(e)}")


if __name__ == "__main__":
    asyncio.run(download_file(url))