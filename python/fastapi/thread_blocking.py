# blocking_example.py
import threading
import time
from fastapi import FastAPI
import uvicorn

app = FastAPI()

def background_worker():
    while True:
        print("[Background Thread] Running...")
        time.sleep(1)

@app.get("/")
def read_root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    # we must start the thread before running uvicorn otherwise...
    thread = threading.Thread(target=background_worker)
    thread.start()
    
    # this will block the main thread
    uvicorn.run(app, host="127.0.0.1", port=8000)
    # Anything after this line will never run unless you run uvicorn in a thread
    