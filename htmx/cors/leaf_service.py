from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn



# Create a FastAPI instance
app = FastAPI()
app.mount("/static", StaticFiles(directory="../static"), name="static")

# Define a route for the root URL
@app.get("/snippet", status_code=200, response_class=HTMLResponse)
async def read_root():
    return """
    <p>Snippet From Leaf Service</p>
    """



if __name__ == "__main__":
    config = uvicorn.Config("leaf_service:app", host="192.168.100.2", port=8002, log_level="info")
    server = uvicorn.Server(config)
    server.run()