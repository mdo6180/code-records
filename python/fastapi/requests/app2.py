from fastapi import FastAPI
import requests


# run app on port 8000
app = FastAPI()

@app.get("/")
async def home():
    response = requests.get("http://localhost:8080")
    return response.text

