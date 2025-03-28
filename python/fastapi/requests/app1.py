from fastapi import FastAPI
from fastapi.requests import Request
import requests


# run app on port 8080
app = FastAPI()


@app.get("/")
async def home(request: Request):
    host = request.url.hostname
    port = request.url.port
    scheme = request.url.scheme

    print(request.client.host)
    print(request.client.port)

    return f"host: {host}, port: {port}, scheme: {scheme}"
