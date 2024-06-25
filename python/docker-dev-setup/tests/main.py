from fastapi import FastAPI
from hello import hello_there



app = FastAPI()


@app.get("/")
def hello():
    return hello_there()