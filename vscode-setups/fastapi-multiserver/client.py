from fastapi import FastAPI



app = FastAPI()

@app.get("/")
def read_root():
    print("Hello World Client")
    return {"Hello": "World"}