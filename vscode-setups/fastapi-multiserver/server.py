from fastapi import FastAPI



app = FastAPI()

@app.get("/")
def read_root():
    print("Hello World Server")
    return {"Hello": "World"}
