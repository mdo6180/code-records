from fastapi import FastAPI


app = FastAPI()
@app.get("/")
def read_root():
    return {"message": "Hello World"}

app1 = FastAPI()
@app1.get("/hello_app1")
def read_hello():
    return {"message": "Hello from app1"}

app2 = FastAPI()
@app2.get("/hello_app2")
def read_hello_app2():
    return {"message": "Hello from app2"}

app.mount("/app", app1)
app.mount("/app/app2", app2)


# Note:
# http://localhost:8000/app works
# http://localhost:8000/app/hello_app1 works
# http://localhost:8000/app/app2/hello_app2 does not work
# This is because the path is not correctly mounted in the second app.
# To fix this, we can change the mount path of app1 to "/app/app1"