from fastapi import FastAPI


root_path = "/ged-edap-modelsec/test-container-min-5/anacostia"
app = FastAPI(root_path=root_path)


@app.get("/app")
def read_main():
    return {"message": "Hello World from main app"}


subapi = FastAPI()


@subapi.get("/sub")
def read_sub():
    return {"message": "Hello World from sub API"}


app.mount("/subapi", subapi)

# go to http://anacostia.local/ged-edap-modelsec/test-container-min-5/anacostia/subapi/sub to see the sub API response