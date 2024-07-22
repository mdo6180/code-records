from fastapi import FastAPI
import uvicorn


node1 = FastAPI()
@node1.get("/info")
def node1_info():
    # http://localhost:8000/graph1/node1/info
    return "node1 info"



node2 = FastAPI()
@node2.get("/info")
def node2_info():
    # http://localhost:8000/graph1/node2/info
    return "node2 info"



node3 = FastAPI()
@node3.get("/info")
def node1_info():
    # http://localhost:8000/graph2/node3/info
    return "node3 info"



node4 = FastAPI()
@node4.get("/info")
def node2_info():
    # http://localhost:8000/graph2/node4/info
    return "node4 info"



graph1 = FastAPI()
graph1.mount("/node1", node1)
graph1.mount("/node2", node2)

@graph1.get("/info")
def graph1_info():
    # http://localhost:8000/graph1/info
    return "graph1 info"



graph2 = FastAPI()
graph2.mount("/node3", node3)
graph2.mount("/node4", node4)

@graph2.get("/info")
def graph1_info():
    # http://localhost:8000/graph2/info
    return "graph2 info"



service = FastAPI()
service.mount("/graph1", graph1)
service.mount("/graph2", graph2)

@service.get("/")
def service_info():
    # http://localhost:8000/
    return "service info"



if __name__ == "__main__":
    config = uvicorn.Config(service, host="localhost", port=8000)
    server = uvicorn.Server(config)
    server.run()