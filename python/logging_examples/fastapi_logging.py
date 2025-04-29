import uvicorn
from fastapi import FastAPI
import logging
import os
from contextlib import asynccontextmanager


log_path = "./logs/test.log"
access_log_path = "./logs/access.log"


if os.path.exists(log_path) is True:
    os.remove(log_path)

if os.path.exists(access_log_path) is True:
    os.remove(access_log_path)


log_file_handler = logging.FileHandler(log_path)
log_file_handler.setLevel(logging.INFO)
log_formatter = logging.Formatter(
    fmt='ROOT %(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log_file_handler.setFormatter(log_formatter)
logger = logging.getLogger(__name__)
logger.addHandler(log_file_handler)
logger.setLevel(logging.INFO)  # Make sure it's at least INFO

# Step 1: Create a file handler for access logs
access_log_file_handler = logging.FileHandler(access_log_path)
access_log_file_handler.setLevel(logging.INFO)  # Usually INFO is enough

# Step 2: (optional) Set format for access logs
access_log_formatter = logging.Formatter(
    fmt='ACCESS %(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
access_log_file_handler.setFormatter(access_log_formatter)

# Step 3: Attach the handler to the "uvicorn.access" logger
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.addHandler(access_log_file_handler)
uvicorn_access_logger.setLevel(logging.INFO)  # Make sure it's at least INFO

# Optional (good practice): don't propagate to root logger
uvicorn_access_logger.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    if app.logger is not None:
        app.logger.info("Root server started")
    
    yield
    
    if app.logger is not None:
        app.logger.info("Root server closed")

app = FastAPI(lifespan=lifespan)
app.logger = logger


@app.get("/")
async def func():
    logger.info(f"request / endpoint!")
    return {"message": "hello world!"}


if __name__ == "__main__":
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, access_log=False)
    server = uvicorn.Server(config)
    server.run()