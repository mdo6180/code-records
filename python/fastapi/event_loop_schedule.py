import asyncio
import threading
from fastapi import FastAPI
from contextlib import asynccontextmanager


# ---------- ASYNC TASK DEFINITION ----------
async def async_job(name, delay):
    await asyncio.sleep(delay)
    print(f"[{name}] Finished after {delay} seconds")
    return f"{name} done"


# ---------- BACKGROUND THREAD FUNCTION ----------
def submit_jobs_from_thread(thread_id, loop):
    print(f"[Thread-{thread_id}] Starting...")

    for i in range(3):
        job_name = f"Thread-{thread_id}-Job-{i}"
        delay = (i + 1) * 0.5

        print(f"[Thread-{thread_id}] Submitting {job_name}")
        future = asyncio.run_coroutine_threadsafe(async_job(job_name, delay), loop)

        try:
            result = future.result()  # Blocking wait
            print(f"[Thread-{thread_id}] Got result: {result}")
        except Exception as e:
            print(f"[Thread-{thread_id}] Error: {e}")


# ---------- LIFESPAN HOOK ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # get the event loop for the FastAPI app
    loop = asyncio.get_event_loop()

    # Launch multiple background threads that submit to the loop
    for thread_id in range(3):
        t = threading.Thread(target=submit_jobs_from_thread, args=(thread_id, loop), daemon=True)
        t.start()

    print("[Main] All background threads started")
    yield
    print("[Main] FastAPI shutting down")


# ---------- FASTAPI APP ----------
app = FastAPI(lifespan=lifespan)

@app.get("/")
def index():
    return {"status": "FastAPI app running"}
