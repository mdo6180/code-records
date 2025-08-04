import asyncio
import threading
import time

shared_task = None

class Node(threading.Thread):
    def __init__(self, name):
        super().__init__(name=name, daemon=True)
        self.name = name
        self.loop = None

    async def run_async(self):
        async def coro():
            await asyncio.sleep(1)
            print(f"hello from {self.name}")
            return "done"

        global shared_task
        # Schedule coroutine as a task and save reference
        shared_task = asyncio.create_task(coro())

        # Keep the loop alive so the task stays pending
        await asyncio.sleep(5)

    def run(self):
        # Use asyncio.run to create and run the loop
        asyncio.run(self.run_async())

# Start the thread with its own asyncio loop
node = Node("Node-1")
node.start()

# Wait a bit for the coroutine to be scheduled but not finished
time.sleep(0.5)

# Main thread with a different event loop tries to await the task
async def main():
    global shared_task
    print("Trying to await a pending Task created on another loop...")
    result = await shared_task  # ❌ RuntimeError
    print("Result:", result)

try:
    asyncio.run(main())
except Exception as e:
    print("Caught exception:", type(e).__name__, "-", e)

# Note: in Anacostia, the main thread is the one running the FastAPI app,
# thus, this raises an error because the task was created in the Node's thread

# In Anacostia, the problem takes shape in a slightly different form but it's the same problem nonetheless: 
# we are returning raw coroutine objects, not scheduled tasks. 
# These coroutine objects are created inside the node's event loop 
# that's running on a different thread (event loop started via us calling asyncio.run() in the node's run() method), 
# but yet we're awaiting them (via calling .gather()) from the main PipelineServer event loop (aka the FastAPI app's event loop),
# which is more or less the exact same bug we just reproduced earlier: ❌ You cannot await a coroutine or Future bound to a different loop.