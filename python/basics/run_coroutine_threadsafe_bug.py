import asyncio

def run_loop_and_schedule_task():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def hello():
        print("Hello from coroutine")
        return "done"

    # ❌ WRONG: This blocks forever because you're calling `run_coroutine_threadsafe` from the same thread that owns the loop
    future = asyncio.run_coroutine_threadsafe(hello(), loop)
    
    # Start the loop (but it never gets to run)
    print("Running loop forever...")
    loop.run_forever()

    result = future.result()
    print("Result:", result)        # This line is never reached because the loop is blocked

run_loop_and_schedule_task()

# Note: in Anacostia, this is a similar issue where the main thread is running the FastAPI app,
# and we're trying to run a coroutine that was scheduled in the same event loop (i.e., the FastAPI app's event loop).
# The FilesystemStoreGUI class tries to call self.metadata_store_client.get_entries() which internally uses run_coroutine_threadsafe,
# but since it's called from the same thread that owns the event loop, it blocks forever.
# This is a common pitfall when using run_coroutine_threadsafe in the same thread as the event loop.
# The correct approach is to ensure that run_coroutine_threadsafe is called from a different thread than the one that owns the event loop.
# In Anacostia, this is handled by ensuring each node runs its own event loop in a separate thread, 
# and all other nodes and the FastAPI app submits tasks to that loop using run_coroutine_threadsafe.