import asyncio
import threading
import time

# Shared task object
shared_task = None

def thread1_target():
    global shared_task
    loop1 = asyncio.new_event_loop()
    asyncio.set_event_loop(loop1)

    async def coro():
        await asyncio.sleep(1)
        print("hello from loop1")
        return "done"

    # Schedule the coroutine on this loop but DO NOT run it
    shared_task = loop1.create_task(coro())

    # Keep the loop running so the task is pending
    loop1.run_forever()

# Start thread1 and its loop
threading.Thread(target=thread1_target, daemon=True).start()

# Wait a moment for thread1 to schedule the task
time.sleep(0.5)

# Try to await the task from a different loop (should raise error)
async def main():
    global shared_task
    print("Trying to await a pending Task created on another loop...")
    result = await shared_task  # ❌ RuntimeError expected
    print("Result:", result)

try:
    asyncio.run(main())
except Exception as e:
    print("Caught exception:", type(e).__name__, "-", e)
