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
