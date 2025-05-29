import asyncio
import httpx


async def call_endpoint():
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://localhost:8000/items/",
            headers={
                "X-Token": "fake-super-secret-token",
                "X-Key": "fake-super-secret-key",
            }
        )
        print(response.status_code)
        print(response.json())


asyncio.run(call_endpoint())