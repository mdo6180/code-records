import asyncio


class A:
    def __init__(self) -> None:
        pass

    def method(self):
        print("A method")
    
    def run(self):
        self.method()


class B(A):
    def __init__(self) -> None:
        super().__init__()

    # note: a method is overridden even though the parent version of the method is not async
    async def method(self):
        print("B method")
    
    async def run(self):
        super().method()
        await self.method() 


if __name__ == "__main__":
    b = B()
    asyncio.run(b.run())