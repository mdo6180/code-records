import inspect
import asyncio
from typing import Callable, Any


class Parent:
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
    
    def save_resource_decorator(self, state: str = "new"):
        def inner(func):
            async def wrapper(*args, **kwargs):
                print(f"[{self.name}] saving '{state}' resource in {self.path}")
                await func(*args, **kwargs)
                print(f"[{self.name}] saved '{state}' resource in {self.path}")
            return wrapper
        return inner


class Child(Parent):
    def __init__(self, name: str, path: str):
        super().__init__(name=name, path=path)

    def save_file_decorator(self, state: str = "new"):
        def inner(func):
            # Check at decoration time
            sig = inspect.signature(func)
            if "filename" not in sig.parameters:
                raise TypeError(f"Function '{func.__name__}' must have a 'filename' parameter in order to use the @save_file_decorator")

            @self.save_resource_decorator(state=state)
            async def wrapper(*args, **kwargs):

                # We use inspect.signature to bind the arguments to the function and extract the filename
                # this means that we must assume that the function has a parameter named 'filename'
                # Bind call-time arguments to extract filename
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()

                filename = bound.arguments.get("filename")
                if filename is None:
                    raise ValueError(f"'filename' must be passed when calling '{func.__name__}'")

                filepath = f"{self.path}/{filename}"

                print(f"[{self.name}] saving file → {filename} at {filepath}")
                await asyncio.sleep(2)  # Simulating an async operation
                await func(*args, **kwargs)
                print(f"[{self.name}] saved file → {filename} at {filepath}")

            return wrapper
        return inner
    

    async def child_method(self):

        @self.save_resource_decorator()
        async def save_new_fn(filename: str, data: str):
            print(f"{data} is in {filename}: new")

        await save_new_fn("new.txt", "new hello")
        print("---------------------------")

        @self.save_resource_decorator(state="old")
        async def save_old_fn(filename: str, data: str):
            print(f"{data} is in {filename}: old")

        await save_old_fn("old.txt", "old hello")
        print("---------------------------")

        @self.save_file_decorator(state="current")
        async def save_fn(filename: str, data: str):
            print(f"{data} is in {filename}: current")

        await save_fn("current.txt", "current hello")
        print("---------------------------")



async def main():
    child = Child("Minh", "/path/to/resources")
    await child.child_method()


if __name__ == "__main__":
    asyncio.run(main())
