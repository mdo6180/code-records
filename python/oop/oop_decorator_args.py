import inspect


class Parent:
    def __init__(self, name: str):
        self.name = name
    
    def save_resource_decorator(self, state: str = "new"):
        def inner(func):
            def wrapper(*args, **kwargs):
                print(f"[{self.name}] saving '{state}' resource")
                func(*args, **kwargs)
                print(f"[{self.name}] saved '{state}' resource")
            return wrapper
        return inner


class Child(Parent):
    def __init__(self, name: str):
        super().__init__(name=name)
    
    def save_file_decorator(self, state: str = "new"):
        def inner(func):

            @self.save_resource_decorator(state=state)
            def wrapper(*args, **kwargs):
                sig = inspect.signature(func)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                filename = bound.arguments.get("filename")

                print(f"[{self.name}] saving file → {filename}")
                func(*args, **kwargs)
                print(f"[{self.name}] saved file → {filename}")

            return wrapper
        return inner
    
    def child_method(self):

        @self.save_resource_decorator()
        def save_new_fn(filename: str, data: str):
            print(f"{data} is in {filename}: new")

        @self.save_resource_decorator(state="old")
        def save_old_fn(filename: str, data: str):
            print(f"{data} is in {filename}: old")

        @self.save_file_decorator(state="current")
        def save_fn(filename: str, data: str):
            print(f"{data} is in {filename}: current")
        
        save_new_fn("/path/to/new/file.txt", "new hello")
        print("---------------------------")
        save_old_fn("/path/to/new/file.txt", "old hello")
        print("---------------------------")
        save_fn("/path/to/current/file.txt", "current hello")
    


if __name__ == "__main__":
    child = Child("Minh")
    child.child_method()
