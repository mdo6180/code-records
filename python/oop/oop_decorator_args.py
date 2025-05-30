import inspect


class Parent:
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
    
    def save_resource_decorator(self, state: str = "new"):
        def inner(func):
            def wrapper(*args, **kwargs):
                print(f"[{self.name}] saving '{state}' resource in {self.path}")
                func(*args, **kwargs)
                print(f"[{self.name}] saved '{state}' resource in {self.path}")
            return wrapper
        return inner


class Child(Parent):
    def __init__(self, name: str, path: str):
        super().__init__(name=name, path=path)

    def save_file_decorator(self, state: str = "new"):
        def inner(func):

            @self.save_resource_decorator(state=state)
            def wrapper(*args, **kwargs):
                sig = inspect.signature(func)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                filename = bound.arguments.get("filename")

                filepath = f"{self.path}/{filename}"

                print(f"[{self.name}] saving file → {filename} at {filepath}")
                func(*args, **kwargs)
                print(f"[{self.name}] saved file → {filename} at {filepath}")

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
        
        save_new_fn("new.txt", "new hello")
        print("---------------------------")
        save_old_fn("old.txt", "old hello")
        print("---------------------------")
        save_fn("current.txt", "current hello")



if __name__ == "__main__":
    child = Child("Minh", "/path/to/resources")
    child.child_method()
