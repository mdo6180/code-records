from functools import wraps

def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("hello")
        result = func(*args, **kwargs)
        print("goodbye")
        return result
    return wrapper

class Parent:
    def parent_decorator(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        return wrapper

    @parent_decorator
    def parent_method(self):
        print("parent here")

class Child(Parent):
    # Inherits my_decorator from Parent
    def parent_method(self): 
        print("child here")

if __name__ == "__main__":
    child = Child()
    child.parent_method()

    print("----------------")
    parent = Parent()
    parent.parent_method()
