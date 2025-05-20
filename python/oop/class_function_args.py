from typing import Callable, Any



def default_save(filepath: str, model: Any):
    print(f"default saved {filepath}: model type: {type(model)}")

def special_save(filepath: str, model: Any):
    print(f"specialize saved {filepath}: model type: {type(model)}")

def default_load(filepath: str, model: Any):
    print(f"default loaded {filepath}: model type: {type(model)}")

def special_load(filepath: str, model: Any):
    print(f"specialize loaded {filepath}: model type: {type(model)}")



class Resource:
    def load_file(self, filepath: str, load_fn: Callable[[str, Any], None], *args, **kwargs):
        load_fn(filepath, *args, **kwargs)
    
    def save_file(self, filepath: str, save_fn: Callable[[str, Any], None], *args, **kwargs):
        save_fn(filepath, *args, **kwargs)


class ModelRegistry(Resource):
    def load_file(self, filepath: str, load_fn: Callable[[str, Any], None] = special_load, *args, **kwargs):
        load_fn(filepath, *args, **kwargs)
    
    def save_file(self, filepath: str, save_fn: Callable[[str, Any], None] = special_save, *args, **kwargs):
        save_fn(filepath, *args, **kwargs)



resource = Resource()
resource.load_file("/hello/there", load_fn=default_load, model="hello")

registry = ModelRegistry()
registry.load_file("/hello/there", model=8)
registry.load_file("/hello/there", load_fn=default_load, model=8)