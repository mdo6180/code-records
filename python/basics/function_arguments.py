import os
from typing import Callable



class ArtifactStore:
    """
    A class to manage the storage of artifacts.
    
    Attributes:
        root_dir (str): The root directory for storing artifacts.
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir

        if os.path.exists(self.root_dir) is False:
            os.makedirs(self.root_dir)
            print(f"Directory {self.root_dir} created.")
        else:
            print(f"Directory {self.root_dir} already exists.")


    def save_artifact(self, func: Callable, filepath: str, *args, **kwargs):
        """
        Save a file using the provided function and filepath.
        
        Args:
            func (callable): Function to save the file. Function must accept a filepath as the first argument.
            filepath (str): Name of the file to save.
            **kwargs: Additional keyword arguments for the function.
        """

        # Ensure the root directory exists
        folder_path = os.path.join(self.root_dir, os.path.dirname(filepath))
        if os.path.exists(folder_path) is False:
            os.makedirs(folder_path)
        
        artifact_save_path = os.path.join(self.root_dir, filepath)

        # Call the function with the filename and additional keyword arguments
        func(artifact_save_path, *args, **kwargs)

        print(f"File saved at: {artifact_save_path}")



# example 1: save a file with a function using args
def example_func(filepath: str, content: str):
    with open(filepath, "w") as f:
        f.write(content)
    
artifact_store = ArtifactStore(root_dir="artifacts")
artifact_store.save_artifact(
    example_func,
    "example_dir/example_file.txt",
    "This is an example content."
)


# example 2: save a file with a function using kwargs
def example_func2(filepath: str, content1: str, content2: str):
    with open(filepath, "w") as f:
        f.write(content1)
        f.write("\n")
        f.write(content2)

artifact_store.save_artifact(
    func=example_func2,
    filepath="example_dir/example_file2.txt",
    content1="This is another example content.",
    content2="This is the second line of content."
)