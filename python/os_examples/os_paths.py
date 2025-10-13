import os
import pathlib

resource_path = "/path/to" 
artifact_path = "file.txt"
full_path = os.path.join(resource_path, artifact_path)
print(full_path)  # Output: /path/to/file.txt

resource_path = "/path/to/" 
artifact_path = "/path/to/file.txt"
full_path = os.path.join(resource_path, artifact_path)
print(full_path)  # Output: /path/to/file.txt

resource_path = "/path/to/" 
artifact_path = "/to/file.txt"
full_path = os.path.join(resource_path, artifact_path)
print(full_path)  # Output: /path/to/file.txt

resource_path = "/path/to"
artifact_path = "subdir/file.txt"
full_path = os.path.join(resource_path, artifact_path)
print(full_path)  # Output: /path/to/subdir/file.txt

resource_path = pathlib.Path("/path/to/")
artifact_path = pathlib.Path("/subdir/file.txt")
full_path = resource_path / artifact_path
print(full_path)  # Output: /path/to/subdir/file.txt