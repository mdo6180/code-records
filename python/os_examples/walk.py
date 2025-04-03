import os


def walk_directory(root_folder):
    """
    Walk through a directory and print the names of files and directories.
    """

    for root, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            print(f"File: {os.path.join(root, filename)}")


def no_root(root_folder):
    """
    Walk through a directory without the root.
    """

    for root, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            filepath = os.path.join(root, filename).removeprefix(root_folder)
            print(f"File: {filepath}")
    
    print(f"Root: {root_folder}")


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "sample_dir")
    print(f"Path: {path}")
    no_root(path)