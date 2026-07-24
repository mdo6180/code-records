from pathlib import Path
import os
import json


def create_large_file(output_file: Path, size: int):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as f:
        remaining = size
        chunk_size = 1024 * 1024  # Write 1 MiB at a time

        while remaining > 0:
            n = min(chunk_size, remaining)
            f.write(os.urandom(n))
            remaining -= n

    print(f"Created {output_file} ({size:,} bytes)")

# Size of the file in bytes (400 MiB)
FILE_SIZE_LARGE = 50 * 1024 * 1024
FILE_SIZE_MEDIUM = 10 * 1024 * 1024  # 100 MiB
FILE_SIZE_SMALL = 1 * 1024 * 1024  # 1 MiB

large_file = Path("./mydata") / "contents" / "large_50mb.bin"
medium_file = Path("./mydata") / "contents" / "medium_10mb.bin"
small_file = Path("./mydata") / "contents" / "small_1mb.bin"

create_large_file(large_file, FILE_SIZE_LARGE)
create_large_file(medium_file, FILE_SIZE_MEDIUM)
create_large_file(small_file, FILE_SIZE_SMALL)

with open(Path("./mydata") / "manifest.json", "w") as f:
    json.dump({"description": "This is a test bag with large files."}, f, indent=4)