import os
import json
from datetime import datetime
import hashlib

import mlcroissant as mlc


def hash_file(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


resource_path = "dataset"

# 1) One FileObject per concrete file (good for checksums)
distribution = []
for filepath in os.listdir("dataset"):
    fullpath = os.path.join(resource_path, filepath)
    distribution.append(
        mlc.FileObject(
            name=os.path.basename(fullpath),
            content_url=filepath,          # relative path is portable
            encoding_formats=["text/plain"],
            sha256=hash_file(fullpath)
        )
    )

# 2) A FileSet that groups all *.txt files into one logical resource (lets RecordSet refer to them as a single source)
text_files = mlc.FileSet(
    id="text-files",
    name="text-files",
    includes=f"{resource_path}/*.txt",
    encoding_formats=["text/plain"]
)

# 3) RecordSet: each record = one file; expose filename and content
record_set = mlc.RecordSet(
    name="examples",
    description="Each record corresponds to one text file.",
    fields=[
        mlc.Field(
            name="filename",
            data_types=mlc.DataType.TEXT,
            source=mlc.Source(
                file_set="text-files",             # refer to the FileSet by name
                extract=mlc.Extract(file_property="filename"),
            ),
        ),
        mlc.Field(
            name="content",
            data_types=mlc.DataType.TEXT,
            source=mlc.Source(
                file_set="text-files",
                extract=mlc.Extract(file_property="content"),
            ),
        ),
    ],
)

# 4) Top-level Metadata (schema.org Dataset), then serialize to JSON-LD
metadata = mlc.Metadata(
    name="Test Text Dataset",
    description="A simple Croissant dataset containing local text files for testing.",
    license="https://creativecommons.org/licenses/by/4.0/",
    url="https://example.com/dataset/test-text",
    conforms_to="http://mlcommons.org/croissant/1.0",
    distribution=[*distribution, text_files],
    record_sets=[record_set],
)
metadata.date_published = datetime.now().strftime("%Y-%m-%d")

# 5) Save to JSON-LD file in the data store
run_id = 1
data_card_path = os.path.join(resource_path, f"data_card_{run_id}.json")
with open(data_card_path, 'w', encoding='utf-8') as json_file:
    content = metadata.to_json()
    json.dump(content, json_file, indent=4)
