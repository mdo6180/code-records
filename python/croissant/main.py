import mlcroissant as mlc
import json
from datetime import datetime



# FileObjects and FileSets define the resources of the dataset.
distribution = [
    # gpt-3 is hosted on a GitHub repository:
    mlc.FileObject(
        id="github-repository",
        name="github-repository",
        description="OpenAI repository on GitHub.",
        content_url="https://github.com/openai/gpt-3",
        encoding_formats=["git+https"],
        sha256="main",
    ),
    # Within that repository, a FileSet lists all JSONL files:
    mlc.FileSet(
        id="jsonl-files",
        name="jsonl-files",
        description="JSONL files are hosted on the GitHub repository.",
        contained_in=["github-repository"],
        encoding_formats=["application/jsonlines"],
        includes="data/*.jsonl",
    ),
]
record_sets = [
    # RecordSets contains records in the dataset.
    mlc.RecordSet(
        id="jsonl",
        name="jsonl",
        # Each record has one or many fields...
        fields=[
            # Fields can be extracted from the FileObjects/FileSets.
            mlc.Field(
                id="jsonl/context",
                name="context",
                description="",
                data_types=mlc.DataType.TEXT,
                source=mlc.Source(
                    file_set="jsonl-files",
                    # Extract the field from the column of a FileObject/FileSet:
                    extract=mlc.Extract(column="context"),
                ),
            ),
            mlc.Field(
                id="jsonl/completion",
                name="completion",
                description="The expected completion of the promt.",
                data_types=mlc.DataType.TEXT,
                source=mlc.Source(
                    file_set="jsonl-files",
                    extract=mlc.Extract(column="completion"),
                ),
            ),
            mlc.Field(
                id="jsonl/task",
                name="task",
                description=(
                    "The machine learning task appearing as the name of the"
                    " file."
                ),
                data_types=mlc.DataType.TEXT,
                source=mlc.Source(
                    file_set="jsonl-files",
                    extract=mlc.Extract(
                        file_property=mlc._src.structure_graph.nodes.source.FileProperty.filename
                    ),
                    # Extract the field from a regex on the filename:
                    transforms=[mlc.Transform(regex="^(.*)\.jsonl$")],
                ),
            ),
        ],
    )
]

# Metadata contains information about the dataset.
metadata = mlc.Metadata(
    name="gpt-3",
    # Descriptions can contain plain text or markdown.
    url="https://github.com/openai/gpt-3",
    distribution=distribution,
    record_sets=record_sets,
    license="MIT",
    version="1.0.0"
)

metadata.date_published = datetime.now().strftime("%Y-%m-%d")

# Save the metadata to a JSON file.
with open("croissant.json", "w") as f:
    content = metadata.to_json()
    print(content)
    content = json.dumps(content, indent=2)
    #print(content)
    f.write(content)