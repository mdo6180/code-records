CREATE TABLE IF NOT EXISTS transfers (
    manifest_hash TEXT PRIMARY KEY,
    manifest_signature TEXT NOT NULL,
    manifest TEXT NOT NULL,
    transfer_id TEXT GENERATED ALWAYS AS (
        json_extract(manifest, '$.transfer_id')
    ) STORED,
    archive_size INTEGER GENERATED ALWAYS AS (
        json_extract(manifest, '$.archive_size')
    ) STORED,
    chunk_count INTEGER GENERATED ALWAYS AS (
        json_extract(manifest, '$.chunk_count')
    ) STORED,
    previous_transfer_id TEXT GENERATED ALWAYS AS (
        json_extract(manifest, '$.previous_transfer_id')
    ) STORED,
    previous_manifest_hash TEXT GENERATED ALWAYS AS (
        json_extract(manifest, '$.previous_manifest_hash')
    ) STORED,
    node_name TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(manifest_hash)
);

CREATE TABLE IF NOT EXISTS transfer_artifacts (
    artifact_hash TEXT NOT NULL,
    manifest_hash TEXT NOT NULL,
    filepath TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    source_pipeline_name TEXT NOT NULL,
    destination_pipeline_name TEXT NOT NULL,
    destination_stream TEXT NOT NULL,
    PRIMARY KEY (artifact_hash, manifest_hash),
    FOREIGN KEY (manifest_hash) REFERENCES transfers(manifest_hash)
);

CREATE TRIGGER insert_transfer_artifacts
AFTER INSERT ON transfers
BEGIN
    INSERT INTO transfer_artifacts (
        artifact_hash,
        manifest_hash,
        filepath,
        size_bytes,
        source_pipeline_name,
        destination_pipeline_name,
        destination_stream
    )
    SELECT
        json_extract(artifact.value, '$.artifact_hash'),
        NEW.manifest_hash,
        json_extract(artifact.value, '$.filepath'),
        json_extract(artifact.value, '$.size_bytes'),
        json_extract(artifact.value, '$.source_pipeline_name'),
        json_extract(artifact.value, '$.destination_pipeline_name'),
        json_extract(artifact.value, '$.destination_stream')
    FROM json_each(NEW.manifest, '$.transfer_artifacts') AS artifact;
END;

INSERT INTO transfers (manifest_hash, manifest_signature, manifest, node_name)
VALUES (
    'hash1',
    'signature1',
    '{
        "transfer_id": "transfer_6f25e1c6e37b4ce183c1a6ab6b0f8b1b",
        "transfer_artifacts": [
            {
                "artifact_hash": "0402c50b3f860c02ba6e9151c91a26acd67a2c6d1b2a6aea77a99b9984640a0d",
                "filepath": "10mb.txt",
                "size_bytes": 10485760,
                "source_pipeline_name": "example_src_pipeline",
                "destination_pipeline_name": "example_dest_pipeline",
                "destination_stream": "example_dest_stream"
            }
        ],
        "archive_size": 30797,
        "chunk_count": 4,
        "previous_transfer_id": null,
        "previous_manifest_hash": null
    }',
    'node1'
);

INSERT INTO transfers (manifest_hash, manifest_signature, manifest, node_name)
VALUES (
    'hash2',
    'signature2',
    '{
        "transfer_id": "transfer_7f25e1c6e37b4ce183c1a6ab6b0f8b2c",
        "transfer_artifacts": [
            {
                "artifact_hash": "1502c50b3f860c02ba6e9151c91a26acd67a2c6d1b2a6aea77a99b9984640a0e",
                "filepath": "20mb.txt",
                "size_bytes": 20971520,
                "source_pipeline_name": "example_src_pipeline",
                "destination_pipeline_name": "example_dest_pipeline",
                "destination_stream": "example_dest_stream"
            },
            {
                "artifact_hash": "2502c50b3f860c02ba6e9151c91a26acd67a2c6d1b2a6aea77a99b9984640a0f",
                "filepath": "30mb.txt",
                "size_bytes": 31457280,
                "source_pipeline_name": "example_src_pipeline",
                "destination_pipeline_name": "example_dest_pipeline",
                "destination_stream": "example_dest_stream"
            }
        ],
        "archive_size": 40960,
        "chunk_count": 8,
        "previous_transfer_id": "transfer_6f25e1c6e37b4ce183c1a6ab6b0f8b1b",
        "previous_manifest_hash": "hash1"
    }',
    'node2'
);