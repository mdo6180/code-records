CREATE TABLE IF NOT EXISTS artifacts (
    artifact_hash TEXT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    artifact_location TEXT NOT NULL,
    metadata TEXT,
    topic TEXT GENERATED ALWAYS AS (
        json_extract(
            artifact_location,
            '$.topic'
        )
    ) STORED,
    partition INTEGER GENERATED ALWAYS AS (
        json_extract(
            artifact_location,
            '$.partition'
        )
    ) STORED,
    offset INTEGER GENERATED ALWAYS AS (
        json_extract(
            artifact_location,
            '$.offset'
        )
    ) STORED,
    UNIQUE(artifact_hash)
);

INSERT INTO artifacts (artifact_hash, artifact_location, metadata)
VALUES ('hash1', '{"topic": "events", "partition": 1, "offset": 100}', '{"author": "John Doe", "version": "1.0"}'),
       ('hash2', '{"topic": "events", "partition": 2, "offset": 200}', '{"author": "Jane Smith", "version": "2.0"}'),
       ('hash3', '{"topic": "events", "partition": 3, "offset": 300}', '{"author": "Alice Johnson", "version": "3.0"}');

INSERT INTO artifacts (artifact_hash, artifact_location, metadata)
VALUES ('hash4', '{"topic": "events", "partition": 4, "offset": 400}', '{"tags": ["tag1", "tag2"], "description": "This is a sample artifact."}'),
       ('hash5', '{"topic": "events", "partition": 5, "offset": 500}', '{"tags": ["tag3", "tag4"], "description": "Another sample artifact."}');

INSERT INTO artifacts (artifact_hash, artifact_location, metadata)
VALUES ('hash6', '{"topic": "logs", "partition": 1, "offset": 600}', '{
    "a": {
        "b": {
            "c": 123
        }
    }
}');