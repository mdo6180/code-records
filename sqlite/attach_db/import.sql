-- Attach the low-side database
ATTACH DATABASE 'low_side.db' AS low;

BEGIN;

-- Copy rows from low_side.db into high_side.db
INSERT OR IGNORE INTO main.artifacts
SELECT *
FROM low.artifacts;

-- Create the metadata table in the high-side database and copy rows from low_side.db into it
CREATE TABLE IF NOT EXISTS main.metadata (
    artifact_hash TEXT PRIMARY KEY,
    artifact_name TEXT,
    artifact_description TEXT
);

INSERT OR IGNORE INTO main.metadata
SELECT *
FROM low.metadata;

COMMIT;

DETACH DATABASE low;