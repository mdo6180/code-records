CREATE TABLE IF NOT EXISTS artifacts (
    artifact_hash TEXT PRIMARY KEY,
    artifact_location TEXT
);

INSERT OR IGNORE INTO artifacts VALUES
('hash1', '/high/model.pt');