CREATE TABLE artifacts (
    artifact_hash TEXT PRIMARY KEY,
    artifact_location TEXT
);

INSERT INTO artifacts VALUES
('hash1', '/high/model.pt');