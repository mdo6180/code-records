CREATE TABLE IF NOT EXISTS artifacts (
    artifact_hash TEXT PRIMARY KEY,
    artifact_location TEXT
);

INSERT OR IGNORE INTO artifacts VALUES
('hash2', '/low/image.png'),
('hash3', '/low/report.pdf');

CREATE TABLE IF NOT EXISTS metadata (
    artifact_hash TEXT PRIMARY KEY,
    artifact_name TEXT,
    artifact_description TEXT
);

INSERT OR IGNORE INTO metadata VALUES
('hash2', 'Image', 'A sample image file'),
('hash3', 'Report', 'A sample report file');