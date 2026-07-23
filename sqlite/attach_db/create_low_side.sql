CREATE TABLE artifacts (
    artifact_hash TEXT PRIMARY KEY,
    artifact_location TEXT
);

INSERT INTO artifacts VALUES
('hash2', '/low/image.png'),
('hash3', '/low/report.pdf');