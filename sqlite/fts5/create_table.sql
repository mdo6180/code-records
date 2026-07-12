CREATE VIRTUAL TABLE IF NOT EXISTS documents
USING fts5(
    title,
    body
);

INSERT INTO documents(title, body) VALUES
('SQLite', 'SQLite is a lightweight embedded database.'),
('FastAPI', 'FastAPI is an asynchronous Python web framework.'),
('Anacostia', 'Anacostia is a lightweight MLOps framework built on SQLite.');