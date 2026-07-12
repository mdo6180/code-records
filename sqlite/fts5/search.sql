-- Search for a word
SELECT *
FROM documents
WHERE documents MATCH 'SQLite';

-- Search for a word in a specific column
SELECT *
FROM documents
WHERE documents MATCH 'title:SQLite';

-- Search for a phrase
SELECT *
FROM documents
WHERE documents MATCH '"lightweight embedded"';

-- Boolean search
SELECT *
FROM documents
WHERE documents MATCH 'SQLite AND lightweight';

SELECT *
FROM documents
WHERE documents MATCH 'SQLite OR FastAPI';

SELECT *
FROM documents
WHERE documents MATCH 'SQLite NOT embedded';

-- Search for a prefix
SELECT *
FROM documents
WHERE documents MATCH 'light*';

-- BM25 ranking
SELECT
    title, 
    bm25(documents) AS score
FROM documents
WHERE documents MATCH 'SQLite'
ORDER BY score;