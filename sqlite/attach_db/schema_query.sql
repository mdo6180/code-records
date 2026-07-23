-- Attach the low-side database
ATTACH DATABASE 'low_side.db' AS low;

BEGIN;

-- display the schema of the low-side database
SELECT * FROM low.sqlite_schema;

-- filter out internal SQLite tables and ensuring that only tables with valid SQL definitions are shown
SELECT name, sql
FROM low.sqlite_schema
WHERE type = 'table'
    AND name NOT LIKE 'sqlite_%'
    AND sql IS NOT NULL
ORDER BY name;

COMMIT;

DETACH DATABASE low;