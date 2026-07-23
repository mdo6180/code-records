-- Attach the low-side database
ATTACH DATABASE 'low_side.db' AS low;

BEGIN;

-- Copy rows from low_side.db into high_side.db
INSERT OR IGNORE INTO main.artifacts
SELECT *
FROM low.artifacts;

COMMIT;

DETACH DATABASE low;