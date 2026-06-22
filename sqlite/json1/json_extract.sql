SELECT artifact_hash,
    json_extract(metadata, '$.author') AS author,
    json_extract(metadata, '$.version') AS version,
    json_extract(artifact_location, '$.topic') AS topic,
    json_extract(artifact_location, '$.partition') AS partition,
    json_extract(artifact_location, '$.offset') AS offset
FROM artifacts;

SELECT artifact_hash,
    json_extract(metadata, '$.tags[0]') AS tag1,
    json_extract(metadata, '$.tags[1]') AS tag2,
    json_extract(metadata, '$.description') AS description
FROM artifacts
WHERE json_extract(metadata, '$.tags') IS NOT NULL;

SELECT *
FROM artifacts
WHERE json_extract(artifact_location, '$.topic') = 'events';

SELECT value
FROM artifacts,
    json_each(
        metadata,
        '$.tags'
    )
WHERE json_extract(metadata, '$.tags') IS NOT NULL;

SELECT *
FROM json_tree(
    '{"a":{"b":{"c":123}}}'
);

SELECT *
FROM artifacts
WHERE partition = 1;