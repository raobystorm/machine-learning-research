WITH
  families AS (
    -- select families with only one child
  WITH
    one_child_family AS (
    SELECT
      family_id,
      COUNT(id) AS cnt
    FROM
      production_embulk.child
    GROUP BY
      family_id
    HAVING
      COUNT(id) = 1),
    -- select families that has enough media
    media_count AS (
    SELECT
      family_id
    FROM
      production_embulk.media
    WHERE
      upload_content_type = "image/jpeg"
      AND thumbnail_generated = 1
      AND audience_type IN (2,
        3)
    GROUP BY
      family_id
    HAVING
      COUNT(id) > 1000 )
    -- join those families
  SELECT
    m.family_id
  FROM
    one_child_family AS o
  JOIN
    media_count AS m
  ON
    o.family_id = m.family_id
  ORDER BY
    m.family_id
  LIMIT
    15000 ),
  media AS (
    -- select media with family_id, uuid, took_at and row number
  SELECT
    family_id,
    uuid,
    took_at,
    upload_file_name,
    ROW_NUMBER() OVER (PARTITION BY family_id ORDER BY took_at DESC) AS media_number
  FROM
    production_embulk.media
  WHERE
    upload_content_type = "image/jpeg"
    AND thumbnail_generated = 1
    AND audience_type IN (2,
      3) )
-- select 600 valid images per family
SELECT
  families.family_id,
  media.uuid,
  upload_file_name
FROM
  families
JOIN
  media
ON
  families.family_id = media.family_id
WHERE
  media.media_number <= 600