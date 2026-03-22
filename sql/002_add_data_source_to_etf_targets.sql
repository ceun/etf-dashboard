ALTER TABLE etf_targets
ADD COLUMN IF NOT EXISTS data_source TEXT;

UPDATE etf_targets
SET data_source = CASE
    WHEN UPPER(COALESCE(index_code, '')) LIKE 'CN%'
      OR UPPER(COALESCE(index_code, '')) LIKE '399%'
      OR UPPER(COALESCE(index_code, '')) LIKE '48%'
    THEN 'SZ'
    ELSE 'ZZ'
END
WHERE data_source IS NULL;

ALTER TABLE etf_targets
ALTER COLUMN data_source SET DEFAULT 'ZZ';
