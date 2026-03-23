ALTER TABLE etf_targets
ADD COLUMN IF NOT EXISTS index_code TEXT;

ALTER TABLE etf_targets
ADD COLUMN IF NOT EXISTS etf_code TEXT;

ALTER TABLE etf_prices
ADD COLUMN IF NOT EXISTS index_code TEXT;

UPDATE etf_targets
SET index_code = UPPER(TRIM(index_code))
WHERE index_code IS NOT NULL;

UPDATE etf_targets
SET etf_code = NULLIF(TRIM(etf_code), '')
WHERE etf_code IS NOT NULL;

UPDATE etf_prices p
SET index_code = t.index_code
FROM etf_targets t
WHERE p.index_code IS NULL
  AND p.etf_code = t.etf_code;

ALTER TABLE etf_targets
ALTER COLUMN index_code SET NOT NULL;

ALTER TABLE etf_prices
ALTER COLUMN index_code SET NOT NULL;

ALTER TABLE etf_targets
DROP CONSTRAINT IF EXISTS etf_targets_pkey;

ALTER TABLE etf_targets
ADD CONSTRAINT etf_targets_pkey PRIMARY KEY (index_code);

ALTER TABLE etf_prices
DROP CONSTRAINT IF EXISTS etf_prices_pkey;

ALTER TABLE etf_prices
ADD CONSTRAINT etf_prices_pkey PRIMARY KEY (index_code, date);

CREATE INDEX IF NOT EXISTS idx_etf_prices_index_code ON etf_prices(index_code);
CREATE UNIQUE INDEX IF NOT EXISTS idx_etf_targets_name ON etf_targets(name);
CREATE UNIQUE INDEX IF NOT EXISTS idx_etf_targets_etf_code_not_null
ON etf_targets(etf_code)
WHERE etf_code IS NOT NULL;
