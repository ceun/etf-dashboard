ALTER TABLE etf_targets
ADD COLUMN IF NOT EXISTS asset_currency TEXT;

ALTER TABLE etf_targets
ADD COLUMN IF NOT EXISTS report_currency TEXT;

UPDATE etf_targets
SET asset_currency = COALESCE(NULLIF(TRIM(asset_currency), ''), 'CNY')
WHERE asset_currency IS NULL OR TRIM(asset_currency) = '';

UPDATE etf_targets
SET report_currency = COALESCE(NULLIF(TRIM(report_currency), ''), 'CNY')
WHERE report_currency IS NULL OR TRIM(report_currency) = '';

ALTER TABLE etf_targets
ALTER COLUMN asset_currency SET DEFAULT 'CNY';

ALTER TABLE etf_targets
ALTER COLUMN report_currency SET DEFAULT 'CNY';

ALTER TABLE etf_prices
ADD COLUMN IF NOT EXISTS asset_close_native DOUBLE PRECISION;

ALTER TABLE etf_prices
ADD COLUMN IF NOT EXISTS fx_to_cny DOUBLE PRECISION;

ALTER TABLE etf_prices
ADD COLUMN IF NOT EXISTS close_cny DOUBLE PRECISION;

UPDATE etf_prices
SET asset_close_native = COALESCE(asset_close_native, index_close, combined_close)
WHERE asset_close_native IS NULL;

UPDATE etf_prices
SET fx_to_cny = COALESCE(fx_to_cny, 1.0)
WHERE fx_to_cny IS NULL;

UPDATE etf_prices
SET close_cny = COALESCE(close_cny, combined_close)
WHERE close_cny IS NULL;

CREATE TABLE IF NOT EXISTS fx_rates (
    date DATE NOT NULL,
    from_currency TEXT NOT NULL,
    to_currency TEXT NOT NULL,
    fx_rate DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (date, from_currency, to_currency)
);

CREATE INDEX IF NOT EXISTS idx_fx_rates_pair_date
ON fx_rates(from_currency, to_currency, date);
