-- One-time migration: split ETF close into raw and HFQ columns
-- Run this script once on your PostgreSQL/Supabase database.

ALTER TABLE etf_prices
ADD COLUMN IF NOT EXISTS etf_close_raw DOUBLE PRECISION;

ALTER TABLE etf_prices
ADD COLUMN IF NOT EXISTS etf_close_hfq DOUBLE PRECISION;

-- Backfill HFQ column from legacy etf_close (historical legacy data was HFQ in this project).
UPDATE etf_prices
SET etf_close_hfq = etf_close
WHERE etf_close_hfq IS NULL
  AND etf_close IS NOT NULL;

-- Optional compatibility backfill: if you need raw column initially populated before full refresh,
-- uncomment the following block. Note this may temporarily copy HFQ values into raw until refreshed.
-- UPDATE etf_prices
-- SET etf_close_raw = etf_close
-- WHERE etf_close_raw IS NULL
--   AND etf_close IS NOT NULL;
