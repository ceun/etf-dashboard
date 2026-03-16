-- One-time DDL: cache table for ETF unadjusted factor
-- Run this once in your PostgreSQL/Supabase database.

CREATE TABLE IF NOT EXISTS etf_unadj_factor_cache (
    etf_code VARCHAR(16) PRIMARY KEY,
    unadj_factor DOUBLE PRECISION NOT NULL,
    factor_source VARCHAR(64) NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE etf_unadj_factor_cache IS 'Cache for unadjusted factor used by ETF display fallback chain';
COMMENT ON COLUMN etf_unadj_factor_cache.etf_code IS 'ETF code, e.g. 510300';
COMMENT ON COLUMN etf_unadj_factor_cache.unadj_factor IS 'Conversion factor where unadj ~= hfq * factor';
COMMENT ON COLUMN etf_unadj_factor_cache.factor_source IS 'Last successful source: baostock or akshare';
COMMENT ON COLUMN etf_unadj_factor_cache.updated_at IS 'Last cache update time';
