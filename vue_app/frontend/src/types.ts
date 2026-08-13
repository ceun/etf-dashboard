export interface Target {
  name: string
  index_code: string
  etf_code: string
  scaling_factor: number
  data_source: 'SZ' | 'ZZ' | 'YH' | 'YHE'
  asset_currency: string
  report_currency: string
}

export interface AnalysisMetrics {
  latest_date: string
  trad_range_start: string
  trad_range_end: string
  latest_close: number
  latest_etf_price: number
  trad_pred: number
  roll_pred: number
  ma_pred: number
  dev_trad: number
  dev_roll: number
  dev_ma: number
  z_trad_latest: number
  cagr_trad: number
  cagr_roll: number
  std_trad: number
  cagr_95ci: string
}

export interface AnalysisPoint {
  Date: string
  Close: number
  Trad_Pred_Price: number
  Roll_Pred_Price: number | null
  MA_Price: number | null
  Trad_Z_Score: number
  Roll_Z_Score: number | null
  MA_Z_Score: number | null
}

export interface AnalysisResponse {
  target: Target
  metrics: AnalysisMetrics
  series: AnalysisPoint[]
}

export interface ComparisonRow {
  name: string
  etf_code: string
  latest_date: string
  trad_deviation_pct: number | null
  roll_deviation_pct: number | null
  trad_z_score: number | null
  trad_cagr_pct: number | null
  roll_cagr_pct: number | null
  sigma_pct: number | null
  cagr_95ci: string | null
  trad_cagr_range: string | null
  [key: string]: string | number | null
}
