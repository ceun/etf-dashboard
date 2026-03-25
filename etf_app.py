import os
import socket
import json
import psycopg2
from psycopg2.extras import execute_values
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tickflow.client import TickFlow
import yfinance as yf
import streamlit as st
from urllib.parse import urlparse, parse_qs, unquote, quote
from urllib.request import Request, urlopen

warnings.filterwarnings('ignore')

# ─── 页面配置 ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="今天买什么", page_icon="📈", layout="wide")

# ─── 全局配置 ────────────────────────────────────────────────────────────────
DEFAULT_TRADITION_START = pd.to_datetime("20081031", format="%Y%m%d").date()
DEFAULT_ROLLING_WINDOW = 1250
# Supabase 连接字符串（从 Streamlit secrets 读取）
DATABASE_URL_POOLER = st.secrets.get("database_url_pooler", None)
DATABASE_URL_DIRECT = st.secrets.get("database_url", None)
DATABASE_URL = DATABASE_URL_POOLER or DATABASE_URL_DIRECT
TICKFLOW_API_KEY = st.secrets.get("tickflow_api_key", os.getenv("TICKFLOW_API_KEY"))

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ─── 数据库连接 ───────────────────────────────────────────────────────────────
def get_db_connection():
    """获取数据库连接（每次新建，避免缓存已关闭连接）"""
    if not DATABASE_URL:
        return None
    if not DATABASE_URL_POOLER:
        st.warning("建议在 Streamlit secrets 配置 database_url_pooler（Supabase Connection Pooling URI），可避免 IPv6 直连问题。")

    # 先按原始 DSN 连接；若网络优先解析到 IPv6 且本机无 IPv6 出口，再回退到 IPv4。
    try:
        return psycopg2.connect(DATABASE_URL, connect_timeout=10)
    except Exception as e:
        err_msg = str(e)
        ipv6_issue = "Cannot assign requested address" in err_msg or "Network is unreachable" in err_msg
        if not ipv6_issue:
            st.error(f"数据库连接失败: {e}")
            return None

        try:
            parsed = urlparse(DATABASE_URL)
            host = parsed.hostname
            if not host:
                st.error(f"数据库连接失败: {e}")
                return None

            ipv4_list = socket.getaddrinfo(host, None, socket.AF_INET)
            if not ipv4_list:
                st.error("数据库连接失败：当前直连地址无 IPv4 解析结果。请改用 Supabase pooler 连接串（database_url_pooler）。")
                return None
            ipv4_addr = ipv4_list[0][4][0]

            dbname = parsed.path.lstrip("/")
            user = unquote(parsed.username or "")
            password = unquote(parsed.password or "")
            port = parsed.port or 5432
            query_params = parse_qs(parsed.query)
            sslmode = query_params.get("sslmode", ["require"])[0]

            # 某些环境下 hostaddr 参数会触发地址格式校验报错，直接用 IPv4 作为 host 更稳。
            return psycopg2.connect(
                dbname=dbname,
                user=user,
                password=password,
                host=ipv4_addr,
                port=port,
                sslmode=sslmode,
                connect_timeout=10,
            )
        except Exception as fallback_e:
            st.error(f"数据库连接失败（IPv4回退后仍失败）: {fallback_e}")
            return None


# ─── 标的元数据 ───────────────────────────────────────────────────────────────
def load_targets_from_db():
    """从 etf_targets 读取所有标的，返回 {name: {etf_code, name, index_code, scaling_factor, data_source, asset_currency, report_currency}}"""
    conn = get_db_connection()
    if not conn:
        return {}
    try:
        df = pd.read_sql(
            "SELECT index_code, etf_code, name, scaling_factor, data_source, asset_currency, report_currency FROM etf_targets ORDER BY name",
            conn,
        )
        if df.empty:
            return {}
        return {
            row['name']: {
                "name": row['name'],
                "etf_code": _normalize_etf_code(row['etf_code']) if pd.notna(row['etf_code']) else "",
                "index_code": _normalize_index_code(row['index_code']) if pd.notna(row['index_code']) else "",
                "scaling_factor": float(row['scaling_factor']) if pd.notna(row['scaling_factor']) else 1.0,
                "data_source": _normalize_data_source(row['data_source']) if pd.notna(row['data_source']) else _infer_data_source_from_index_code(row['index_code']),
                "asset_currency": _normalize_currency(row['asset_currency']) if pd.notna(row['asset_currency']) else "CNY",
                "report_currency": _normalize_currency(row['report_currency']) if pd.notna(row['report_currency']) else "CNY",
            }
            for _, row in df.iterrows()
        }
    except Exception as e:
        st.caption(f"⚠️ 读取标的配置失败: {e}")
        return {}
    finally:
        conn.close()


def save_target_to_db(index_code, name, etf_code=None, scaling_factor=None, stitch_date=None, data_source=None, asset_currency=None, report_currency=None):
    """新增或更新 etf_targets 标的元数据"""
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        normalized_index_code = _normalize_index_code(index_code)
        normalized_etf_code = _normalize_etf_code(etf_code) or None
        normalized_source = _normalize_data_source(data_source) if data_source else _infer_data_source_from_index_code(index_code)
        normalized_asset_currency = _normalize_currency(asset_currency) or "CNY"
        normalized_report_currency = _normalize_currency(report_currency) or "CNY"
        cur.execute("""
            INSERT INTO etf_targets (index_code, etf_code, name, scaling_factor, stitch_date, data_source, asset_currency, report_currency)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (index_code) DO UPDATE SET
                etf_code       = COALESCE(EXCLUDED.etf_code,       etf_targets.etf_code),
                name           = EXCLUDED.name,
                scaling_factor = COALESCE(EXCLUDED.scaling_factor, etf_targets.scaling_factor),
                stitch_date    = COALESCE(EXCLUDED.stitch_date,    etf_targets.stitch_date),
                data_source    = COALESCE(EXCLUDED.data_source,    etf_targets.data_source),
                asset_currency = COALESCE(EXCLUDED.asset_currency, etf_targets.asset_currency),
                report_currency = COALESCE(EXCLUDED.report_currency, etf_targets.report_currency)
        """, (normalized_index_code, normalized_etf_code, name, scaling_factor, stitch_date, normalized_source, normalized_asset_currency, normalized_report_currency))
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        st.warning(f"保存标的元数据失败: {e}")
        return False
    finally:
        conn.close()


def save_prices_to_db(df, index_code):
    """
    写入 etf_prices。df 需含 Date、combined_close，可含 index_close、etf_close_raw、etf_close_hfq、
    asset_close_native、fx_to_cny、close_cny。
    冲突时：价格列按新值优先回填，combined_close 直接覆盖。
    """
    conn = get_db_connection()
    if not conn:
        st.warning("无法连接数据库，数据未保存")
        return 0
    try:
        cur = conn.cursor()

        # 统一列，避免逐行 SQL 循环造成长时间阻塞
        work = df.copy()
        if 'index_close' not in work.columns:
            work['index_close'] = None
        if 'etf_close_raw' not in work.columns:
            work['etf_close_raw'] = None
        if 'etf_close_hfq' not in work.columns:
            work['etf_close_hfq'] = None
        if 'asset_close_native' not in work.columns:
            work['asset_close_native'] = None
        if 'fx_to_cny' not in work.columns:
            work['fx_to_cny'] = None
        if 'close_cny' not in work.columns:
            work['close_cny'] = None

        work['Date'] = pd.to_datetime(work['Date'])
        work['index_close'] = pd.to_numeric(work['index_close'], errors='coerce')
        work['etf_close_raw'] = pd.to_numeric(work['etf_close_raw'], errors='coerce')
        work['etf_close_hfq'] = pd.to_numeric(work['etf_close_hfq'], errors='coerce')
        work['asset_close_native'] = pd.to_numeric(work['asset_close_native'], errors='coerce')
        work['fx_to_cny'] = pd.to_numeric(work['fx_to_cny'], errors='coerce')
        work['close_cny'] = pd.to_numeric(work['close_cny'], errors='coerce')
        work['combined_close'] = pd.to_numeric(work['combined_close'], errors='coerce')

        rows = []
        for _, row in work[['Date', 'index_close', 'etf_close_raw', 'etf_close_hfq', 'asset_close_native', 'fx_to_cny', 'close_cny', 'combined_close']].iterrows():
            rows.append((
                _normalize_index_code(index_code),
                row['Date'].date(),
                None if pd.isna(row['index_close']) else float(row['index_close']),
                None if pd.isna(row['etf_close_raw']) else float(row['etf_close_raw']),
                None if pd.isna(row['etf_close_hfq']) else float(row['etf_close_hfq']),
                None if pd.isna(row['asset_close_native']) else float(row['asset_close_native']),
                None if pd.isna(row['fx_to_cny']) else float(row['fx_to_cny']),
                None if pd.isna(row['close_cny']) else float(row['close_cny']),
                float(row['combined_close']),
            ))

        if not rows:
            cur.close()
            return 0

        sql = """
            INSERT INTO etf_prices (index_code, date, index_close, etf_close_raw, etf_close_hfq, asset_close_native, fx_to_cny, close_cny, combined_close)
            VALUES %s
            ON CONFLICT (index_code, date) DO UPDATE SET
                index_close    = COALESCE(EXCLUDED.index_close,    etf_prices.index_close),
                etf_close_raw  = COALESCE(EXCLUDED.etf_close_raw,  etf_prices.etf_close_raw),
                etf_close_hfq  = COALESCE(EXCLUDED.etf_close_hfq,  etf_prices.etf_close_hfq),
                asset_close_native = COALESCE(EXCLUDED.asset_close_native, etf_prices.asset_close_native),
                fx_to_cny      = COALESCE(EXCLUDED.fx_to_cny,      etf_prices.fx_to_cny),
                close_cny      = COALESCE(EXCLUDED.close_cny,      etf_prices.close_cny),
                combined_close = EXCLUDED.combined_close
        """

        # 分批写入，兼顾性能和语句体积
        batch_size = 1000
        for i in range(0, len(rows), batch_size):
            execute_values(cur, sql, rows[i:i + batch_size])

        conn.commit()
        cur.close()
        return int(len(rows))
    except Exception as e:
        st.warning(f"保存行情数据异常: {e}")
        return 0
    finally:
        conn.close()


def load_from_db(index_code):
    """返回 (df, scaling_factor)：df含 Date、Close(combined_close)、ETF_Close_Raw、ETF_Close_HFQ、Asset_Close_Native、FX_To_CNY、Close_CNY。"""
    conn = get_db_connection()
    if not conn:
        return None, 1.0
    try:
        df = pd.read_sql(
            """
            SELECT
                date,
                combined_close,
                etf_close_raw,
                etf_close_hfq,
                asset_close_native,
                fx_to_cny,
                close_cny
            FROM etf_prices
            WHERE index_code=%s
            ORDER BY date
            """,
            conn, params=(_normalize_index_code(index_code),),
        )
        sf_row = pd.read_sql(
            "SELECT scaling_factor FROM etf_targets WHERE index_code=%s",
            conn, params=(_normalize_index_code(index_code),),
        )
        scaling_factor = 1.0
        if not sf_row.empty and pd.notna(sf_row['scaling_factor'].iloc[0]):
            scaling_factor = float(sf_row['scaling_factor'].iloc[0])
        if df.empty:
            return None, scaling_factor
        df = df.rename(columns={
            'date': 'Date',
            'combined_close': 'Close',
            'etf_close_raw': 'ETF_Close_Raw',
            'etf_close_hfq': 'ETF_Close_HFQ',
            'asset_close_native': 'Asset_Close_Native',
            'fx_to_cny': 'FX_To_CNY',
            'close_cny': 'Close_CNY',
        })
        df['Date']  = pd.to_datetime(df['Date'])
        df['Close'] = df['Close'].astype(float)
        df['ETF_Close_Raw'] = pd.to_numeric(df['ETF_Close_Raw'], errors='coerce')
        df['ETF_Close_HFQ'] = pd.to_numeric(df['ETF_Close_HFQ'], errors='coerce')
        df['Asset_Close_Native'] = pd.to_numeric(df['Asset_Close_Native'], errors='coerce')
        df['FX_To_CNY'] = pd.to_numeric(df['FX_To_CNY'], errors='coerce')
        df['Close_CNY'] = pd.to_numeric(df['Close_CNY'], errors='coerce')
        return df[['Date', 'Close', 'ETF_Close_Raw', 'ETF_Close_HFQ', 'Asset_Close_Native', 'FX_To_CNY', 'Close_CNY']].reset_index(drop=True), scaling_factor
    except Exception as e:
        st.warning(f"读取数据异常: {e}")
        return None, 1.0
    finally:
        conn.close()


# ─── TickFlow ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_tickflow_client():
    if TICKFLOW_API_KEY:
        return TickFlow(api_key=TICKFLOW_API_KEY)
    return TickFlow.free()


def _to_tickflow_etf_symbol(etf_code):
    code = str(etf_code).strip()
    if code.startswith(("5", "6")):
        return f"{code}.SH"
    return f"{code}.SZ"


def _extract_tickflow_date_close(raw_df):
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=["Date", "Close"])

    date_col = None
    if "date" in raw_df.columns:
        date_col = "date"
    elif "time" in raw_df.columns:
        date_col = "time"
    elif "trade_date" in raw_df.columns:
        date_col = "trade_date"
    elif "timestamp" in raw_df.columns:
        date_col = "timestamp"

    close_col = "close" if "close" in raw_df.columns else None
    if date_col is None or close_col is None:
        raise ValueError(f"TickFlow 返回字段不含可识别日期列或 close 列: {list(raw_df.columns)}")

    out = raw_df[[date_col, close_col]].copy()
    out.columns = ["Date", "Close"]
    if date_col == "timestamp":
        # TickFlow 日线 timestamp 为毫秒时间戳（UTC基准），转换到北京时间交易日
        out["Date"] = (
            pd.to_datetime(out["Date"], unit="ms", utc=True, errors="coerce")
            .dt.tz_convert("Asia/Shanghai")
            .dt.tz_localize(None)
            .dt.normalize()
        )
    else:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    return out


def fetch_all_from_tickflow(etf_code):
    """拉取 ETF 日线，返回不复权与后复权两套价格。"""
    client = get_tickflow_client()
    symbol = _to_tickflow_etf_symbol(etf_code)

    raw_df = client.klines.get(
        symbol=symbol,
        period="1d",
        count=100000,
        adjust="none",
        as_dataframe=True,
    )
    hfq_df = client.klines.get(
        symbol=symbol,
        period="1d",
        count=100000,
        adjust="backward",
        as_dataframe=True,
    )

    raw_out = _extract_tickflow_date_close(raw_df).rename(columns={"Close": "ETF_Close_Raw"})
    hfq_out = _extract_tickflow_date_close(hfq_df).rename(columns={"Close": "ETF_Close_HFQ"})
    out = pd.merge(raw_out, hfq_out, on="Date", how="outer").sort_values("Date").reset_index(drop=True)
    return out


def fetch_recent_from_tickflow(etf_code, count=30):
    """拉取 ETF 近期日线（不复权+后复权）。"""
    df = fetch_all_from_tickflow(etf_code)
    return df.tail(int(count)).reset_index(drop=True)


def _normalize_index_code(index_code):
    if index_code is None:
        return ""
    return str(index_code).strip().upper()


def _normalize_etf_code(etf_code):
    if etf_code is None:
        return ""
    return str(etf_code).strip()


def _normalize_currency(currency):
    if currency is None:
        return ""
    code = str(currency).strip().upper()
    return code if code else ""


def _normalize_data_source(data_source):
    if data_source is None:
        return ""
    code = str(data_source).strip().upper()
    if code in {"SZ", "ZZ", "YH"}:
        return code
    return ""


def _infer_data_source_from_index_code(index_code):
    code = _normalize_index_code(index_code)
    if _is_szse_index_code(code):
        return "SZ"
    return "ZZ"


def _is_szse_index_code(index_code):
    code = _normalize_index_code(index_code)
    return code.startswith(("CN", "399", "48"))


def _http_get_json(url, timeout=12):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", errors="ignore")
    return json.loads(text)


def fetch_szse_index_daily(index_code, start_date="1991-01-01", end_date="2050-01-01"):
    """从深证/国证公开接口拉取指数日线（Date, Index_Close）。"""
    code = _normalize_index_code(index_code)
    if not code:
        return pd.DataFrame(columns=["Date", "Index_Close"])

    url = (
        "https://hq.cnindex.com.cn/market/market/getIndexDailyData"
        f"?indexCode={quote(code)}&startDate={quote(str(start_date))}&endDate={quote(str(end_date))}"
    )
    payload = _http_get_json(url)
    if payload.get("code") != 200:
        raise RuntimeError(f"深证指数接口失败: {payload.get('message')}")

    data_root = payload.get("data") or {}
    rows = data_root.get("data") or []
    if not rows:
        return pd.DataFrame(columns=["Date", "Index_Close"])

    parsed = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 6:
            continue
        ts = pd.to_numeric(row[0], errors="coerce")
        close = pd.to_numeric(row[5], errors="coerce")
        if pd.isna(ts) or pd.isna(close):
            continue
        parsed.append((pd.to_datetime(int(ts), unit="ms"), float(close)))

    if not parsed:
        return pd.DataFrame(columns=["Date", "Index_Close"])

    out = pd.DataFrame(parsed, columns=["Date", "Index_Close"]).dropna()
    return out.sort_values("Date").reset_index(drop=True)


def fetch_yahoo_history(symbol, start_date="1991-01-01", end_date=None):
    """从 Yahoo Finance 拉取历史日线（Date, Index_Close）。"""
    code = str(symbol or "").strip()
    if not code:
        return pd.DataFrame(columns=["Date", "Index_Close"])

    hist = yf.download(
        code,
        start=str(start_date),
        end=str(end_date) if end_date else None,
        auto_adjust=False,
        progress=False,
        actions=False,
    )
    if hist is None or hist.empty:
        return pd.DataFrame(columns=["Date", "Index_Close"])

    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = [col[0] for col in hist.columns]

    hist = hist.reset_index()
    if "Date" not in hist.columns or "Close" not in hist.columns:
        raise ValueError(f"Yahoo Finance 返回字段不含 Date/Close: {list(hist.columns)}")

    out = hist[["Date", "Close"]].copy()
    out = out.rename(columns={"Close": "Index_Close"})
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
    out["Index_Close"] = pd.to_numeric(out["Index_Close"], errors="coerce")
    out = out.dropna(subset=["Date", "Index_Close"]).sort_values("Date").reset_index(drop=True)
    return out


def fetch_fx_history(from_currency, to_currency="CNY", start_date="1991-01-01", end_date=None):
    """从 Yahoo Finance 拉取汇率历史（Date, FX_Close），返回 from_currency/to_currency。"""
    base = _normalize_currency(from_currency)
    quote = _normalize_currency(to_currency)
    if not base or not quote:
        return pd.DataFrame(columns=["Date", "FX_Close"])
    if base == quote:
        date_index = pd.date_range(start=pd.to_datetime(start_date), end=pd.Timestamp.today() if end_date is None else pd.to_datetime(end_date), freq="D")
        out = pd.DataFrame({"Date": date_index.normalize(), "FX_Close": 1.0})
        return out

    symbol = f"{base}{quote}=X"
    raw = yf.download(
        symbol,
        start=str(start_date),
        end=str(end_date) if end_date else None,
        auto_adjust=False,
        progress=False,
        actions=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["Date", "FX_Close"])
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]
    raw = raw.reset_index()
    if "Date" not in raw.columns or "Close" not in raw.columns:
        raise ValueError(f"Yahoo Finance 汇率返回字段不含 Date/Close: {list(raw.columns)}")
    out = raw[["Date", "Close"]].rename(columns={"Close": "FX_Close"})
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
    out["FX_Close"] = pd.to_numeric(out["FX_Close"], errors="coerce")
    return out.dropna(subset=["Date", "FX_Close"]).sort_values("Date").reset_index(drop=True)


def save_fx_rates_to_db(df, from_currency, to_currency="CNY"):
    conn = get_db_connection()
    if not conn:
        return 0
    try:
        cur = conn.cursor()
        work = df.copy()
        work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
        work["FX_Close"] = pd.to_numeric(work["FX_Close"], errors="coerce")
        work = work.dropna(subset=["Date", "FX_Close"]).sort_values("Date")
        rows = []
        for _, row in work[["Date", "FX_Close"]].iterrows():
            rows.append((
                row["Date"].date(),
                _normalize_currency(from_currency),
                _normalize_currency(to_currency),
                float(row["FX_Close"]),
            ))
        if not rows:
            cur.close()
            return 0
        sql = """
            INSERT INTO fx_rates (date, from_currency, to_currency, fx_rate)
            VALUES %s
            ON CONFLICT (date, from_currency, to_currency) DO UPDATE SET
                fx_rate = EXCLUDED.fx_rate
        """
        execute_values(cur, sql, rows)
        conn.commit()
        cur.close()
        return int(len(rows))
    except Exception as e:
        st.warning(f"保存汇率数据异常: {e}")
        return 0
    finally:
        conn.close()


def _apply_currency_conversion(price_df, asset_currency, report_currency="CNY"):
    """为价格序列补充原币价格、汇率和人民币价格；当前支持报告币种为 CNY。"""
    work = price_df.copy()
    asset_ccy = _normalize_currency(asset_currency) or "CNY"
    report_ccy = _normalize_currency(report_currency) or "CNY"
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce").dt.normalize()
    work["asset_close_native"] = pd.to_numeric(work["Index_Close"], errors="coerce")

    if report_ccy != "CNY":
        work["fx_to_cny"] = np.nan
        work["close_cny"] = work["asset_close_native"]
        work["combined_close"] = work["asset_close_native"]
        return work

    fx_df = fetch_fx_history(asset_ccy, "CNY", start_date=work["Date"].min().date() if not work.empty else "1991-01-01")
    if not fx_df.empty:
        save_fx_rates_to_db(fx_df, asset_ccy, "CNY")
        work = pd.merge(work, fx_df, on="Date", how="left")
        work["FX_Close"] = work["FX_Close"].ffill().bfill()
    else:
        work["FX_Close"] = 1.0 if asset_ccy == "CNY" else np.nan

    if asset_ccy == "CNY":
        work["FX_Close"] = 1.0

    work = work.rename(columns={"FX_Close": "fx_to_cny"})
    work["close_cny"] = work["asset_close_native"] * work["fx_to_cny"]
    work["combined_close"] = work["close_cny"]
    return work


def _resolve_etf_stitch_series(df, raw_col="ETF_Close_Raw", hfq_col="ETF_Close_HFQ", out_col="ETF_Price_For_Stitch"):
    work = df.copy()
    raw_series = pd.to_numeric(work[raw_col], errors="coerce") if raw_col in work.columns else pd.Series(np.nan, index=work.index)
    hfq_series = pd.to_numeric(work[hfq_col], errors="coerce") if hfq_col in work.columns else pd.Series(np.nan, index=work.index)
    work[out_col] = hfq_series.where(hfq_series.notna(), raw_series)
    return work


def _estimate_scaling_factor_from_overlap(index_df, etf_code, default_sf=1.0):
    """用最近重叠窗口的中位数比值估算缩放比例，提升 ETF 价格换算稳定性。"""
    try:
        etf_code = _normalize_etf_code(etf_code)
        if not etf_code:
            return float(default_sf)
        etf_recent = fetch_recent_from_tickflow(etf_code, count=400)
        if etf_recent.empty:
            return float(default_sf)
        etf_recent = _resolve_etf_stitch_series(etf_recent)
        merged = pd.merge(
            index_df[["Date", "Index_Close"]],
            etf_recent[["Date", "ETF_Price_For_Stitch"]],
            on="Date",
            how="inner",
        )
        if merged.empty:
            return float(default_sf)
        merged = merged[(merged["Index_Close"] > 0) & (merged["ETF_Price_For_Stitch"] > 0)].copy()
        if merged.empty:
            return float(default_sf)
        merged["ratio"] = merged["Index_Close"] / merged["ETF_Price_For_Stitch"]
        ratio = pd.to_numeric(merged["ratio"], errors="coerce").dropna()
        if ratio.empty:
            return float(default_sf)
        # 采用最后一天重叠的精确比值，彻底消除拼接日的“跳空缺口”
        sf = float(ratio.iloc[-1])
        return sf if sf > 0 else float(default_sf)
    except Exception:
        return float(default_sf)


def _estimate_scaling_from_merged(merged_df, index_col, etf_col, default_sf=1.0, window=250):
    """从重叠样本估算缩放比例：使用最近窗口 ratio 的中位数（多点拟合）。"""
    if merged_df is None or merged_df.empty:
        return float(default_sf), None, 0
    work = merged_df.copy()
    work = work[(pd.to_numeric(work[index_col], errors="coerce") > 0) & (pd.to_numeric(work[etf_col], errors="coerce") > 0)].copy()
    if work.empty:
        return float(default_sf), None, 0
    work["ratio"] = pd.to_numeric(work[index_col], errors="coerce") / pd.to_numeric(work[etf_col], errors="coerce")
    work = work.dropna(subset=["ratio"]).sort_values("Date")
    if work.empty:
        return float(default_sf), None, 0

    sample = work.tail(int(window)).copy()
    ratio = pd.to_numeric(sample["ratio"], errors="coerce").dropna()
    if ratio.empty:
        return float(default_sf), None, 0

    # 采用最后一天重叠的精确比值，保证后续涨跌幅与ETF完全一致
    sf = float(ratio.iloc[-1])
    if sf <= 0:
        return float(default_sf), None, 0
    stitch_date = sample["Date"].max().date() if "Date" in sample.columns else None
    return sf, stitch_date, int(len(sample))


def _cn_today_date():
    return pd.Timestamp.now(tz="Asia/Shanghai").date()


def _exclude_today_rows(df, date_col="Date"):
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    today = _cn_today_date()
    return work[work[date_col].dt.date < today].copy()


def _keep_last_n_trading_days(df, n=3, date_col="Date"):
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    trade_days = sorted(work[date_col].dt.date.dropna().unique())
    if len(trade_days) <= n:
        return work.sort_values(date_col).reset_index(drop=True)
    keep_days = set(trade_days[-n:])
    return work[work[date_col].dt.date.isin(keep_days)].sort_values(date_col).reset_index(drop=True)


def _delete_today_prices_from_db(index_code):
    conn = get_db_connection()
    if not conn:
        return
    try:
        today = _cn_today_date()
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM etf_prices WHERE index_code=%s AND date >= %s",
            (_normalize_index_code(index_code), today),
        )
        conn.commit()
        cur.close()
    except Exception:
        pass
    finally:
        conn.close()


def _get_target_name(index_code, default_name=None):
    """获取标的的真实 name，如果不存在则返回默认值"""
    conn = get_db_connection()
    if not conn:
        return default_name or index_code
    try:
        res = pd.read_sql(
            "SELECT name FROM etf_targets WHERE index_code=%s",
            conn, params=(_normalize_index_code(index_code),),
        )
        conn.close()
        if not res.empty and pd.notna(res.iloc[0]['name']):
            return res.iloc[0]['name']
        return default_name or index_code
    except Exception:
        return default_name or index_code


def _get_target_meta(index_code, default_name=None):
    """读取标的元数据，返回 dict(name, index_code, etf_code, scaling_factor, data_source, asset_currency, report_currency)。"""
    conn = get_db_connection()
    fallback = {
        "name": default_name or index_code,
        "index_code": _normalize_index_code(index_code),
        "etf_code": "",
        "scaling_factor": 1.0,
        "data_source": "ZZ",
        "asset_currency": "CNY",
        "report_currency": "CNY",
    }
    if not conn:
        return fallback
    try:
        res = pd.read_sql(
            "SELECT name, index_code, etf_code, scaling_factor, data_source, asset_currency, report_currency FROM etf_targets WHERE index_code=%s",
            conn,
            params=(_normalize_index_code(index_code),),
        )
        conn.close()
        if res.empty:
            return fallback
        row = res.iloc[0]
        return {
            "name": row["name"] if pd.notna(row["name"]) else fallback["name"],
            "index_code": _normalize_index_code(row["index_code"]) if pd.notna(row["index_code"]) else "",
            "etf_code": _normalize_etf_code(row["etf_code"]) if pd.notna(row["etf_code"]) else "",
            "scaling_factor": float(row["scaling_factor"]) if pd.notna(row["scaling_factor"]) else 1.0,
            "data_source": _normalize_data_source(row["data_source"]) if pd.notna(row["data_source"]) else _infer_data_source_from_index_code(row["index_code"]),
            "asset_currency": _normalize_currency(row["asset_currency"]) if pd.notna(row["asset_currency"]) else "CNY",
            "report_currency": _normalize_currency(row["report_currency"]) if pd.notna(row["report_currency"]) else "CNY",
        }
    except Exception:
        return fallback





def _sync_data_from_yahoo(index_code: str):
    """Yahoo Finance 直连同步：指数点位直接入库，并附带 ETF 行情用于展示换算。"""
    _delete_today_prices_from_db(index_code)
    hist = fetch_yahoo_history(index_code, start_date="1991-01-01")
    hist = _exclude_today_rows(hist, date_col="Date")
    if hist.empty:
        raise RuntimeError(f"Yahoo Finance 无可落盘数据: {index_code}")

    meta = _get_target_meta(index_code)
    etf_code = _normalize_etf_code(meta.get("etf_code"))
    asset_currency = _normalize_currency(meta.get("asset_currency")) or "CNY"
    report_currency = _normalize_currency(meta.get("report_currency")) or "CNY"
    sf = _estimate_scaling_factor_from_overlap(hist, etf_code, default_sf=meta["scaling_factor"] if etf_code else 1.0)

    if etf_code:
        etf_all = fetch_all_from_tickflow(etf_code)
        etf_all = _exclude_today_rows(etf_all, date_col="Date")
        merged = pd.merge(hist, etf_all, on="Date", how="left")
    else:
        merged = hist.copy()
        merged["ETF_Close_Raw"] = None
        merged["ETF_Close_HFQ"] = None

    converted = _apply_currency_conversion(merged[["Date", "Index_Close"]], asset_currency=asset_currency, report_currency=report_currency)
    rows = pd.DataFrame({
        "Date": merged["Date"],
        "index_close": merged["Index_Close"],
        "etf_close_raw": merged["ETF_Close_Raw"],
        "etf_close_hfq": merged["ETF_Close_HFQ"],
        "asset_close_native": converted["asset_close_native"],
        "fx_to_cny": converted["fx_to_cny"],
        "close_cny": converted["close_cny"],
        "combined_close": converted["combined_close"],
    })
    written_rows = save_prices_to_db(rows[["Date", "index_close", "etf_close_raw", "etf_close_hfq", "asset_close_native", "fx_to_cny", "close_cny", "combined_close"]], index_code)

    target_name = meta["name"]
    save_target_to_db(
        index_code,
        target_name,
        etf_code=etf_code or None,
        scaling_factor=sf,
        stitch_date=hist["Date"].max().date(),
        data_source="YH",
        asset_currency=asset_currency,
        report_currency=report_currency,
    )
    return int(written_rows)


def _check_needs_full_stitch(index_code):
    """Check whether imported index history still lacks an ETF-stitched tail after the last index date."""
    meta = _get_target_meta(index_code)
    if not _normalize_etf_code(meta.get("etf_code")):
        return False
    conn = get_db_connection()
    if not conn:
        return False
    try:
        normalized_index_code = _normalize_index_code(index_code)
        res = pd.read_sql(
            """
            WITH history_tail AS (
                SELECT MAX(date) AS last_index_date
                FROM etf_prices
                WHERE index_code=%s
                  AND index_close IS NOT NULL
            )
            SELECT
                h.last_index_date,
                COUNT(*) FILTER (
                    WHERE p.date > h.last_index_date
                      AND p.index_close IS NULL
                      AND p.combined_close IS NOT NULL
                      AND (p.etf_close_raw IS NOT NULL OR p.etf_close_hfq IS NOT NULL)
                ) AS stitched_after_tail
            FROM history_tail h
            LEFT JOIN etf_prices p
              ON p.index_code=%s
            GROUP BY h.last_index_date
            """,
            conn,
            params=(normalized_index_code, normalized_index_code),
        )
        conn.close()
        if res.empty:
            return False
        last_index_date = res.iloc[0]['last_index_date']
        stitched_after_tail = int(res.iloc[0]['stitched_after_tail']) if pd.notna(res.iloc[0]['stitched_after_tail']) else 0
        return pd.notna(last_index_date) and stitched_after_tail == 0
    except Exception:
        return False


def _full_stitch_from_db(index_code):
    """完整拼接：从 DB 读取 index_close 数据，拉取 ETF 全量，计算 scaling_factor 并更新所有行"""
    try:
        meta = _get_target_meta(index_code)
        etf_code = _normalize_etf_code(meta.get("etf_code"))
        asset_currency = _normalize_currency(meta.get("asset_currency")) or "CNY"
        report_currency = _normalize_currency(meta.get("report_currency")) or "CNY"
        if not etf_code:
            return 0
        conn = get_db_connection()
        if not conn:
            return False
        
        # 读取 DB 中已有的 index_close 数据
        df_hist = pd.read_sql(
            "SELECT date, index_close FROM etf_prices WHERE index_code=%s AND index_close IS NOT NULL ORDER BY date",
            conn, params=(_normalize_index_code(index_code),),
        )
        conn.close()
        
        if df_hist.empty:
            return False
        
        df_hist['Date'] = pd.to_datetime(df_hist['date'])
        df_hist = df_hist.rename(columns={'index_close': 'index_close_val'})[['Date', 'index_close_val']]
        
        # 拉取 ETF 完整历史
        etf_df = fetch_all_from_tickflow(etf_code)
        etf_df = _exclude_today_rows(etf_df, date_col='Date')
        etf_df = _resolve_etf_stitch_series(etf_df)
        if etf_df.empty:
            return False
        etf_df = _resolve_etf_stitch_series(etf_df)
        
        # 用重叠窗口多点拟合缩放比例（避免单点锚定偏差）
        merged = pd.merge(df_hist, etf_df, on='Date', how='inner')
        if merged.empty:
            return False

        scaling_factor, stitch_date, _ = _estimate_scaling_from_merged(
            merged,
            index_col='index_close_val',
            etf_col='ETF_Price_For_Stitch',
            default_sf=1.0,
            window=250,
        )
        if stitch_date is None:
            return False
        
        # 构造拼接结果：合并历史和ETF，确保所有日期都被包含
        merged = pd.merge(
            df_hist.rename(columns={'index_close_val': 'index_close'}),
            etf_df,
            on='Date',
            how='outer'
        ).sort_values('Date').reset_index(drop=True)
        
        merged = merged.rename(columns={'ETF_Close_Raw': 'etf_close_raw', 'ETF_Close_HFQ': 'etf_close_hfq'})
        merged['index_close'] = pd.to_numeric(merged['index_close'], errors='coerce')
        merged['etf_price_for_stitch'] = pd.to_numeric(merged['ETF_Price_For_Stitch'], errors='coerce')

        # 构造用于换算的 `combined_close`
        # 规则：优先使用真实指数，当指数为空时，用 ETF 价格 * 缩放比例 回填
        scaled_etf_price = merged['etf_price_for_stitch'] * scaling_factor
        price_for_conversion = merged['index_close'].fillna(scaled_etf_price)

        # 统一应用汇率换算
        converted = _apply_currency_conversion(
            pd.DataFrame({'Date': merged['Date'], 'Index_Close': price_for_conversion}),
            asset_currency=asset_currency,
            report_currency=report_currency,
        )

        # 整理最终列
        result = merged[['Date', 'index_close', 'etf_close_raw', 'etf_close_hfq']].copy()
        result = pd.merge(result, converted[['Date', 'asset_close_native', 'fx_to_cny', 'close_cny', 'combined_close']], on='Date', how='left')
        
        # 更新 DB
        written_rows = save_prices_to_db(result, index_code)
        target_name = meta["name"]
        save_target_to_db(
            index_code,
            target_name,
            etf_code=etf_code,
            scaling_factor=scaling_factor,
            stitch_date=stitch_date,
            data_source=meta.get("data_source") or "ZZ",
            asset_currency=asset_currency,
            report_currency=report_currency,
        )

        return int(written_rows)
    except Exception as e:
        st.warning(f"{index_code} 全量拼接失败: {e}")
        return 0


def _incremental_tickflow_update(index_code, etf_code, scaling_factor):
    """增量刷新：每次覆盖最近三个交易日，且当日数据不落盘。"""
    meta = _get_target_meta(index_code)
    asset_currency = _normalize_currency(meta.get("asset_currency")) or "CNY"
    report_currency = _normalize_currency(meta.get("report_currency")) or "CNY"
    recent_all = fetch_recent_from_tickflow(etf_code, count=30)
    recent_all = _exclude_today_rows(recent_all, date_col='Date')
    if recent_all.empty:
        return 0
    recent_all = _resolve_etf_stitch_series(recent_all)

    patch_data = _keep_last_n_trading_days(recent_all, n=3, date_col='Date')
    if patch_data.empty:
        return 0

    patch_data = patch_data.rename(columns={'ETF_Close_Raw': 'etf_close_raw', 'ETF_Close_HFQ': 'etf_close_hfq'})
    patch_data['index_close'] = None
    native_df = pd.DataFrame({
        'Date': patch_data['Date'],
        'Index_Close': patch_data['ETF_Price_For_Stitch'] * scaling_factor,
    })
    conv = _apply_currency_conversion(native_df, asset_currency=asset_currency, report_currency=report_currency)
    patch_data['asset_close_native'] = conv['asset_close_native']
    patch_data['fx_to_cny'] = conv['fx_to_cny']
    patch_data['close_cny'] = conv['close_cny']
    patch_data['combined_close'] = conv['combined_close']
    return int(save_prices_to_db(patch_data[['Date', 'index_close', 'etf_close_raw', 'etf_close_hfq', 'asset_close_native', 'fx_to_cny', 'close_cny', 'combined_close']], index_code))


def sync_data_from_tickflow(index_code: str):
    """
    仅在手动点击「更新全部数据」时调用：
    若配置深证指数代码则优先直连深证接口；否则走 ETF 拼接/回补链路。
    规则：当日行情不落盘，仅落昨日及更早数据。
    """
    meta = _get_target_meta(index_code)
    index_code = _normalize_index_code(meta.get("index_code"))
    etf_code = _normalize_etf_code(meta.get("etf_code"))
    data_source = _normalize_data_source(meta.get("data_source")) or _infer_data_source_from_index_code(index_code)

    if data_source == "SZ":
        try:
            _delete_today_prices_from_db(index_code)
            df_existing, _ = load_from_db(index_code)
            start_date = "1991-01-01"
            if df_existing is not None and not df_existing.empty:
                last_date = df_existing['Date'].max().date()
                start_date = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

            hist = fetch_szse_index_daily(index_code=index_code, start_date=start_date, end_date="2050-01-01")
            hist = _exclude_today_rows(hist, date_col="Date")
            
            written_rows = 0
            if not hist.empty:
                meta = _get_target_meta(index_code)
                etf_code = _normalize_etf_code(meta.get("etf_code"))
                asset_currency = _normalize_currency(meta.get("asset_currency")) or "CNY"
                report_currency = _normalize_currency(meta.get("report_currency")) or "CNY"
                
                # 增量更新时，应基于全量数据估算sf，但为简化起见，此处复用已有逻辑
                sf = _estimate_scaling_factor_from_overlap(hist, etf_code, default_sf=meta["scaling_factor"] if etf_code else 1.0)

                if etf_code:
                    etf_all = fetch_all_from_tickflow(etf_code)
                    etf_all = _exclude_today_rows(etf_all, date_col="Date")
                    merged = pd.merge(hist, etf_all, on="Date", how="left")
                else:
                    merged = hist.copy()
                    merged["ETF_Close_Raw"] = None
                    merged["ETF_Close_HFQ"] = None

                converted = _apply_currency_conversion(merged[["Date", "Index_Close"]], asset_currency=asset_currency, report_currency=report_currency)
                rows = pd.DataFrame({
                    "Date": merged["Date"],
                    "index_close": merged["Index_Close"],
                    "etf_close_raw": merged["ETF_Close_Raw"],
                    "etf_close_hfq": merged["ETF_Close_HFQ"],
                    "asset_close_native": converted["asset_close_native"],
                    "fx_to_cny": converted["fx_to_cny"],
                    "close_cny": converted["close_cny"],
                    "combined_close": converted["combined_close"],
                })
                written_rows = save_prices_to_db(rows[["Date", "index_close", "etf_close_raw", "etf_close_hfq", "asset_close_native", "fx_to_cny", "close_cny", "combined_close"]], index_code)

                target_name = meta["name"]
                save_target_to_db(
                    index_code,
                    target_name,
                    etf_code=etf_code or None,
                    scaling_factor=sf,
                    stitch_date=hist["Date"].max().date(),
                    data_source="SZ",
                    asset_currency=asset_currency,
                    report_currency=report_currency,
                )

            df_latest, scaling_factor_latest = load_from_db(index_code)
            return df_latest, scaling_factor_latest, int(written_rows)
        except Exception as e:
            if not etf_code:
                raise RuntimeError(f"深证接口直连 {index_code} 失败，且当前未绑定 ETF：{e}")
            st.warning(f"深证接口直连 {index_code} 失败 ({e})，将自动走 ETF 拟合兜底方案。")
    elif data_source == "YH":
        written_rows = _sync_data_from_yahoo(index_code)
        df_latest, scaling_factor_latest = load_from_db(index_code)
        return df_latest, scaling_factor_latest, int(written_rows)

    if not etf_code:
        raise RuntimeError("ZZ 数据源必须先绑定 ETF 代码后才能拼接更新")

    _delete_today_prices_from_db(index_code)

    df, scaling_factor = load_from_db(index_code)
    written_rows = 0
    if df is None:
        raw = fetch_all_from_tickflow(etf_code)
        raw = _exclude_today_rows(raw, date_col='Date')
        if raw.empty:
            df_latest, scaling_factor_latest = load_from_db(index_code)
            return df_latest, scaling_factor_latest, 0
        raw = _resolve_etf_stitch_series(raw)
        target_name = _get_target_name(index_code)
        save_target_to_db(
            index_code,
            target_name,
            etf_code=etf_code,
            scaling_factor=1.0,
            data_source=data_source or "ZZ",
            asset_currency=meta.get("asset_currency") or "CNY",
            report_currency=meta.get("report_currency") or "CNY",
        )
        rows = pd.DataFrame({
            'Date':           raw['Date'],
            'index_close':    None,
            'etf_close_raw':  raw['ETF_Close_Raw'],
            'etf_close_hfq':  raw['ETF_Close_HFQ'],
        })
        base_native = pd.DataFrame({
            'Date': raw['Date'],
            'Index_Close': raw['ETF_Price_For_Stitch'],
        })
        base_conv = _apply_currency_conversion(
            base_native,
            asset_currency=meta.get("asset_currency") or "CNY",
            report_currency=meta.get("report_currency") or "CNY",
        )
        rows['asset_close_native'] = base_conv['asset_close_native']
        rows['fx_to_cny'] = base_conv['fx_to_cny']
        rows['close_cny'] = base_conv['close_cny']
        rows['combined_close'] = base_conv['combined_close']
        written_rows = save_prices_to_db(rows, index_code)
    else:
        # 检查是否需要完整拼接（历史数据未初始化 etf_close_raw）
        if _check_needs_full_stitch(index_code):
            written_rows = _full_stitch_from_db(index_code)
        else:
            written_rows = _incremental_tickflow_update(index_code, etf_code, scaling_factor)

    df_latest, scaling_factor_latest = load_from_db(index_code)
    return df_latest, scaling_factor_latest, int(written_rows)


@st.cache_data(ttl=3600, show_spinner=False)
def get_data(index_code: str):
    """
    页面加载只读数据库，不触发任何网络请求。
    实时更新请点击侧边栏「更新全部数据」。
    """
    return load_from_db(index_code)


# ─── 核心分析 ─────────────────────────────────────────────────────────────────
def compute_and_plot(df, etf_name, deviation_pct, tradition_start, tradition_end, rolling_window=1250, ma_window=250, scaling_factor=1.0):
    df = df.copy()
    df['Log_Close'] = np.log(df['Close'])
    # 计算物理时间差（自然年），代替固定的行号索引
    df['Years_Passed'] = (df['Date'] - df['Date'].iloc[0]).dt.days / 365.25
    tradition_start_dt = pd.to_datetime(tradition_start)
    tradition_end_dt = pd.to_datetime(tradition_end)

    # 传统回归
    mask = (df['Date'] >= tradition_start_dt) & \
           (df['Date'] <= tradition_end_dt)
    sample_df = df[mask]
    if len(sample_df) < 100:
        raise ValueError(f"传统回归样本不足（{len(sample_df)} 条），请检查数据起止日期")
    k_trad, b_trad = np.polyfit(sample_df['Years_Passed'], sample_df['Log_Close'], 1)
    df['Trad_Pred_Log']   = k_trad * df['Years_Passed'] + b_trad
    df['Trad_Pred_Price'] = np.exp(df['Trad_Pred_Log'])
    resids_trad = sample_df['Log_Close'] - (k_trad * sample_df['Years_Passed'] + b_trad)
    std_trad    = np.std(resids_trad)
    df['Trad_Z_Score'] = (df['Log_Close'] - df['Trad_Pred_Log']) / std_trad
    z_plus  = np.log(1 + deviation_pct / 100.0) / std_trad
    z_minus = np.log(1 - deviation_pct / 100.0) / std_trad

    # 滚动回归
    rolling_preds = np.full(len(df), np.nan)
    rolling_z     = np.full(len(df), np.nan)
    k_roll_last   = np.nan
    for i in range(rolling_window, len(df)):
        ys = df['Log_Close'].values[i - rolling_window:i]
        xs = df['Years_Passed'].values[i - rolling_window:i]
        k_r, b_r = np.polyfit(xs, ys, 1)
        pred = k_r * xs[-1] + b_r
        rolling_preds[i] = pred
        std_r = np.std(ys - (k_r * xs + b_r))
        if std_r > 0:
            rolling_z[i] = (ys[-1] - pred) / std_r
        k_roll_last = k_r
    df['Roll_Pred_Price'] = np.exp(rolling_preds)
    df['Roll_Z_Score']    = rolling_z

    # MA（可调窗口）
    df['MA_Price'] = df['Close'].rolling(window=ma_window, min_periods=ma_window).mean()
    ma_log_diff = np.full(len(df), np.nan)
    ma_mask = df['MA_Price'].notna() & (df['MA_Price'] > 0)
    ma_log_diff[ma_mask.values] = (
        df.loc[ma_mask, 'Log_Close'].values - np.log(df.loc[ma_mask, 'MA_Price'].values)
    )
    std_ma = np.nanstd(ma_log_diff)
    if pd.notna(std_ma) and std_ma > 0:
        df['MA_Z_Score'] = ma_log_diff / std_ma
    else:
        df['MA_Z_Score'] = np.nan

    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                   gridspec_kw={'height_ratios': [2.5, 1]})
    ax1.plot(df['Date'], df['Close'], color='black', linewidth=1.2, label='指数')
    ax1.plot(df['Date'], df['Trad_Pred_Price'], color='red', linestyle='--',
             linewidth=2, label=f'传统回归({tradition_start_dt.year}-{tradition_end_dt.year})')
    ax1.fill_between(df['Date'],
                     np.exp(df['Trad_Pred_Log'] - 2 * std_trad),
                     np.exp(df['Trad_Pred_Log'] + 2 * std_trad),
                     color='red', alpha=0.1, label='传统通道(±2σ)')
    ax1.plot(df['Date'], df['Roll_Pred_Price'], color='blue', linestyle='-.',
             linewidth=1.5, label=f'滚动回归({rolling_window}日)')
    ax1.plot(df['Date'], df['MA_Price'], color='#2F9E44', linestyle='-',
             linewidth=1.5, label=f'MA{ma_window}')
    ax1.set_yscale('log')
    ax1.set_title(f'{etf_name} 全收益', fontsize=15, fontweight='bold')
    ax1.set_ylabel('点位（对数）', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, which='both', linestyle=':', alpha=0.6)

    ax2.axhline(0,  color='black', linewidth=1)
    ax2.axhline(2,  color='red',   linestyle='--', alpha=0.5, label='+2σ')
    ax2.axhline(-2, color='green', linestyle='--', alpha=0.5, label='-2σ')
    ax2.axhline(z_plus,  color='orange', linestyle=':', alpha=0.9,
                label=f'+{deviation_pct:.0f}%(传统Z)')
    ax2.axhline(z_minus, color='orange', linestyle=':', alpha=0.9,
                label=f'-{deviation_pct:.0f}%(传统Z)')
    ax2.plot(df['Date'], df['Roll_Z_Score'], color='blue', linestyle='-',
             alpha=0.8, linewidth=1.2, label='滚动Z', zorder=2)
    ax2.plot(df['Date'], df['Trad_Z_Score'], color='black',
             alpha=0.95, linewidth=1.5, label='传统Z', zorder=3)
    ax2.plot(df['Date'], df['MA_Z_Score'], color='#2F9E44', linestyle='-',
             alpha=0.9, linewidth=1.2, label=f'MA{ma_window}-Z', zorder=2)
    ax2.set_title('Z-Score 偏离度', fontsize=13)
    ax2.set_ylabel('Z-Score', fontsize=11)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.6)

    yl = mdates.YearLocator(1)
    yf = mdates.DateFormatter('%Y')
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(yl)
        ax.xaxis.set_major_formatter(yf)
        ax.set_xlim(df['Date'].min(), df['Date'].max())
        ax.margins(x=0)
    fig.autofmt_xdate()
    plt.tight_layout()

    latest_close = float(df['Close'].iloc[-1])
    trad_pred    = float(df['Trad_Pred_Price'].iloc[-1])
    roll_pred    = float(df['Roll_Pred_Price'].iloc[-1])
    ma_pred   = float(df['MA_Price'].iloc[-1]) if pd.notna(df['MA_Price'].iloc[-1]) else np.nan

    if pd.notna(ma_pred) and ma_pred > 0:
        dev_ma = (latest_close / ma_pred - 1) * 100
    else:
        dev_ma = np.nan

    # 展示口径：优先使用原生不复权 ETF 收盘；缺失时才使用缩放换算
    def to_etf(point_value):
        if scaling_factor > 0:
            return point_value / scaling_factor
        return point_value

    latest_raw = np.nan
    if 'ETF_Close_Raw' in df.columns:
        latest_raw = pd.to_numeric(df['ETF_Close_Raw'].iloc[-1], errors='coerce')
    latest_etf_price = float(latest_raw) if pd.notna(latest_raw) else to_etf(latest_close)
    
    return fig, {
        "latest_date":  df['Date'].iloc[-1].strftime('%Y-%m-%d'),
        "trad_range_start": sample_df['Date'].iloc[0].strftime('%Y-%m-%d'),
        "trad_range_end": sample_df['Date'].iloc[-1].strftime('%Y-%m-%d'),
        "latest_close": latest_close,
        "latest_etf_price": latest_etf_price,
        "trad_pred":    trad_pred,
        "trad_pred_etf": to_etf(trad_pred),
        "roll_pred":    roll_pred,
        "roll_pred_etf": to_etf(roll_pred),
        "ma_pred":   ma_pred,
        "ma_pred_etf": to_etf(ma_pred) if pd.notna(ma_pred) else np.nan,
        "dev_trad":     (latest_close / trad_pred - 1) * 100,
        "dev_roll":     (latest_close / roll_pred - 1) * 100,
        "dev_ma":       dev_ma,
        "cagr_trad":    (np.exp(k_trad) - 1) * 100,
        "cagr_roll":    (np.exp(k_roll_last) - 1) * 100,
        "scaling_factor": scaling_factor,
        "z_plus": z_plus,
        "z_minus": z_minus,
        "std_trad": std_trad,
        "plot_df": df[[
            'Date', 'Close', 'Trad_Pred_Price', 'Roll_Pred_Price', 'MA_Price',
            'Trad_Z_Score', 'Roll_Z_Score', 'MA_Z_Score', 'Trad_Pred_Log',
        ]].copy(),
    }


def render_native_charts(res, etf_name, deviation_pct, tradition_start, tradition_end, rolling_window=1250, ma_window=250, date_range=None):
    """使用 Plotly 渲染，现代简洁风格，含置信带、对数坐标、悬停。"""
    df       = res['plot_df'].copy()
    if date_range is not None and len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0], errors='coerce')
        end_date = pd.to_datetime(date_range[1], errors='coerce')
        if pd.notna(start_date) and pd.notna(end_date):
            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
    if df.empty:
        st.info("当前时间范围内暂无可展示数据。")
        return
    z_plus   = float(res['z_plus'])
    z_minus  = float(res['z_minus'])
    std_trad = float(res['std_trad'])
    tradition_start_dt = pd.to_datetime(tradition_start)
    tradition_end_dt = pd.to_datetime(tradition_end)

    # 传统回归置信带（对数空间 ±2σ）
    band_upper = np.exp(df['Trad_Pred_Log'] + 2 * std_trad)
    band_lower = np.exp(df['Trad_Pred_Log'] - 2 * std_trad)

    # ── 配色（海洋清风）────────────────────────────────────────────
    C_INDEX  = '#51999F'   # 海蓝绿 - 指数主线
    C_TRAD   = '#D97745'   # 暖橙棕 - 传统回归
    C_ROLL   = '#2D8CFF'   # 明亮蓝 - 滚动回归
    C_MA  = '#2F9E44'   # 绿色 - MA
    C_BAND   = 'rgba(236,182,108,0.16)'  # 暖沙色置信带
    C_TZERO  = '#C8CDD2'
    C_SIGMA  = '#EA9E58'
    C_THRESH = '#DBCB92'

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[f'{etf_name}  全收益指数', 'Z-Score 偏离度'],
    )

    # ── 置信带（先画，让它垫在线条下面）─────────────────────────────
    fig.add_trace(go.Scatter(
        x=pd.concat([df['Date'], df['Date'][::-1]]),
        y=pd.concat([band_upper, band_lower[::-1]]),
        fill='toself', fillcolor=C_BAND,
        line=dict(width=0), showlegend=True, name='传统通道 ±2σ',
        hoverinfo='skip',
    ), row=1, col=1)

    # ── 传统回归线（虚线）────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Trad_Pred_Price'],
        name=f'传统回归 ({tradition_start_dt.year}–{tradition_end_dt.year})',
        line=dict(color=C_TRAD, width=1.35, dash='solid'),
        hovertemplate='%{x|%Y-%m-%d}  传统: %{y:,.1f}<extra></extra>',
    ), row=1, col=1)

    # ── 滚动回归线（点划线）──────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Roll_Pred_Price'],
        name=f'滚动回归 ({rolling_window}日)',
        line=dict(color=C_ROLL, width=1.35, dash='dashdot'),
        hovertemplate='%{x|%Y-%m-%d}  滚动: %{y:,.1f}<extra></extra>',
    ), row=1, col=1)

    # ── MA ────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['MA_Price'],
        name=f'MA{ma_window}',
        line=dict(color=C_MA, width=1.35, dash='solid'),
        hovertemplate='%{x|%Y-%m-%d}  MA: %{y:,.1f}<extra></extra>',
    ), row=1, col=1)

    # ── 指数主线（最顶层）────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'],
        name='指数', line=dict(color=C_INDEX, width=1.3),
        hovertemplate='%{x|%Y-%m-%d}  指数: %{y:,.1f}<extra></extra>',
    ), row=1, col=1)

    # ── Z-Score 主线 ──────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Roll_Z_Score'],
        name='滚动Z', line=dict(color=C_ROLL, width=1.2, dash='solid'),
        hovertemplate='%{x|%Y-%m-%d}  滚动Z: %{y:.3f}<extra></extra>',
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Trad_Z_Score'],
        name='传统Z', line=dict(color=C_INDEX, width=1.2),
        hovertemplate='%{x|%Y-%m-%d}  传统Z: %{y:.3f}<extra></extra>',
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['MA_Z_Score'],
        name=f'MA{ma_window}-Z', line=dict(color=C_MA, width=1.2),
        hovertemplate='%{x|%Y-%m-%d}  MA-Z: %{y:.3f}<extra></extra>',
    ), row=2, col=1)

    # ── Z-Score 水平参考线 ────────────────────────────────────────
    for y_val, color, dash, label in [
        (0,       C_TZERO,  'solid',  ''),
        (2,       C_SIGMA,  'dash',   '+2σ'),
        (-2,      C_SIGMA,  'dash',   '-2σ'),
        (z_plus,  C_THRESH, 'dot',    f'+{deviation_pct:.0f}%'),
        (z_minus, C_THRESH, 'dot',    f'-{deviation_pct:.0f}%'),
    ]:
        fig.add_hline(
            y=y_val, row=2, col=1,
            line=dict(color=color, width=1.0, dash=dash),
            annotation_text=label,
            annotation_font=dict(size=11, color=color),
            annotation_position='right',
        )

    # ── 布局：白底、轻网格、无边框感 ─────────────────────────────────
    axis_common = dict(
        showgrid=True, gridcolor='rgba(200,200,200,0.4)',
        zeroline=False, showline=False,
        tickfont=dict(size=11),
    )
    fig.update_yaxes(
        type='log',
        title_text='点位（对数）',
        tickformat=',.0f',
        dtick='D1',
        row=1,
        col=1,
        **axis_common,
    )
    fig.update_yaxes(title_text='Z-Score',                 row=2, col=1, **axis_common)
    fig.update_xaxes(
        dtick='M12',
        tickformat='%Y',
        ticklabelmode='instant',
        **axis_common,
    )
    fig.update_layout(
        height=720,
        hovermode='x unified',
        dragmode='pan',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='PingFang SC, Microsoft YaHei, sans-serif', size=12),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02,
            xanchor='left', x=0,
            bgcolor='rgba(255,255,255,0.8)',
            borderwidth=0,
        ),
        margin=dict(l=60, r=80, t=80, b=40),
        hoverlabel=dict(bgcolor='white', bordercolor='#ccc', font_size=12),
    )
    # 子图标题样式
    for ann in fig.layout.annotations:
        ann.font.size = 13
        ann.font.color = '#333333'

    st.plotly_chart(fig, use_container_width=True)


# ─── 全市场对比 ───────────────────────────────────────────────────────────────
def build_comparison(deviation_pct, etf_config, tradition_start, tradition_end, rolling_window=1250, ma_window=250):
    ma_dev_col = f"ma_{ma_window}_deviation_pct"
    trad_range_col = "trad_cagr_range"
    rows = []
    for name, cfg in etf_config.items():
        display_etf_code = cfg.get('etf_code') or "-"
        try:
            df, scaling_factor = get_data(cfg['index_code'])
        except Exception as e:
            rows.append({"name": name, "etf_code": display_etf_code,
                         "latest_date": f"加载失败: {e}",
                         "trad_deviation_pct": None, "roll_deviation_pct": None, ma_dev_col: None,
                         "trad_cagr_pct": None, "roll_cagr_pct": None, trad_range_col: None})
            continue
        if df is None or len(df) < rolling_window + 10:
            rows.append({"name": name, "etf_code": display_etf_code,
                         "latest_date": "无数据（请先拼接入库）",
                         "trad_deviation_pct": None, "roll_deviation_pct": None, ma_dev_col: None,
                         "trad_cagr_pct": None, "roll_cagr_pct": None, trad_range_col: None})
            continue
        try:
            fig, res = compute_and_plot(df, name, deviation_pct, tradition_start, tradition_end, rolling_window, ma_window, scaling_factor)
            plt.close(fig)
            rows.append({
                "name": name, "etf_code": display_etf_code,
                "latest_date": res['latest_date'],
                "trad_deviation_pct": round(res['dev_trad'], 2),
                "roll_deviation_pct": round(res['dev_roll'], 2),
                ma_dev_col: round(res['dev_ma'], 2) if pd.notna(res['dev_ma']) else None,
                "trad_cagr_pct": round(res['cagr_trad'], 2),
                "roll_cagr_pct": round(res['cagr_roll'], 2),
                trad_range_col: f"{res['trad_range_start']} ~ {res['trad_range_end']}",
            })
        except Exception as e:
            rows.append({"name": name, "etf_code": display_etf_code,
                         "latest_date": f"出错: {e}",
                         "trad_deviation_pct": None, "roll_deviation_pct": None, ma_dev_col: None,
                         "trad_cagr_pct": None, "roll_cagr_pct": None, trad_range_col: None})
    return pd.DataFrame(rows)


# ─── 数据管理工具 ─────────────────────────────────────────────────────────────
def parse_upload_file(uploaded_file):
    """解析 Excel/CSV 文件，返回 (DataFrame, message)"""
    try:
        # parse_dates=False 阻止 pandas 自动转换日期，避免后续误判
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, parse_dates=False)
        else:
            df = pd.read_excel(uploaded_file, parse_dates=False)
        
        # 智能检测日期和收盘列
        date_col = None
        close_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if '日期' in col_lower or 'date' in col_lower or col_lower == '时间':
                date_col = col
            if '收盘' in col_lower or 'close' in col_lower or '点位' in col_lower:
                close_col = col
        
        if date_col is None or close_col is None:
            return None, f"❌ 无法识别日期列或收盘列。找到的列: {list(df.columns)}"
        
        df = df[[date_col, close_col]].rename(columns={date_col: 'Date', close_col: 'Close'})
        raw_date = df['Date']

        # ── 情况1：pandas 读出的已经是 datetime / Timestamp 类型 ──────────────
        if pd.api.types.is_datetime64_any_dtype(raw_date):
            parsed_date = raw_date
        else:
            # ── 情况2：字符串或数字，需要手动解析 ───────────────────────────────
            numeric_raw  = pd.to_numeric(raw_date, errors='coerce')
            parsed_date  = pd.Series(pd.NaT, index=df.index, dtype='datetime64[ns]')

            # 文本日期（如 "2020-01-01" / "2020/01/01" / "20200101"）
            text_mask = numeric_raw.isna()
            if text_mask.any():
                parsed_date.loc[text_mask] = pd.to_datetime(
                    raw_date.loc[text_mask], infer_datetime_format=True, errors='coerce'
                )

            # 数字日期：按量级判断类型，完全避免 1970 误判
            num_mask = numeric_raw.notna()
            if num_mask.any():
                n = numeric_raw.loc[num_mask]
                fixed = pd.Series(pd.NaT, index=n.index, dtype='datetime64[ns]')

                # YYYYMMDD 整数（优先匹配，避免与 Excel 序列号混淆）
                ymd_mask = (n >= 19000101) & (n <= 21001231)
                if ymd_mask.any():
                    ymd_str = n.loc[ymd_mask].round().astype('Int64').astype(str)
                    fixed.loc[ymd_mask] = pd.to_datetime(
                        ymd_str, format='%Y%m%d', errors='coerce'
                    )

                # Excel 序列日期（1900-2100 对应 ~14000–73050 天）
                excel_mask = (n >= 14000) & (n <= 80000) & ~ymd_mask
                if excel_mask.any():
                    fixed.loc[excel_mask] = pd.to_datetime(
                        n.loc[excel_mask], unit='D', origin='1899-12-30', errors='coerce'
                    )

                parsed_date.loc[num_mask] = fixed

        df['Date']  = pd.to_datetime(parsed_date, errors='coerce')
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna(subset=['Date', 'Close'])
        df = df.sort_values('Date').reset_index(drop=True)

        # 合理性检查：日期应在 1990 年之后
        bad = (df['Date'].dt.year < 1990).sum()
        if bad > 0:
            return None, (
                f"❌ 检测到 {bad} 条日期早于 1990 年（可能解析错误），"
                f"请检查文件格式。首行日期原始值: {raw_date.iloc[0]!r}"
            )

        return df, f"✅ 成功解析 {len(df)} 条数据（{df['Date'].min().date()} ~ {df['Date'].max().date()}）"
    except Exception as e:
        return None, f"❌ 解析文件出错: {e}"


def stitch_with_tickflow(history_df, etf_code, asset_currency="CNY", report_currency="CNY"):
    """
    将历史指数数据（history_df: Date, Close=指数点位）与 TickFlow ETF 数据拼接。
    返回 (structured_df, scaling_factor, stitch_date, message)
    structured_df 列：Date, index_close, etf_close_raw, etf_close_hfq, combined_close
    
    注意：本函数用于「已有标的：上传并拼接」流程，需要自动从 TickFlow 拉取数据。
    """
    try:
        etf_df = fetch_all_from_tickflow(etf_code)
        etf_df = _exclude_today_rows(etf_df, date_col='Date')
        etf_df = _resolve_etf_stitch_series(etf_df)
        if etf_df.empty:
            return None, 1.0, None, "❌ TickFlow暂无可落盘的历史日线（当日数据不会落盘）"

        # 外连接历史和 TickFlow
        merged = pd.merge(
            history_df[['Date', 'Close']].rename(columns={'Close': 'index_close'}),
            etf_df,
            on='Date', how='outer',
        ).sort_values('Date')

        merged['etf_close_raw'] = pd.to_numeric(merged['ETF_Close_Raw'], errors='coerce')
        merged['etf_close_hfq'] = pd.to_numeric(merged['ETF_Close_HFQ'], errors='coerce')
        merged['etf_price_for_stitch'] = pd.to_numeric(merged['ETF_Price_For_Stitch'], errors='coerce')

        # 使用重叠窗口多点拟合缩放比例，找不到重叠样本才回落到 1.0
        overlap = merged[merged['index_close'].notna() & merged['etf_price_for_stitch'].notna()]
        if not overlap.empty:
            scaling_factor, stitch_date, used_n = _estimate_scaling_from_merged(
                overlap,
                index_col='index_close',
                etf_col='etf_price_for_stitch',
                default_sf=1.0,
                window=250,
            )
            if stitch_date is None:
                scaling_factor = 1.0
                stitch_date = history_df['Date'].max().date()
                msg_prefix = "⚠️ 重叠样本无有效价格，缩放比例暂设为 1.0"
            else:
                msg_prefix = f"✅ 拼接成功，缩放比例: {scaling_factor:.4f}（多点拟合样本: {used_n}）"
        else:
            scaling_factor = 1.0
            stitch_date    = history_df['Date'].max().date()
            msg_prefix = "⚠️ 历史数据与 TickFlow 无重叠日期，缩放比例暂设为 1.0"

        # 基于外连接的完整数据集，填充并计算衍生列
        last_hist_date = history_df['Date'].max()
        merged['index_close'] = pd.to_numeric(merged['index_close'], errors='coerce')

        # 构造用于换算和最终输出的 `combined_close`
        # 规则：优先使用真实指数，当指数为空时，用 ETF 价格 * 缩放比例 回填
        scaled_etf_price = merged['etf_price_for_stitch'] * scaling_factor
        price_for_conversion = merged['index_close'].fillna(scaled_etf_price)
        
        # 统一应用汇率换算
        converted = _apply_currency_conversion(
            pd.DataFrame({'Date': merged['Date'], 'Index_Close': price_for_conversion}),
            asset_currency=asset_currency,
            report_currency=report_currency,
        )

        # 整理最终列
        result = merged[['Date', 'index_close', 'etf_close_raw', 'etf_close_hfq']].copy()
        result = pd.merge(result, converted[['Date', 'asset_close_native', 'fx_to_cny', 'close_cny', 'combined_close']], on='Date', how='left')

        num_new = (merged['Date'] > last_hist_date).sum()
        return result, scaling_factor, stitch_date, \
               msg_prefix + f"，新增近期 {num_new} 条"
    except Exception as e:
        return None, 1.0, None, f"❌ 拼接失败: {e}"



# ─── Streamlit UI ─────────────────────────────────────────────────────────────
st.title("📈 今天买什么")

# 每次页面 rerun 都从数据库同步标的列表
st.session_state["etf_config_runtime"] = load_targets_from_db()

ACTIVE_ETF_CONFIG = st.session_state["etf_config_runtime"]
deviation_pct = 15
ma_window = 250
rolling_window = DEFAULT_ROLLING_WINDOW
selected = None
tradition_start = DEFAULT_TRADITION_START
tradition_end = pd.Timestamp.today().date()

with st.sidebar:
    st.header("⚙️ 参数设置")
    ACTIVE_ETF_CONFIG = st.session_state["etf_config_runtime"]

    if not ACTIVE_ETF_CONFIG:
        st.info("数据库暂无标的，请在「数据管理」中新增。")
    else:
        selected = st.selectbox("选择标的", list(ACTIVE_ETF_CONFIG.keys()))
        deviation_pct = st.slider("偏离阈值 (%)", 5, 30, 15, 1)
        ma_window = st.number_input("MA周期", min_value=1, max_value=1000, value=250, step=5)
        rolling_window = st.number_input("滚动回归周期", min_value=1, value=DEFAULT_ROLLING_WINDOW, step=5)
        today = pd.Timestamp.today().date()
        tradition_start = st.date_input(
            "传统回归起始日期",
            value=DEFAULT_TRADITION_START,
            min_value=DEFAULT_TRADITION_START,
            max_value=today,
            format="YYYY/MM/DD",
            key="tradition_start_date",
        )
        tradition_end = st.date_input(
            "传统回归结束日期",
            value=today,
            min_value=DEFAULT_TRADITION_START,
            max_value=today,
            format="YYYY/MM/DD",
            key="tradition_end_date",
        )
        st.caption("旧固定区间参考：20081031-20221031")
        if tradition_start > tradition_end:
            st.warning("起始日期不能晚于结束日期，已自动调整为结束日期。")
            tradition_start = tradition_end

    st.divider()
    if st.button("🔄 更新全部数据", use_container_width=True, type="primary"):
        st.cache_data.clear()
        prog = st.progress(0)
        etf_list = list(ACTIVE_ETF_CONFIG.items())
        for idx, (name, cfg) in enumerate(etf_list):
            with st.spinner(f"拉取 {name}..."):
                try:
                    _, _, written_rows = sync_data_from_tickflow(cfg['index_code'])
                    if written_rows > 0:
                        st.caption(f"🗄️ {name} 本次落库 {written_rows} 条")
                    else:
                        st.caption(f"🗄️ {name} 本次无新增落库")
                except Exception as e:
                    st.warning(f"{name} 失败: {e}")
                    prog.progress((idx + 1) / len(etf_list))
                    continue
            prog.progress((idx + 1) / len(etf_list))

        st.success("✅ 全部数据已更新！")
        st.rerun()

    st.divider()
    if DATABASE_URL:
        try:
            conn = get_db_connection()
            if conn:
                summary = pd.read_sql(
                    "SELECT p.index_code, t.name, t.etf_code, COUNT(*) AS 条数, MAX(p.date) AS 最新日期 "
                    "FROM etf_prices p LEFT JOIN etf_targets t ON p.index_code = t.index_code "
                    "GROUP BY p.index_code, t.name, t.etf_code ORDER BY p.index_code",
                    conn,
                )
                conn.close()
                if not summary.empty:
                    st.caption("📊 数据库概览")
                    st.dataframe(summary, hide_index=True, use_container_width=True)
        except Exception as e:
            st.caption(f"⚠️ 无法查询数据库: {e}")
    else:
        st.warning("⚠️ 未配置 database_url，请先设置 Streamlit secrets")

tab1, tab2, tab3 = st.tabs(["📊 单标的详情", "📋 全市场对比", "⚙️ 数据管理"])

with tab1:
    if not selected:
        st.info("请先在「数据管理」中新增标的。")
    else:
        cfg = ACTIVE_ETF_CONFIG[selected]
        index_code = cfg['index_code']
        etf_name = cfg['name']

        with st.spinner(f"加载 {etf_name} ({index_code}) 数据..."):
            try:
                df, scaling_factor = get_data(index_code)
            except Exception as e:
                st.error(f"❌ 数据加载失败：{e}")
                st.stop()

        if df is None or len(df) < rolling_window + 10:
            st.error("数据不足，无法计算回归，请先在「数据管理」中上传历史指数文件并拼接。")
        else:
            try:
                fig, res = compute_and_plot(df, etf_name, deviation_pct, tradition_start, tradition_end, rolling_window, ma_window, scaling_factor)
                plot_df = res['plot_df'].copy()
                min_detail_date = plot_df['Date'].min().date()
                max_detail_date = plot_df['Date'].max().date()
                chart_placeholder = st.empty()
                selected_detail_range = st.slider(
                    "详情时间范围",
                    min_value=min_detail_date,
                    max_value=max_detail_date,
                    value=(min_detail_date, max_detail_date),
                    format="YYYY-MM-DD",
                    key=f"detail_range_{index_code}",
                )
                with chart_placeholder.container():
                    render_native_charts(
                        res,
                        etf_name,
                        deviation_pct,
                        tradition_start,
                        tradition_end,
                        rolling_window,
                        ma_window,
                        date_range=selected_detail_range,
                    )
                plt.close(fig)

                st.caption("ETF价格展示口径：优先原生不复权；缺失时才使用缩放换算")

                st.divider()
                st.subheader("指数点位 & ETF价格")
                c1, c2, c3 = st.columns(3)
                c1.metric("最新日期", res['latest_date'])
                c2.metric("指数点位", f"{res['latest_close']:,.0f}")
                c3.metric("ETF原生不复权", f"{res['latest_etf_price']:.4f}")

                st.subheader("传统回归")
                c4, c5, c6, c7 = st.columns(4)
                c4.metric("指数点位", f"{res['trad_pred']:,.0f}")
                c5.metric("换算预估价格", f"{res['trad_pred_etf']:.4f}")
                c6.metric("偏离度", f"{res['dev_trad']:+.2f}%", delta_color="inverse")
                c7.metric("年化", f"{res['cagr_trad']:.2f}%")

                st.subheader("滚动回归")
                c8, c9, c10, c11 = st.columns(4)
                c8.metric("指数点位", f"{res['roll_pred']:,.0f}")
                c9.metric("换算预估价格", f"{res['roll_pred_etf']:.4f}")
                c10.metric("偏离度", f"{res['dev_roll']:+.2f}%", delta_color="inverse")
                c11.metric("年化", f"{res['cagr_roll']:.2f}%")

                st.subheader(f"MA{ma_window}")
                c12, c13, c14 = st.columns(3)
                c12.metric("指数点位", f"{res['ma_pred']:,.0f}" if pd.notna(res['ma_pred']) else "—")
                c13.metric("换算预估价格", f"{res['ma_pred_etf']:.4f}" if pd.notna(res['ma_pred_etf']) else "—")
                c14.metric("偏离度", f"{res['dev_ma']:+.2f}%" if pd.notna(res['dev_ma']) else "—", delta_color="inverse")
            except Exception as e:
                st.error(f"计算出错：{e}")

with tab2:
    if not ACTIVE_ETF_CONFIG:
        st.info("暂无标的数据。")
    else:
        st.caption("对比数据来自数据库，更新请点击侧边栏「更新全部数据」")
        ma_dev_col = f"ma_{ma_window}_deviation_pct"
        display_columns = {
            "name": "标的",
            "etf_code": "ETF代码",
            "latest_date": "最新日期",
            "trad_deviation_pct": "传统偏离度(%)",
            "roll_deviation_pct": "滚动偏离度(%)",
            ma_dev_col: f"MA{ma_window}偏离度(%)",
            "trad_cagr_pct": "传统CAGR(%)",
            "roll_cagr_pct": "滚动CAGR(%)",
            "trad_cagr_range": "传统年化范围",
        }
        with st.spinner("计算全市场偏离度..."):
            compare_df = build_comparison(deviation_pct, ACTIVE_ETF_CONFIG, tradition_start, tradition_end, rolling_window, ma_window)

        if compare_df.empty:
            st.info("无数据，请先拼接入库。")
        else:
            numeric_cols = ["trad_deviation_pct", "roll_deviation_pct", ma_dev_col]
            display_df = compare_df.rename(columns=display_columns)
            gradient_subset = [display_columns[c] for c in numeric_cols if c in compare_df.columns]
            styled = display_df.style.background_gradient(
                subset=gradient_subset,
                cmap="coolwarm", vmin=-100, vmax=100,
            ).format({
                "传统偏离度(%)": lambda x: f"{x:+.2f}" if pd.notna(x) else "—",
                "滚动偏离度(%)": lambda x: f"{x:+.2f}" if pd.notna(x) else "—",
                f"MA{ma_window}偏离度(%)": lambda x: f"{x:+.2f}" if pd.notna(x) else "—",
                "传统CAGR(%)": lambda x: f"{x:.2f}" if pd.notna(x) else "—",
                "滚动CAGR(%)": lambda x: f"{x:.2f}" if pd.notna(x) else "—",
            })
            st.dataframe(styled, use_container_width=True, hide_index=True)

            plot_df = compare_df.dropna(subset=["trad_deviation_pct", "roll_deviation_pct", ma_dev_col])
            if not plot_df.empty:
                st.subheader("偏离度对比")
                plot_df = plot_df.copy().sort_values("trad_deviation_pct", ascending=True)

                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    y=plot_df["name"],
                    x=plot_df["trad_deviation_pct"],
                    name="传统偏离度",
                    orientation='h',
                    marker=dict(color="#BFDFD2"),
                    opacity=0.88,
                    hovertemplate="标的: %{y}<br>传统偏离度: %{x:+.2f}%<extra></extra>",
                ))
                fig2.add_trace(go.Bar(
                    y=plot_df["name"],
                    x=plot_df["roll_deviation_pct"],
                    name="滚动偏离度",
                    orientation='h',
                    marker=dict(color="#7BC0CD"),
                    opacity=0.88,
                    hovertemplate="标的: %{y}<br>滚动偏离度: %{x:+.2f}%<extra></extra>",
                ))
                fig2.add_trace(go.Bar(
                    y=plot_df["name"],
                    x=plot_df[ma_dev_col],
                    name=f"MA{ma_window}偏离度",
                    orientation='h',
                    marker=dict(color="#2F9E44"),
                    opacity=0.88,
                    hovertemplate=f"标的: %{{y}}<br>MA{ma_window}偏离度: %{{x:+.2f}}%<extra></extra>",
                ))

                fig2.add_vline(x=0, line_color="#666", line_width=1)
                fig2.add_vline(x=deviation_pct, line_color="#D4A43E", line_dash="dot", line_width=1)
                fig2.add_vline(x=-deviation_pct, line_color="#D4A43E", line_dash="dot", line_width=1)

                fig2.update_layout(
                    barmode="group",
                    bargap=0.22,
                    height=max(360, 52 * len(plot_df)),
                    template="plotly_white",
                    margin=dict(l=20, r=20, t=10, b=20),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                    ),
                )
                fig2.update_xaxes(title="偏离度 (%)")
                fig2.update_yaxes(title=None)
                st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("📤 数据管理")

    mode = st.radio(
        "选择操作模式",
        ["已有标的：上传并拼接", "新增标的：绑定并导入", "已有标的：补绑ETF"],
        horizontal=True,
    )

    if mode == "已有标的：上传并拼接":
        if not ACTIVE_ETF_CONFIG:
            st.info("暂无标的，请先新增标的。")
        else:
            col_upload, col_stitch = st.columns(2)

            with col_upload:
                st.write("**第1步：上传该标的的指数历史文件**")
                uploaded_file = st.file_uploader(
                    "选择 Excel 或 CSV 文件",
                    type=['xlsx', 'xls', 'csv'],
                    key="existing_target_history_file",
                )

                df_uploaded = None
                if uploaded_file:
                    df_uploaded, msg = parse_upload_file(uploaded_file)
                    st.info(msg)
                    if df_uploaded is not None:
                        st.write("📊 预览（前10行）:")
                        st.dataframe(df_uploaded.head(10), use_container_width=True)

            with col_stitch:
                st.write("**第2步：选择已绑定 ETF 的标的，一键拼接入库**")
                stitch_candidates = [name for name, cfg in ACTIVE_ETF_CONFIG.items() if cfg.get("etf_code")]
                selected_etf = st.selectbox("选择标的", stitch_candidates, key="stitch_etf") if stitch_candidates else None

                if st.button("🔗 拼接并保存到数据库", use_container_width=True, type="primary"):
                    if df_uploaded is None:
                        st.error("❌ 请先上传并解析历史文件")
                    elif not selected_etf:
                        st.error("❌ 当前没有已绑定 ETF 的标的可供拼接")
                    else:
                        index_code = ACTIVE_ETF_CONFIG[selected_etf]['index_code']
                        etf_code = ACTIVE_ETF_CONFIG[selected_etf]['etf_code']
                        asset_currency = ACTIVE_ETF_CONFIG[selected_etf].get('asset_currency') or "CNY"
                        report_currency = ACTIVE_ETF_CONFIG[selected_etf].get('report_currency') or "CNY"
                        with st.spinner("正在拼接并保存..."):
                            df_combined, scaling_factor, stitch_date, msg = stitch_with_tickflow(df_uploaded, etf_code, asset_currency=asset_currency, report_currency=report_currency)

                        st.info(msg)

                        if df_combined is not None:
                            save_target_to_db(
                                index_code,
                                selected_etf,
                                etf_code=etf_code,
                                scaling_factor=scaling_factor,
                                stitch_date=stitch_date,
                                data_source=(ACTIVE_ETF_CONFIG[selected_etf].get("data_source") or "ZZ"),
                                asset_currency=asset_currency,
                                report_currency=report_currency,
                            )
                            written_rows = save_prices_to_db(df_combined, index_code)
                            st.session_state["etf_config_runtime"][selected_etf]['scaling_factor'] = scaling_factor
                            st.cache_data.clear()
                            st.success(f"✅ {selected_etf} 已拼接并保存，共 {len(df_combined)} 条")
                            st.caption(f"🗄️ 本次落库 {written_rows} 条")

    elif mode == "新增标的：绑定并导入":
        st.subheader("➕ 新增标的（标的名称 + 指数代码/符号；ZZ 需 ETF 和历史文件，SZ 可选传或在线拉取）")
        st.caption("说明：`index_code` 是主键。`SZ` 走深证接口，`ZZ` 走 ETF 拼接，`YH` 走 Yahoo Finance。")

        col_name, col_code, col_index, col_source, col_currency = st.columns(5)
        with col_name:
            new_etf_name = st.text_input("标的名称", placeholder="例如：新etf指数")
        with col_code:
            new_etf_code = st.text_input("关联 ETF 代码（可选）", placeholder="例如：159999")
        with col_index:
            new_index_code = st.text_input("指数代码/符号", placeholder="SZ/ZZ 例如：CN2324；YH 例如：^GSPC")
        with col_source:
            new_data_source = st.selectbox("数据源", ["ZZ", "SZ", "YH"], index=0)
        with col_currency:
            new_asset_currency = st.selectbox("资产币种", ["CNY", "USD", "HKD", "JPY", "EUR"], index=0)

        new_history_file = None
        if new_data_source == "ZZ":
            new_history_file = st.file_uploader(
                "上传该标的的指数历史文件（xlsx/xls/csv）",
                type=['xlsx', 'xls', 'csv'],
                key="new_target_history_file_zz",
            )
        elif new_data_source == "SZ":
            st.info("对于 SZ 标的，可上传本地历史文件（推荐）；若不上传，则将从在线接口全量拉取。")
            new_history_file = st.file_uploader(
                "上传指数历史文件（可选）",
                type=['xlsx', 'xls', 'csv'],
                key="new_target_history_file_sz",
            )
        elif new_data_source == "YH":
            st.info("YH 模式会直接从 Yahoo Finance 拉取全量历史数据并保存，无需上传历史文件。")

        if st.button("➕ 绑定并导入", use_container_width=True, type="primary"):
            target_name = (new_etf_name or "").strip()
            target_etf_code = _normalize_etf_code(new_etf_code)
            target_index_code = _normalize_index_code((new_index_code or "").strip())
            target_data_source = _normalize_data_source(new_data_source) or "ZZ"
            target_asset_currency = _normalize_currency(new_asset_currency) or "CNY"
            target_report_currency = "CNY"

            # ZZ源必须有ETF和文件；其他源ETF可选；SZ源文件可选
            if target_data_source == "ZZ" and not target_etf_code:
                st.error("❌ ZZ 数据源必须提供 ETF 代码")
            elif target_data_source == "ZZ" and new_history_file is None:
                st.error("❌ ZZ 数据源必须提供指数历史文件")
            elif not target_name or not target_index_code:
                st.error("❌ 必须提供标的名称和指数代码")
            elif target_name in ACTIVE_ETF_CONFIG:
                st.error(f"❌ 标的 {target_name} 已存在，请更换名称")
            elif any(cfg.get("index_code") == target_index_code for cfg in ACTIVE_ETF_CONFIG.values()):
                st.error(f"❌ 指数代码 {target_index_code} 已存在，请直接使用已有标的")
            else:
                df_new, parse_msg = None, ""
                if target_data_source == "ZZ":
                    df_new, parse_msg = parse_upload_file(new_history_file)
                elif target_data_source == "SZ":
                    if new_history_file:
                        df_new, parse_msg = parse_upload_file(new_history_file)
                    else:
                        try:
                            df_hist = fetch_szse_index_daily(target_index_code, start_date="1991-01-01", end_date="2050-01-01")
                            df_hist = _exclude_today_rows(df_hist, date_col="Date")
                            df_new = df_hist.rename(columns={"Index_Close": "Close"})
                            parse_msg = f"✅ 已从深证接口拉取 {len(df_new)} 条数据"
                        except Exception as e:
                            df_new, parse_msg = None, f"❌ 深证接口拉取失败: {e}"
                elif target_data_source == "YH":
                    try:
                        df_hist = fetch_yahoo_history(target_index_code, start_date="1991-01-01")
                        df_hist = _exclude_today_rows(df_hist, date_col="Date")
                        df_new = df_hist.rename(columns={"Index_Close": "Close"})
                        parse_msg = f"✅ 已从 Yahoo Finance 拉取 {len(df_new)} 条数据"
                    except Exception as e:
                        df_new, parse_msg = None, f"❌ Yahoo Finance 拉取失败: {e}"

                if parse_msg:
                    st.info(parse_msg)

                if df_new is not None and not df_new.empty:
                    with st.spinner("正在保存指数历史数据..."):
                        # 直接保存指数历史，etf_close_raw/etf_close_hfq=NULL，等待更新全部数据时拼接
                        df_to_save = df_new[['Date', 'Close']].copy()
                        df_to_save['index_close']    = df_to_save['Close']
                        df_to_save['etf_close_raw']  = None
                        df_to_save['etf_close_hfq']  = None
                        converted = _apply_currency_conversion(
                            df_to_save[['Date', 'Close']].rename(columns={'Close': 'Index_Close'}),
                            asset_currency=target_asset_currency,
                            report_currency=target_report_currency,
                        )
                        df_to_save['asset_close_native'] = converted['asset_close_native']
                        df_to_save['fx_to_cny'] = converted['fx_to_cny']
                        df_to_save['close_cny'] = converted['close_cny']
                        df_to_save['combined_close'] = converted['combined_close']
                        
                        # 保存元数据和历史数据
                        target_saved = save_target_to_db(
                            target_index_code,
                            target_name,
                            etf_code=target_etf_code or None,
                            scaling_factor=1.0,
                            stitch_date=df_new['Date'].max().date(),
                            data_source=target_data_source,
                            asset_currency=target_asset_currency,
                            report_currency=target_report_currency,
                        )
                        if not target_saved:
                            st.error("标的元数据保存失败，已停止行情入库。")
                        else:
                            written_rows = save_prices_to_db(df_to_save[['Date', 'index_close', 'etf_close_raw', 'etf_close_hfq', 'asset_close_native', 'fx_to_cny', 'close_cny', 'combined_close']], target_index_code)

                            st.session_state["etf_config_runtime"][target_name] = {
                                "name": target_name,
                                "etf_code": target_etf_code,
                                "index_code": target_index_code,
                                "scaling_factor": 1.0,
                                "data_source": target_data_source,
                                "asset_currency": target_asset_currency,
                                "report_currency": target_report_currency,
                            }
                            st.cache_data.clear()
                            etf_hint = f" / ETF {target_etf_code}" if target_etf_code else ""
                            st.success(f"✅ 已新增：{target_name}（{target_data_source} / {target_index_code}{etf_hint}），指数历史已保存")
                            st.caption(f"🗄️ 本次落库 {written_rows} 条")
                            st.info("💡 点击「更新全部数据」后：SZ 走深证接口、ZZ 走 ETF 拟合、YH 走 Yahoo Finance。未绑定 ETF 时会先只保存指数历史。")
                            st.rerun()
                else:
                    st.error("❌ 未获取到可保存的历史数据，请检查指数代码/符号或数据源设置")

    else:
        st.subheader("🔗 已有标的：补绑 ETF")
        if not ACTIVE_ETF_CONFIG:
            st.info("暂无标的，请先新增标的。")
        else:
            bind_target = st.selectbox("选择标的", list(ACTIVE_ETF_CONFIG.keys()), key="bind_etf_target")
            current_etf = ACTIVE_ETF_CONFIG[bind_target].get("etf_code") or ""
            bind_etf_code = st.text_input("ETF 代码", value=current_etf, key="bind_etf_code")

            if st.button("🔗 保存绑定并回补数据", use_container_width=True, type="primary"):
                target_cfg = ACTIVE_ETF_CONFIG[bind_target]
                target_index_code = target_cfg["index_code"]
                normalized_etf = _normalize_etf_code(bind_etf_code)
                if not normalized_etf:
                    st.error("❌ 请输入 ETF 代码")
                else:
                    with st.spinner("正在保存 ETF 绑定并回补数据..."):
                        ok = save_target_to_db(
                            target_index_code,
                            bind_target,
                            etf_code=normalized_etf,
                            scaling_factor=target_cfg.get("scaling_factor"),
                            data_source=target_cfg.get("data_source") or "ZZ",
                            asset_currency=target_cfg.get("asset_currency") or "CNY",
                            report_currency=target_cfg.get("report_currency") or "CNY",
                        )
                        if ok:
                            df_latest, scaling_factor_latest, written_rows = sync_data_from_tickflow(target_index_code)
                            st.session_state["etf_config_runtime"][bind_target]["etf_code"] = normalized_etf
                            st.session_state["etf_config_runtime"][bind_target]["scaling_factor"] = scaling_factor_latest
                            st.cache_data.clear()
                            st.success(f"✅ 已为 {bind_target} 绑定 ETF {normalized_etf}，并回补 {written_rows} 条数据")
                            if df_latest is None:
                                st.warning("⚠️ 绑定已保存，但暂时还没有可展示的数据")
                            st.rerun()
                        else:
                            st.error("❌ ETF 绑定保存失败")
