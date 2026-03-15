import os
import socket
import psycopg2
from psycopg2.extras import execute_values
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import akshare as ak
import streamlit as st
from urllib.parse import urlparse, parse_qs, unquote

try:
    import baostock as bs
except Exception:
    bs = None

warnings.filterwarnings('ignore')

# ─── 页面配置 ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="今天买什么", page_icon="📈", layout="wide")

# ─── 全局配置 ────────────────────────────────────────────────────────────────
TRADITION_START = "20081031"
TRADITION_END   = "20221031"
ROLLING_WINDOW  = 1250
# Supabase 连接字符串（从 Streamlit secrets 读取）
DATABASE_URL_POOLER = st.secrets.get("database_url_pooler", None)
DATABASE_URL_DIRECT = st.secrets.get("database_url", None)
DATABASE_URL = DATABASE_URL_POOLER or DATABASE_URL_DIRECT

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
    """从 etf_targets 读取所有标的，返回 {name: {etf_code, name, scaling_factor}}"""
    conn = get_db_connection()
    if not conn:
        return {}
    try:
        df = pd.read_sql(
            "SELECT etf_code, name, scaling_factor FROM etf_targets ORDER BY name",
            conn,
        )
        if df.empty:
            return {}
        return {
            row['name']: {
                "name": row['name'],
                "etf_code": str(row['etf_code']),
                "scaling_factor": float(row['scaling_factor']) if pd.notna(row['scaling_factor']) else 1.0,
            }
            for _, row in df.iterrows()
        }
    except Exception as e:
        st.caption(f"⚠️ 读取标的配置失败: {e}")
        return {}
    finally:
        conn.close()


def save_target_to_db(etf_code, name, scaling_factor=None, stitch_date=None, index_code=None):
    """新增或更新 etf_targets 标的元数据"""
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO etf_targets (etf_code, name, index_code, scaling_factor, stitch_date)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (etf_code) DO UPDATE SET
                name           = EXCLUDED.name,
                index_code     = COALESCE(EXCLUDED.index_code,     etf_targets.index_code),
                scaling_factor = COALESCE(EXCLUDED.scaling_factor, etf_targets.scaling_factor),
                stitch_date    = COALESCE(EXCLUDED.stitch_date,    etf_targets.stitch_date)
        """, (etf_code, name, index_code, scaling_factor, stitch_date))
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        st.warning(f"保存标的元数据失败: {e}")
        return False
    finally:
        conn.close()


def save_prices_to_db(df, etf_code):
    """
    写入 etf_prices。df 需含 Date、combined_close，可含 index_close、etf_close。
    冲突时：index_close/etf_close 保留原值（COALESCE），combined_close 直接覆盖。
    """
    conn = get_db_connection()
    if not conn:
        st.warning("无法连接数据库，数据未保存")
        return
    try:
        cur = conn.cursor()

        # 统一列，避免逐行 SQL 循环造成长时间阻塞
        work = df.copy()
        if 'index_close' not in work.columns:
            work['index_close'] = None
        if 'etf_close' not in work.columns:
            work['etf_close'] = None

        work['Date'] = pd.to_datetime(work['Date'])
        work['index_close'] = pd.to_numeric(work['index_close'], errors='coerce')
        work['etf_close'] = pd.to_numeric(work['etf_close'], errors='coerce')
        work['combined_close'] = pd.to_numeric(work['combined_close'], errors='coerce')

        rows = []
        for _, row in work[['Date', 'index_close', 'etf_close', 'combined_close']].iterrows():
            rows.append((
                etf_code,
                row['Date'].date(),
                None if pd.isna(row['index_close']) else float(row['index_close']),
                None if pd.isna(row['etf_close']) else float(row['etf_close']),
                float(row['combined_close']),
            ))

        if not rows:
            cur.close()
            return

        sql = """
            INSERT INTO etf_prices (etf_code, date, index_close, etf_close, combined_close)
            VALUES %s
            ON CONFLICT (etf_code, date) DO UPDATE SET
                index_close    = COALESCE(EXCLUDED.index_close, etf_prices.index_close),
                etf_close      = COALESCE(EXCLUDED.etf_close,   etf_prices.etf_close),
                combined_close = EXCLUDED.combined_close
        """

        # 分批写入，兼顾性能和语句体积
        batch_size = 1000
        for i in range(0, len(rows), batch_size):
            execute_values(cur, sql, rows[i:i + batch_size])

        conn.commit()
        cur.close()
    except Exception as e:
        st.warning(f"保存行情数据异常: {e}")
    finally:
        conn.close()


def load_from_db(etf_code):
    """返回 (df[Date, Close], scaling_factor)，Close 对应 combined_close"""
    conn = get_db_connection()
    if not conn:
        return None, 1.0
    try:
        df = pd.read_sql(
            "SELECT date, combined_close FROM etf_prices WHERE etf_code=%s ORDER BY date",
            conn, params=(etf_code,),
        )
        sf_row = pd.read_sql(
            "SELECT scaling_factor FROM etf_targets WHERE etf_code=%s",
            conn, params=(etf_code,),
        )
        scaling_factor = 1.0
        if not sf_row.empty and pd.notna(sf_row['scaling_factor'].iloc[0]):
            scaling_factor = float(sf_row['scaling_factor'].iloc[0])
        if df.empty:
            return None, scaling_factor
        df = df.rename(columns={'date': 'Date', 'combined_close': 'Close'})
        df['Date']  = pd.to_datetime(df['Date'])
        df['Close'] = df['Close'].astype(float)
        return df[['Date', 'Close']].reset_index(drop=True), scaling_factor
    except Exception as e:
        st.warning(f"读取数据异常: {e}")
        return None, 1.0
    finally:
        conn.close()


# ─── AkShare ──────────────────────────────────────────────────────────────────
def fetch_all_from_akshare(etf_code):
    """拉取 ETF 后复权全量日线，返回 DataFrame(Date, ETF_Close)"""
    raw = ak.fund_etf_hist_em(symbol=etf_code, period="daily", adjust="hfq")
    raw['日期'] = pd.to_datetime(raw['日期'])
    df = raw[['日期', '收盘']].rename(columns={'日期': 'Date', '收盘': 'ETF_Close'})
    return df.sort_values('Date').reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_latest_ak_hfq_unadj(etf_code):
    """返回 AkShare 最新 (hfq, unadj) 收盘价，用于对齐显示口径。"""
    hfq_raw = ak.fund_etf_hist_em(symbol=etf_code, period="daily", adjust="hfq")
    unadj_raw = ak.fund_etf_hist_em(symbol=etf_code, period="daily", adjust="")
    hfq_latest = float(hfq_raw.sort_values('日期').iloc[-1]['收盘'])
    unadj_latest = float(unadj_raw.sort_values('日期').iloc[-1]['收盘'])
    return hfq_latest, unadj_latest


def _to_baostock_code(etf_code):
    s = str(etf_code).strip()
    if s.startswith(('5', '6', '9')):
        return f"sh.{s}"
    return f"sz.{s}"


@st.cache_data(ttl=3600, show_spinner=False)
def get_unadj_factor_from_baostock(etf_code):
    """
    使用 baostock query_adjust_factor 估算“后复权 -> 不复权”的换算系数。
    返回 (factor, source_msg)，其中 factor 满足: unadj ~= hfq * factor。
    """
    # 先拿 AkShare 比值做方向校准（避免因子方向理解差异）
    ak_ratio = None
    try:
        hfq_latest, unadj_latest = fetch_latest_ak_hfq_unadj(etf_code)
        if hfq_latest > 0:
            ak_ratio = unadj_latest / hfq_latest
    except Exception:
        ak_ratio = None

    if bs is None:
        if ak_ratio is not None:
            return float(ak_ratio), "akshare-fallback"
        return 1.0, "no-baostock"

    login_res = bs.login()
    if getattr(login_res, 'error_code', '1') != '0':
        if ak_ratio is not None:
            return float(ak_ratio), "akshare-fallback"
        return 1.0, "baostock-login-failed"

    try:
        bs_code = _to_baostock_code(etf_code)
        rs = bs.query_adjust_factor(
            code=bs_code,
            start_date="1990-01-01",
            end_date=pd.Timestamp.today().strftime('%Y-%m-%d'),
        )
        rows = []
        while rs.error_code == '0' and rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            if ak_ratio is not None:
                return float(ak_ratio), "akshare-fallback"
            return 1.0, "baostock-empty"

        df_factor = pd.DataFrame(rows, columns=rs.fields)
        if 'dividOperateDate' in df_factor.columns:
            df_factor['dividOperateDate'] = pd.to_datetime(df_factor['dividOperateDate'], errors='coerce')
            latest = df_factor.sort_values('dividOperateDate').iloc[-1]
        else:
            latest = df_factor.iloc[-1]

        raw = None
        for col in ('backAdjustFactor', 'adjustFactor', 'foreAdjustFactor'):
            if col in df_factor.columns and pd.notna(latest.get(col)):
                raw = pd.to_numeric(latest.get(col), errors='coerce')
                if pd.notna(raw):
                    raw = float(raw)
                    break

        if raw is None or raw <= 0:
            if ak_ratio is not None:
                return float(ak_ratio), "akshare-fallback"
            return 1.0, "baostock-invalid"

        cand_mul = raw
        cand_div = 1.0 / raw

        # 用 AkShare 当日比值选择更合理方向
        if ak_ratio is not None:
            if abs(cand_mul - ak_ratio) <= abs(cand_div - ak_ratio):
                return float(cand_mul), "baostock"
            return float(cand_div), "baostock"

        # 无法校准时优先使用 1/raw（常见情形：hfq = unadj * factor）
        return float(cand_div), "baostock"
    except Exception:
        if ak_ratio is not None:
            return float(ak_ratio), "akshare-fallback"
        return 1.0, "baostock-exception"
    finally:
        try:
            bs.logout()
        except Exception:
            pass


def _get_target_name(etf_code, default_name=None):
    """获取标的的真实 name，如果不存在则返回默认值"""
    conn = get_db_connection()
    if not conn:
        return default_name or etf_code
    try:
        res = pd.read_sql(
            "SELECT name FROM etf_targets WHERE etf_code=%s",
            conn, params=(etf_code,),
        )
        conn.close()
        if not res.empty and pd.notna(res.iloc[0]['name']):
            return res.iloc[0]['name']
        return default_name or etf_code
    except Exception:
        return default_name or etf_code


def _check_needs_full_stitch(etf_code):
    """检查标的是否需要完整拼接（历史数据 etf_close 全为 NULL 但 index_close 有值）"""
    conn = get_db_connection()
    if not conn:
        return False
    try:
        res = pd.read_sql(
            """SELECT 
                COUNT(CASE WHEN index_close IS NOT NULL THEN 1 END) AS has_index,
                COUNT(CASE WHEN etf_close IS NOT NULL THEN 1 END) AS has_etf
            FROM etf_prices WHERE etf_code=%s""",
            conn, params=(etf_code,),
        )
        conn.close()
        if res.empty:
            return False
        has_index = res.iloc[0]['has_index'] > 0
        has_etf = res.iloc[0]['has_etf'] == 0
        return has_index and has_etf  # 有指数数据但 etf_close 全为 NULL
    except Exception:
        return False


def _full_stitch_from_db(etf_code):
    """完整拼接：从 DB 读取 index_close 数据，拉取 AkShare 全量，计算 scaling_factor 并更新所有行"""
    try:
        conn = get_db_connection()
        if not conn:
            return False
        
        # 读取 DB 中已有的 index_close 数据
        df_hist = pd.read_sql(
            "SELECT date, index_close FROM etf_prices WHERE etf_code=%s AND index_close IS NOT NULL ORDER BY date",
            conn, params=(etf_code,),
        )
        conn.close()
        
        if df_hist.empty:
            return False
        
        df_hist['Date'] = pd.to_datetime(df_hist['date'])
        df_hist = df_hist.rename(columns={'index_close': 'index_close_val'})[['Date', 'index_close_val']]
        
        # 拉取 AkShare 完整历史
        ak_df = fetch_all_from_akshare(etf_code)
        
        # 查找重叠日期计算缩放比例
        merged = pd.merge(df_hist, ak_df, on='Date', how='inner')
        if merged.empty:
            return False
        
        anchor = merged.iloc[-1]
        scaling_factor = float(anchor['index_close_val']) / float(anchor['ETF_Close'])
        stitch_date = anchor['Date'].date()
        
        # 构造拼接结果：历史段 + 新增段
        hist_part = df_hist.copy()
        hist_part = pd.merge(hist_part, ak_df, on='Date', how='left')
        hist_part = hist_part.rename(columns={'index_close_val': 'index_close', 'ETF_Close': 'etf_close'})
        hist_part['combined_close'] = hist_part['index_close']
        
        last_hist_date = df_hist['Date'].max()
        new_part = ak_df[ak_df['Date'] > last_hist_date].copy()
        new_part = new_part.rename(columns={'ETF_Close': 'etf_close'})
        new_part['index_close'] = None
        new_part['combined_close'] = new_part['etf_close'] * scaling_factor
        
        result = pd.concat(
            [hist_part[['Date', 'index_close', 'etf_close', 'combined_close']],
             new_part[['Date', 'index_close', 'etf_close', 'combined_close']]],
            ignore_index=True,
        ).sort_values('Date').reset_index(drop=True)
        
        # 更新 DB
        save_prices_to_db(result, etf_code)
        target_name = _get_target_name(etf_code)
        save_target_to_db(etf_code, target_name, scaling_factor=scaling_factor, stitch_date=stitch_date)
        
        return True
    except Exception as e:
        return False


def _incremental_akshare_update(etf_code, scaling_factor):
    """增量拉取 AkShare 新数据追加到 etf_prices（新行：index_close=NULL）"""
    conn = get_db_connection()
    if not conn:
        return
    try:
        res = pd.read_sql(
            "SELECT MAX(date) AS last_date FROM etf_prices WHERE etf_code=%s",
            conn, params=(etf_code,),
        )
        last_date = pd.to_datetime(res['last_date'].iloc[0]) if not res.empty and pd.notna(res['last_date'].iloc[0]) else None
    except Exception:
        last_date = None
    finally:
        conn.close()

    new_all  = fetch_all_from_akshare(etf_code)
    new_data = new_all[new_all['Date'] > last_date].copy() if last_date is not None else new_all.copy()
    if new_data.empty:
        return
    new_data = new_data.rename(columns={'ETF_Close': 'etf_close'})
    new_data['index_close']    = None
    new_data['combined_close'] = new_data['etf_close'] * scaling_factor
    save_prices_to_db(new_data[['Date', 'index_close', 'etf_close', 'combined_close']], etf_code)


def sync_data_from_akshare(etf_code: str):
    """
    仅在手动点击「更新全部数据」时调用：
    DB 有数据则执行完整拼接检查/增量更新；DB 无数据则全量初始化。
    """
    df, scaling_factor = load_from_db(etf_code)
    if df is None:
        raw = fetch_all_from_akshare(etf_code)
        target_name = _get_target_name(etf_code)
        save_target_to_db(etf_code, target_name, scaling_factor=1.0)
        rows = pd.DataFrame({
            'Date':           raw['Date'],
            'index_close':    None,
            'etf_close':      raw['ETF_Close'],
            'combined_close': raw['ETF_Close'],
        })
        save_prices_to_db(rows, etf_code)
    else:
        # 检查是否需要完整拼接（历史数据未初始化 etf_close）
        if _check_needs_full_stitch(etf_code):
            _full_stitch_from_db(etf_code)
        else:
            _incremental_akshare_update(etf_code, scaling_factor)

    return load_from_db(etf_code)


@st.cache_data(ttl=3600, show_spinner=False)
def get_data(etf_code: str):
    """
    页面加载只读数据库，不触发任何 AkShare 网络请求。
    实时更新请点击侧边栏「更新全部数据」。
    """
    return load_from_db(etf_code)


# ─── 核心分析 ─────────────────────────────────────────────────────────────────
def compute_and_plot(df, etf_name, deviation_pct, scaling_factor=1.0, unadj_factor=1.0):
    df = df.copy()
    df['Log_Close'] = np.log(df['Close'])
    df['Time_Idx']  = np.arange(len(df))

    # 传统回归
    mask = (df['Date'] >= pd.to_datetime(TRADITION_START)) & \
           (df['Date'] <= pd.to_datetime(TRADITION_END))
    sample_df = df[mask]
    if len(sample_df) < 100:
        raise ValueError(f"传统回归样本不足（{len(sample_df)} 条），请检查数据起止日期")
    k_trad, b_trad = np.polyfit(sample_df['Time_Idx'], sample_df['Log_Close'], 1)
    df['Trad_Pred_Log']   = k_trad * df['Time_Idx'] + b_trad
    df['Trad_Pred_Price'] = np.exp(df['Trad_Pred_Log'])
    resids_trad = sample_df['Log_Close'] - (k_trad * sample_df['Time_Idx'] + b_trad)
    std_trad    = np.std(resids_trad)
    df['Trad_Z_Score'] = (df['Log_Close'] - df['Trad_Pred_Log']) / std_trad
    z_plus  = np.log(1 + deviation_pct / 100.0) / std_trad
    z_minus = np.log(1 - deviation_pct / 100.0) / std_trad

    # 滚动回归
    rolling_preds = np.full(len(df), np.nan)
    rolling_z     = np.full(len(df), np.nan)
    k_roll_last   = np.nan
    for i in range(ROLLING_WINDOW, len(df)):
        ys = df['Log_Close'].values[i - ROLLING_WINDOW:i]
        xs = np.arange(ROLLING_WINDOW)
        k_r, b_r = np.polyfit(xs, ys, 1)
        pred = k_r * (ROLLING_WINDOW - 1) + b_r
        rolling_preds[i] = pred
        std_r = np.std(ys - (k_r * xs + b_r))
        if std_r > 0:
            rolling_z[i] = (ys[-1] - pred) / std_r
        k_roll_last = k_r
    df['Roll_Pred_Price'] = np.exp(rolling_preds)
    df['Roll_Z_Score']    = rolling_z

    # MA250 年均线
    df['MA250_Price'] = df['Close'].rolling(window=250, min_periods=250).mean()
    ma_log_diff = np.full(len(df), np.nan)
    ma_mask = df['MA250_Price'].notna() & (df['MA250_Price'] > 0)
    ma_log_diff[ma_mask.values] = (
        df.loc[ma_mask, 'Log_Close'].values - np.log(df.loc[ma_mask, 'MA250_Price'].values)
    )
    std_ma250 = np.nanstd(ma_log_diff)
    if pd.notna(std_ma250) and std_ma250 > 0:
        df['MA250_Z_Score'] = ma_log_diff / std_ma250
    else:
        df['MA250_Z_Score'] = np.nan

    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                   gridspec_kw={'height_ratios': [2.5, 1]})
    ax1.plot(df['Date'], df['Close'], color='black', linewidth=1.2, label='指数')
    ax1.plot(df['Date'], df['Trad_Pred_Price'], color='red', linestyle='--',
             linewidth=2, label=f'传统回归({TRADITION_START[:4]}-{TRADITION_END[:4]})')
    ax1.fill_between(df['Date'],
                     np.exp(df['Trad_Pred_Log'] - 2 * std_trad),
                     np.exp(df['Trad_Pred_Log'] + 2 * std_trad),
                     color='red', alpha=0.1, label='传统通道(±2σ)')
    ax1.plot(df['Date'], df['Roll_Pred_Price'], color='blue', linestyle='-.',
             linewidth=1.5, label=f'滚动回归({ROLLING_WINDOW}日)')
    ax1.plot(df['Date'], df['MA250_Price'], color='#2F9E44', linestyle='-',
             linewidth=1.5, label='年均线(MA250)')
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
    ax2.plot(df['Date'], df['MA250_Z_Score'], color='#2F9E44', linestyle='-',
             alpha=0.9, linewidth=1.2, label='MA250-Z', zorder=2)
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
    ma250_pred   = float(df['MA250_Price'].iloc[-1]) if pd.notna(df['MA250_Price'].iloc[-1]) else np.nan

    if pd.notna(ma250_pred) and ma250_pred > 0:
        dev_ma250 = (latest_close / ma250_pred - 1) * 100
    else:
        dev_ma250 = np.nan

    ma250_prev = df['MA250_Price'].shift(250).iloc[-1] if len(df) > 250 else np.nan
    if pd.notna(ma250_pred) and pd.notna(ma250_prev) and ma250_prev > 0:
        cagr_ma250 = ((ma250_pred / ma250_prev) ** (252 / 250) - 1) * 100
    else:
        cagr_ma250 = np.nan

    # 展示口径：指数点位 -> 后复权ETF -> 不复权ETF
    def to_unadj_etf(point_value):
        if scaling_factor > 0:
            return point_value / scaling_factor * unadj_factor
        return point_value
    
    return fig, {
        "latest_date":  df['Date'].iloc[-1].strftime('%Y-%m-%d'),
        "latest_close": latest_close,
        "latest_etf_price": to_unadj_etf(latest_close),
        "trad_pred":    trad_pred,
        "trad_pred_etf": to_unadj_etf(trad_pred),
        "roll_pred":    roll_pred,
        "roll_pred_etf": to_unadj_etf(roll_pred),
        "ma250_pred":   ma250_pred,
        "ma250_pred_etf": to_unadj_etf(ma250_pred) if pd.notna(ma250_pred) else np.nan,
        "dev_trad":     (latest_close / trad_pred - 1) * 100,
        "dev_roll":     (latest_close / roll_pred - 1) * 100,
        "dev_ma250":    dev_ma250,
        "cagr_trad":    (np.exp(k_trad * 252) - 1) * 100,
        "cagr_roll":    (np.exp(k_roll_last * 252) - 1) * 100,
        "cagr_ma250":   cagr_ma250,
        "scaling_factor": scaling_factor,
        "unadj_factor": unadj_factor,
        "z_plus": z_plus,
        "z_minus": z_minus,
        "std_trad": std_trad,
        "plot_df": df[[
            'Date', 'Close', 'Trad_Pred_Price', 'Roll_Pred_Price', 'MA250_Price',
            'Trad_Z_Score', 'Roll_Z_Score', 'MA250_Z_Score', 'Trad_Pred_Log',
        ]].copy(),
    }


def render_native_charts(res, etf_name, deviation_pct):
    """使用 Plotly 渲染，现代简洁风格，含置信带、对数坐标、悬停。"""
    df       = res['plot_df'].copy()
    z_plus   = float(res['z_plus'])
    z_minus  = float(res['z_minus'])
    std_trad = float(res['std_trad'])

    # 传统回归置信带（对数空间 ±2σ）
    band_upper = np.exp(df['Trad_Pred_Log'] + 2 * std_trad)
    band_lower = np.exp(df['Trad_Pred_Log'] - 2 * std_trad)

    # ── 配色（海洋清风）────────────────────────────────────────────
    C_INDEX  = '#51999F'   # 海蓝绿 - 指数主线
    C_TRAD   = '#D97745'   # 暖橙棕 - 传统回归
    C_ROLL   = '#2D8CFF'   # 明亮蓝 - 滚动回归
    C_MA250  = '#2F9E44'   # 绿色 - MA250
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
        name=f'传统回归 ({TRADITION_START[:4]}–{TRADITION_END[:4]})',
        line=dict(color=C_TRAD, width=1.35, dash='solid'),
        hovertemplate='%{x|%Y-%m-%d}  传统: %{y:,.1f}<extra></extra>',
    ), row=1, col=1)

    # ── 滚动回归线（点划线）──────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Roll_Pred_Price'],
        name=f'滚动回归 ({ROLLING_WINDOW}日)',
        line=dict(color=C_ROLL, width=1.35, dash='dashdot'),
        hovertemplate='%{x|%Y-%m-%d}  滚动: %{y:,.1f}<extra></extra>',
    ), row=1, col=1)

    # ── MA250 年均线 ───────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['MA250_Price'],
        name='年均线 (MA250)',
        line=dict(color=C_MA250, width=1.35, dash='solid'),
        hovertemplate='%{x|%Y-%m-%d}  MA250: %{y:,.1f}<extra></extra>',
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
        x=df['Date'], y=df['MA250_Z_Score'],
        name='MA250-Z', line=dict(color=C_MA250, width=1.2),
        hovertemplate='%{x|%Y-%m-%d}  MA250-Z: %{y:.3f}<extra></extra>',
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
def build_comparison(deviation_pct, etf_config):
    rows = []
    for name, cfg in etf_config.items():
        try:
            df, scaling_factor = get_data(cfg['etf_code'])
        except Exception as e:
            rows.append({"标的": name, "ETF代码": cfg['etf_code'],
                         "最新日期": f"加载失败: {e}",
                         "传统偏离度(%)": None, "滚动偏离度(%)": None, "MA250偏离度(%)": None,
                         "传统CAGR(%)": None, "滚动CAGR(%)": None, "MA250年化(%)": None})
            continue
        if df is None or len(df) < ROLLING_WINDOW + 10:
            rows.append({"标的": name, "ETF代码": cfg['etf_code'],
                         "最新日期": "无数据（请先拼接入库）",
                         "传统偏离度(%)": None, "滚动偏离度(%)": None, "MA250偏离度(%)": None,
                         "传统CAGR(%)": None, "滚动CAGR(%)": None, "MA250年化(%)": None})
            continue
        try:
            fig, res = compute_and_plot(df, name, deviation_pct, scaling_factor)
            plt.close(fig)
            rows.append({
                "标的": name, "ETF代码": cfg['etf_code'],
                "最新日期":    res['latest_date'],
                "传统偏离度(%)": round(res['dev_trad'], 2),
                "滚动偏离度(%)": round(res['dev_roll'], 2),
                "MA250偏离度(%)": round(res['dev_ma250'], 2) if pd.notna(res['dev_ma250']) else None,
                "传统CAGR(%)":   round(res['cagr_trad'], 2),
                "滚动CAGR(%)":   round(res['cagr_roll'], 2),
                "MA250年化(%)":   round(res['cagr_ma250'], 2) if pd.notna(res['cagr_ma250']) else None,
            })
        except Exception as e:
            rows.append({"标的": name, "ETF代码": cfg['etf_code'],
                         "最新日期": f"出错: {e}",
                         "传统偏离度(%)": None, "滚动偏离度(%)": None, "MA250偏离度(%)": None,
                         "传统CAGR(%)": None, "滚动CAGR(%)": None, "MA250年化(%)": None})
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


def stitch_with_akshare(history_df, etf_code):
    """
    将历史指数数据（history_df: Date, Close=指数点位）与 AkShare ETF 数据拼接。
    返回 (structured_df, scaling_factor, stitch_date, message)
    structured_df 列：Date, index_close, etf_close, combined_close
    
    注意：本函数用于「已有标的：上传并拼接」流程，需要自动从AkShare拉取数据。
    """
    try:
        ak_df = fetch_all_from_akshare(etf_code)

        # 外连接历史和AkShare
        merged = pd.merge(
            history_df[['Date', 'Close']].rename(columns={'Close': 'index_close'}),
            ak_df.rename(columns={'ETF_Close': 'etf_close'}),
            on='Date', how='outer',
        ).sort_values('Date')

        # 找最后一个两列都有值的日期（锚点）来计算缩放比例，找不到就用1.0
        overlap = merged[merged['index_close'].notna() & merged['etf_close'].notna()]
        if not overlap.empty:
            anchor         = overlap.iloc[-1]
            scaling_factor = float(anchor['index_close']) / float(anchor['etf_close'])
            stitch_date    = anchor['Date'].date()
            msg_prefix = f"✅ 拼接成功，缩放比例: {scaling_factor:.4f}"
        else:
            scaling_factor = 1.0
            stitch_date    = history_df['Date'].max().date()
            msg_prefix = "⚠️ 历史数据与 AkShare 无重叠日期，缩放比例暂设为 1.0"

        # 历史段：所有历史日期（index_close有值）
        hist = history_df[['Date', 'Close']].rename(columns={'Close': 'index_close'}).copy()
        hist = pd.merge(hist, ak_df.rename(columns={'ETF_Close': 'etf_close'}), on='Date', how='left')
        hist['combined_close'] = hist['index_close']

        # 近期段：AkShare 独有日期（晚于历史最新日期）
        last_hist = history_df['Date'].max()
        recent = ak_df[ak_df['Date'] > last_hist].copy()
        recent = recent.rename(columns={'ETF_Close': 'etf_close'})
        recent['index_close']    = None
        recent['combined_close'] = recent['etf_close'] * scaling_factor

        result = pd.concat(
            [hist[['Date', 'index_close', 'etf_close', 'combined_close']],
             recent[['Date', 'index_close', 'etf_close', 'combined_close']]],
            ignore_index=True,
        ).sort_values('Date').reset_index(drop=True)

        return result, scaling_factor, stitch_date, \
               msg_prefix + f"，新增近期 {len(recent)} 条"
    except Exception as e:
        return None, 1.0, None, f"❌ 拼接失败: {e}"


# ─── Streamlit UI ─────────────────────────────────────────────────────────────
st.title("📈 今天买什么")

if "etf_config_runtime" not in st.session_state:
    st.session_state["etf_config_runtime"] = load_targets_from_db()

ACTIVE_ETF_CONFIG = st.session_state["etf_config_runtime"]
deviation_pct = 15
selected = None

with st.sidebar:
    st.header("⚙️ 参数设置")
    if st.button("🔄 从数据库同步标的", use_container_width=True):
        st.session_state["etf_config_runtime"] = load_targets_from_db()
        st.rerun()

    ACTIVE_ETF_CONFIG = st.session_state["etf_config_runtime"]

    if not ACTIVE_ETF_CONFIG:
        st.info("数据库暂无标的，请在「数据管理」中新增。")
    else:
        selected = st.selectbox("选择标的", list(ACTIVE_ETF_CONFIG.keys()))
        deviation_pct = st.slider("偏离阈值 (%)", 5, 30, 15, 1)

    st.divider()
    if st.button("🔄 更新全部数据", use_container_width=True, type="primary"):
        st.cache_data.clear()
        prog = st.progress(0)
        etf_list = list(ACTIVE_ETF_CONFIG.items())
        for idx, (name, cfg) in enumerate(etf_list):
            with st.spinner(f"拉取 {name}..."):
                try:
                    sync_data_from_akshare(cfg['etf_code'])
                except Exception as e:
                    st.warning(f"{name} 失败: {e}")
            prog.progress((idx + 1) / len(etf_list))
        st.success("✅ 全部数据已更新！")
        st.rerun()

    st.divider()
    if DATABASE_URL:
        try:
            conn = get_db_connection()
            if conn:
                summary = pd.read_sql(
                    "SELECT p.etf_code, t.name, COUNT(*) AS 条数, MAX(p.date) AS 最新日期 "
                    "FROM etf_prices p LEFT JOIN etf_targets t ON p.etf_code = t.etf_code "
                    "GROUP BY p.etf_code, t.name ORDER BY p.etf_code",
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
        etf_code = cfg['etf_code']
        etf_name = cfg['name']

        with st.spinner(f"加载 {etf_name} ({etf_code}) 数据..."):
            try:
                df, scaling_factor = get_data(etf_code)
            except Exception as e:
                st.error(f"❌ 数据加载失败：{e}")
                st.stop()

        if df is None or len(df) < ROLLING_WINDOW + 10:
            st.error("数据不足，无法计算回归，请先在「数据管理」中上传历史指数文件并拼接。")
        else:
            try:
                unadj_factor, factor_source = get_unadj_factor_from_baostock(etf_code)
                fig, res = compute_and_plot(df, etf_name, deviation_pct, scaling_factor, unadj_factor)
                render_native_charts(res, etf_name, deviation_pct)
                plt.close(fig)

                if factor_source != "baostock":
                    st.caption(f"ETF价格展示口径：不复权（因子来源: {factor_source}）")
                else:
                    st.caption("ETF价格展示口径：不复权（因子来源: baostock query_adjust_factor）")

                st.divider()
                st.subheader("指数点位 & ETF价格")
                c1, c2, c3 = st.columns(3)
                c1.metric("最新日期", res['latest_date'])
                c2.metric("指数点位", f"{res['latest_close']:,.0f}")
                c3.metric("ETF价格", f"{res['latest_etf_price']:.4f}")

                st.subheader("传统回归")
                c4, c5, c6, c7 = st.columns(4)
                c4.metric("指数点位", f"{res['trad_pred']:,.0f}")
                c5.metric("预估价格", f"{res['trad_pred_etf']:.4f}")
                c6.metric("偏离度", f"{res['dev_trad']:+.2f}%", delta_color="inverse")
                c7.metric("年化", f"{res['cagr_trad']:.2f}%")

                st.subheader("滚动回归")
                c8, c9, c10, c11 = st.columns(4)
                c8.metric("指数点位", f"{res['roll_pred']:,.0f}")
                c9.metric("预估价格", f"{res['roll_pred_etf']:.4f}")
                c10.metric("偏离度", f"{res['dev_roll']:+.2f}%", delta_color="inverse")
                c11.metric("年化", f"{res['cagr_roll']:.2f}%")

                st.subheader("年均线 MA250")
                c12, c13, c14, c15 = st.columns(4)
                c12.metric("指数点位", f"{res['ma250_pred']:,.0f}" if pd.notna(res['ma250_pred']) else "—")
                c13.metric("预估价格", f"{res['ma250_pred_etf']:.4f}" if pd.notna(res['ma250_pred_etf']) else "—")
                c14.metric("偏离度", f"{res['dev_ma250']:+.2f}%" if pd.notna(res['dev_ma250']) else "—", delta_color="inverse")
                c15.metric("年化", f"{res['cagr_ma250']:.2f}%" if pd.notna(res['cagr_ma250']) else "—")
            except Exception as e:
                st.error(f"计算出错：{e}")

with tab2:
    if not ACTIVE_ETF_CONFIG:
        st.info("暂无标的数据。")
    else:
        st.caption("对比数据来自数据库，更新请点击侧边栏「更新全部数据」")
        with st.spinner("计算全市场偏离度..."):
            compare_df = build_comparison(deviation_pct, ACTIVE_ETF_CONFIG)

        if compare_df.empty:
            st.info("无数据，请先拼接入库。")
        else:
            numeric_cols = ["传统偏离度(%)", "滚动偏离度(%)", "MA250偏离度(%)"]
            styled = compare_df.style.background_gradient(
                subset=[c for c in numeric_cols if c in compare_df.columns],
                cmap="coolwarm", vmin=-100, vmax=100,
            ).format({
                "传统偏离度(%)": lambda x: f"{x:+.2f}" if pd.notna(x) else "—",
                "滚动偏离度(%)": lambda x: f"{x:+.2f}" if pd.notna(x) else "—",
                "MA250偏离度(%)": lambda x: f"{x:+.2f}" if pd.notna(x) else "—",
                "传统CAGR(%)": lambda x: f"{x:.2f}" if pd.notna(x) else "—",
                "滚动CAGR(%)": lambda x: f"{x:.2f}" if pd.notna(x) else "—",
                "MA250年化(%)": lambda x: f"{x:.2f}" if pd.notna(x) else "—",
            })
            st.dataframe(styled, use_container_width=True, hide_index=True)

            plot_df = compare_df.dropna(subset=["传统偏离度(%)", "滚动偏离度(%)", "MA250偏离度(%)"])
            if not plot_df.empty:
                st.subheader("偏离度对比")
                plot_df = plot_df.copy()
                plot_df = plot_df.sort_values("传统偏离度(%)", ascending=True)

                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    y=plot_df["标的"],
                    x=plot_df["传统偏离度(%)"],
                    name="传统偏离度",
                    orientation='h',
                    marker=dict(color="#BFDFD2"),
                    opacity=0.88,
                    hovertemplate="标的: %{y}<br>传统偏离度: %{x:+.2f}%<extra></extra>",
                ))
                fig2.add_trace(go.Bar(
                    y=plot_df["标的"],
                    x=plot_df["滚动偏离度(%)"],
                    name="滚动偏离度",
                    orientation='h',
                    marker=dict(color="#7BC0CD"),
                    opacity=0.88,
                    hovertemplate="标的: %{y}<br>滚动偏离度: %{x:+.2f}%<extra></extra>",
                ))
                fig2.add_trace(go.Bar(
                    y=plot_df["标的"],
                    x=plot_df["MA250偏离度(%)"],
                    name="MA250偏离度",
                    orientation='h',
                    marker=dict(color="#2F9E44"),
                    opacity=0.88,
                    hovertemplate="标的: %{y}<br>MA250偏离度: %{x:+.2f}%<extra></extra>",
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
                    xaxis=dict(
                        title="偏离度 (%)",
                        zeroline=False,
                        gridcolor="rgba(200,200,200,0.35)",
                    ),
                    yaxis=dict(title=""),
                )

                st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("📤 数据管理")

    mode = st.radio(
        "选择操作模式",
        ["已有标的：上传并拼接", "新增标的：绑定并导入"],
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
                st.write("**第2步：选择关联 ETF，一键拼接入库**")
                selected_etf = st.selectbox("选择标的", list(ACTIVE_ETF_CONFIG.keys()), key="stitch_etf")

                if st.button("🔗 拼接并保存到数据库", use_container_width=True, type="primary"):
                    if df_uploaded is None:
                        st.error("❌ 请先上传并解析历史文件")
                    else:
                        etf_code = ACTIVE_ETF_CONFIG[selected_etf]['etf_code']
                        with st.spinner("正在拼接并保存..."):
                            df_combined, scaling_factor, stitch_date, msg = stitch_with_akshare(df_uploaded, etf_code)
                        st.info(msg)

                        if df_combined is not None:
                            save_target_to_db(etf_code, selected_etf, scaling_factor=scaling_factor, stitch_date=stitch_date)
                            save_prices_to_db(df_combined, etf_code)
                            st.session_state["etf_config_runtime"][selected_etf]['scaling_factor'] = scaling_factor
                            st.cache_data.clear()
                            st.success(f"✅ {selected_etf} 已拼接并保存，共 {len(df_combined)} 条")

    else:
        st.subheader("➕ 新增标的（标的名称 + ETF代码 + 指数历史文件，三者必填）")
        st.caption("说明：新增时只保存指数历史数据；点击「更新全部数据」时会自动从 AkShare 拉取并拼接。")

        col_name, col_code = st.columns(2)
        with col_name:
            new_etf_name = st.text_input("标的名称", placeholder="例如：新etf指数")
        with col_code:
            new_etf_code = st.text_input("关联 ETF 代码", placeholder="例如：159999")

        new_history_file = st.file_uploader(
            "上传该标的的指数历史文件（xlsx/xls/csv）",
            type=['xlsx', 'xls', 'csv'],
            key="new_target_history_file",
        )

        if st.button("➕ 绑定并导入", use_container_width=True, type="primary"):
            target_name = (new_etf_name or "").strip()
            target_code = (new_etf_code or "").strip()

            if not target_name or not target_code or new_history_file is None:
                st.error("❌ 必须同时提供：标的名称、ETF代码、指数历史文件")
            elif target_name in ACTIVE_ETF_CONFIG:
                st.error(f"❌ 标的 {target_name} 已存在，请更换名称")
            else:
                df_new, parse_msg = parse_upload_file(new_history_file)
                st.info(parse_msg)

                if df_new is not None:
                    with st.spinner("正在保存指数历史数据..."):
                        # 直接保存指数历史，etf_close=NULL，等待更新全部数据时拼接
                        df_to_save = df_new[['Date', 'Close']].copy()
                        df_to_save['index_close']    = df_to_save['Close']
                        df_to_save['etf_close']      = None
                        df_to_save['combined_close'] = df_to_save['Close']
                        
                        # 保存元数据和历史数据
                        save_target_to_db(target_code, target_name, scaling_factor=1.0, stitch_date=df_new['Date'].max().date())
                        save_prices_to_db(df_to_save[['Date', 'index_close', 'etf_close', 'combined_close']], target_code)
                        
                        st.session_state["etf_config_runtime"][target_name] = {
                            "name": target_name,
                            "etf_code": target_code,
                            "scaling_factor": 1.0,
                        }
                        st.cache_data.clear()
                        st.success(f"✅ 已新增：{target_name} ↔ {target_code}，指数历史已保存")
                        st.info("💡 点击「更新全部数据」会自动从 AkShare 拉取并拼接，计算正确的缩放比例")
                        st.rerun()
