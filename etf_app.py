import os
import socket
import psycopg2
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import akshare as ak
import streamlit as st
from urllib.parse import urlparse, parse_qs, unquote

warnings.filterwarnings('ignore')

# ─── 页面配置 ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ETF估值", page_icon="📈", layout="wide")

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
        for _, row in df.iterrows():
            ic = float(row['index_close']) if 'index_close' in df.columns and pd.notna(row.get('index_close')) else None
            ec = float(row['etf_close'])   if 'etf_close'   in df.columns and pd.notna(row.get('etf_close'))   else None
            cc = float(row['combined_close'])
            cur.execute("""
                INSERT INTO etf_prices (etf_code, date, index_close, etf_close, combined_close)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (etf_code, date) DO UPDATE SET
                    index_close    = COALESCE(EXCLUDED.index_close, etf_prices.index_close),
                    etf_close      = COALESCE(EXCLUDED.etf_close,   etf_prices.etf_close),
                    combined_close = EXCLUDED.combined_close
            """, (etf_code, row['Date'].date(), ic, ec, cc))
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


@st.cache_data(ttl=3600, show_spinner=False)
def get_data(etf_code: str):
    """
    DB 优先 → 增量 AkShare 更新 → 返回 (df[Date, Close], scaling_factor)
    若 DB 无数据，全量拉 AkShare（scaling_factor=1.0，需后续上传指数文件重新拼接）。
    """
    df, scaling_factor = load_from_db(etf_code)
    if df is None:
        raw = fetch_all_from_akshare(etf_code)
        save_target_to_db(etf_code, etf_code, scaling_factor=1.0)
        rows = pd.DataFrame({
            'Date':           raw['Date'],
            'index_close':    None,
            'etf_close':      raw['ETF_Close'],
            'combined_close': raw['ETF_Close'],
        })
        save_prices_to_db(rows, etf_code)
        df = raw[['Date']].copy()
        df['Close'] = raw['ETF_Close'].values
        scaling_factor = 1.0
    else:
        _incremental_akshare_update(etf_code, scaling_factor)
        df, scaling_factor = load_from_db(etf_code)
    return df, scaling_factor


# ─── 核心分析 ─────────────────────────────────────────────────────────────────
def compute_and_plot(df, etf_name, deviation_pct, scaling_factor=1.0):
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
    
    return fig, {
        "latest_date":  df['Date'].iloc[-1].strftime('%Y-%m-%d'),
        "latest_close": latest_close,
        "latest_etf_price": latest_close / scaling_factor if scaling_factor > 0 else latest_close,
        "trad_pred":    trad_pred,
        "trad_pred_etf": trad_pred / scaling_factor if scaling_factor > 0 else trad_pred,
        "roll_pred":    roll_pred,
        "roll_pred_etf": roll_pred / scaling_factor if scaling_factor > 0 else roll_pred,
        "dev_trad":     (latest_close / trad_pred - 1) * 100,
        "dev_roll":     (latest_close / roll_pred - 1) * 100,
        "cagr_trad":    (np.exp(k_trad * 252) - 1) * 100,
        "cagr_roll":    (np.exp(k_roll_last * 252) - 1) * 100,
        "scaling_factor": scaling_factor,
    }


# ─── 全市场对比 ───────────────────────────────────────────────────────────────
def build_comparison(deviation_pct, etf_config):
    rows = []
    for name, cfg in etf_config.items():
        try:
            df, scaling_factor = get_data(cfg['etf_code'])
        except Exception as e:
            rows.append({"标的": name, "ETF代码": cfg['etf_code'],
                         "最新日期": f"加载失败: {e}",
                         "传统偏离度(%)": None, "滚动偏离度(%)": None,
                         "传统CAGR(%)": None, "滚动CAGR(%)": None})
            continue
        if df is None or len(df) < ROLLING_WINDOW + 10:
            rows.append({"标的": name, "ETF代码": cfg['etf_code'],
                         "最新日期": "无数据（请先拼接入库）",
                         "传统偏离度(%)": None, "滚动偏离度(%)": None,
                         "传统CAGR(%)": None, "滚动CAGR(%)": None})
            continue
        try:
            fig, res = compute_and_plot(df, name, deviation_pct, scaling_factor)
            plt.close(fig)
            rows.append({
                "标的": name, "ETF代码": cfg['etf_code'],
                "最新日期":    res['latest_date'],
                "传统偏离度(%)": round(res['dev_trad'], 2),
                "滚动偏离度(%)": round(res['dev_roll'], 2),
                "传统CAGR(%)":   round(res['cagr_trad'], 2),
                "滚动CAGR(%)":   round(res['cagr_roll'], 2),
            })
        except Exception as e:
            rows.append({"标的": name, "ETF代码": cfg['etf_code'],
                         "最新日期": f"出错: {e}",
                         "传统偏离度(%)": None, "滚动偏离度(%)": None,
                         "传统CAGR(%)": None, "滚动CAGR(%)": None})
    return pd.DataFrame(rows)


# ─── 数据管理工具 ─────────────────────────────────────────────────────────────
def parse_upload_file(uploaded_file):
    """解析 Excel/CSV 文件，返回 (DataFrame, message)"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # 智能检测日期和收盘列
        date_col = None
        close_col = None
        for col in df.columns:
            col_lower = col.lower()
            if '日期' in col_lower or 'date' in col_lower or col_lower == '时间':
                date_col = col
            if '收盘' in col_lower or 'close' in col_lower or '点位' in col_lower:
                close_col = col
        
        if date_col is None or close_col is None:
            return None, f"❌ 无法识别日期列或收盘列。找到的列: {list(df.columns)}"
        
        df = df[[date_col, close_col]].rename(columns={date_col: 'Date', close_col: 'Close'})

        # 兼容多种日期格式，避免数字日期被错误解析成 1970 年。
        raw_date = df['Date']
        parsed_date = pd.to_datetime(raw_date, errors='coerce')

        numeric_raw = pd.to_numeric(raw_date, errors='coerce')
        need_fix = parsed_date.isna() & numeric_raw.notna()

        if need_fix.any():
            n = numeric_raw[need_fix]
            fixed = pd.Series(pd.NaT, index=n.index, dtype='datetime64[ns]')

            # Excel 序列日期（天数）
            excel_mask = (n >= 20000) & (n <= 80000)
            if excel_mask.any():
                fixed.loc[excel_mask] = pd.to_datetime(
                    n.loc[excel_mask], unit='D', origin='1899-12-30', errors='coerce'
                )

            # 形如 20240311 的 YYYYMMDD 数字日期
            ymd_mask = (n >= 19000101) & (n <= 21001231)
            if ymd_mask.any():
                ymd_str = n.loc[ymd_mask].round().astype('Int64').astype(str)
                fixed.loc[ymd_mask] = pd.to_datetime(ymd_str, format='%Y%m%d', errors='coerce')

            # Unix 时间戳（秒 / 毫秒）
            sec_mask = (n >= 1_000_000_000) & (n < 10_000_000_000)
            if sec_mask.any():
                fixed.loc[sec_mask] = pd.to_datetime(n.loc[sec_mask], unit='s', errors='coerce')

            ms_mask = (n >= 1_000_000_000_000) & (n < 10_000_000_000_000)
            if ms_mask.any():
                fixed.loc[ms_mask] = pd.to_datetime(n.loc[ms_mask], unit='ms', errors='coerce')

            parsed_date.loc[need_fix] = fixed

        df['Date'] = parsed_date
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna()
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df, f"✅ 成功解析 {len(df)} 条数据"
    except Exception as e:
        return None, f"❌ 解析文件出错: {e}"


def stitch_with_akshare(history_df, etf_code):
    """
    将历史指数数据（history_df: Date, Close=指数点位）与 AkShare ETF 数据拼接。
    返回 (structured_df, scaling_factor, stitch_date, message)
    structured_df 列：Date, index_close, etf_close, combined_close
    
    逻辑：
    1. 外连接历史和AkShare数据
    2. 找最后一个"两个值都有"的日期作为锚点计算缩放比例
    3. 如果无重叠，返回错误
    4. 拼接：历史全部 + 近期AkShare独有部分
    """
    try:
        ak_df = fetch_all_from_akshare(etf_code)

        # 外连接，保留所有日期
        merged = pd.merge(
            history_df[['Date', 'Close']].rename(columns={'Close': 'index_close'}),
            ak_df.rename(columns={'ETF_Close': 'etf_close'}),
            on='Date', how='outer',
        ).sort_values('Date')

        # 找最后一个两列都有值的日期（锚点）
        overlap = merged[merged['index_close'].notna() & merged['etf_close'].notna()]
        if overlap.empty:
            return None, 1.0, None, "❌ 历史数据与 AkShare 无重叠日期，无法计算缩放比例"

        anchor         = overlap.iloc[-1]
        scaling_factor = float(anchor['index_close']) / float(anchor['etf_close'])
        stitch_date    = anchor['Date'].date()

        # 历史段：按原顺序包含历史数据的所有日期
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
               f"✅ 拼接成功，缩放比例: {scaling_factor:.4f}，锚点日期: {stitch_date}，新增近期 {len(recent)} 条"
    except Exception as e:
        return None, 1.0, None, f"❌ 拼接失败: {e}"


# ─── Streamlit UI ─────────────────────────────────────────────────────────────
st.title("📈 ETF 回归估值仪表板")

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
                    get_data(cfg['etf_code'])
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
                fig, res = compute_and_plot(df, etf_name, deviation_pct, scaling_factor)
                st.pyplot(fig)
                plt.close(fig)

                st.divider()
                st.subheader("指数点位 & ETF价格")
                c1, c2, c3 = st.columns(3)
                c1.metric("最新日期", res['latest_date'])
                c2.metric("指数点位", f"{res['latest_close']:,.0f}")
                c3.metric("ETF价格", f"{res['latest_etf_price']:.4f}")

                st.subheader("传统回归")
                c4, c5, c6, c7 = st.columns(4)
                c4.metric("点位", f"{res['trad_pred']:,.0f}")
                c5.metric("ETF价", f"{res['trad_pred_etf']:.4f}")
                c6.metric("偏离度", f"{res['dev_trad']:+.2f}%", delta_color="inverse")
                c7.metric("年化收益", f"{res['cagr_trad']:.2f}%")

                st.subheader("滚动回归")
                c8, c9, c10, c11 = st.columns(4)
                c8.metric("点位", f"{res['roll_pred']:,.0f}")
                c9.metric("ETF价", f"{res['roll_pred_etf']:.4f}")
                c10.metric("偏离度", f"{res['dev_roll']:+.2f}%", delta_color="inverse")
                c11.metric("年化收益", f"{res['cagr_roll']:.2f}%")
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
            numeric_cols = ["传统偏离度(%)", "滚动偏离度(%)"]
            styled = compare_df.style.background_gradient(
                subset=[c for c in numeric_cols if c in compare_df.columns],
                cmap="RdYlGn_r", vmin=-30, vmax=30,
            ).format({
                "传统偏离度(%)": lambda x: f"{x:+.2f}" if pd.notna(x) else "—",
                "滚动偏离度(%)": lambda x: f"{x:+.2f}" if pd.notna(x) else "—",
                "传统CAGR(%)": lambda x: f"{x:.2f}" if pd.notna(x) else "—",
                "滚动CAGR(%)": lambda x: f"{x:.2f}" if pd.notna(x) else "—",
            })
            st.dataframe(styled, use_container_width=True, hide_index=True)

            plot_df = compare_df.dropna(subset=["传统偏离度(%)", "滚动偏离度(%)"])
            if not plot_df.empty:
                st.subheader("偏离度可视化对比")
                fig2, ax = plt.subplots(figsize=(12, max(4, len(plot_df) * 0.7)))
                x = np.arange(len(plot_df))
                w = 0.35
                ax.barh(x + w / 2, plot_df["传统偏离度(%)"], w, color='red', alpha=0.7, label='传统偏离度')
                ax.barh(x - w / 2, plot_df["滚动偏离度(%)"], w, color='blue', alpha=0.7, label='滚动偏离度')
                ax.axvline(0, color='black', linewidth=1)
                ax.axvline(deviation_pct, color='orange', linestyle=':', alpha=0.8)
                ax.axvline(-deviation_pct, color='orange', linestyle=':', alpha=0.8, label=f'±{deviation_pct}%')
                ax.set_yticks(x)
                ax.set_yticklabels(plot_df["标的"], fontsize=11)
                ax.set_xlabel("偏离度 (%)")
                ax.legend(fontsize=10)
                ax.grid(True, axis='x', linestyle=':', alpha=0.5)
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)

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
                    with st.spinner("正在拼接并入库..."):
                        df_combined, scaling_factor, stitch_date, stitch_msg = stitch_with_akshare(df_new, target_code)
                    st.info(stitch_msg)

                    if df_combined is not None:
                        save_target_to_db(target_code, target_name, scaling_factor=scaling_factor, stitch_date=stitch_date)
                        save_prices_to_db(df_combined, target_code)
                        st.session_state["etf_config_runtime"][target_name] = {
                            "name": target_name,
                            "etf_code": target_code,
                            "scaling_factor": scaling_factor,
                        }
                        st.cache_data.clear()
                        st.success(f"✅ 已新增：{target_name} ↔ {target_code}，历史数据已入库")
                        st.rerun()
