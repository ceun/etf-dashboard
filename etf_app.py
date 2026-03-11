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
from io import BytesIO
from urllib.parse import urlparse, parse_qs, unquote

warnings.filterwarnings('ignore')

# ─── 页面配置 ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ETF估值仪表板", page_icon="📈", layout="wide")

# ─── 全局配置 ────────────────────────────────────────────────────────────────
ETF_CONFIG = {
    "沪深300":    {"name": "沪深300",    "etf_code": "510300"},
    "红利低波":   {"name": "红利低波",   "etf_code": "563020"},
    "800现金流":  {"name": "800现金流",  "etf_code": "563990"},
    "消费龙头":   {"name": "消费龙头",   "etf_code": "159520"},
    "食品饮料":   {"name": "食品饮料",   "etf_code": "516900"},
    "港股通非银": {"name": "港股通非银", "etf_code": "513750"},
    "价值100":    {"name": "价值100",    "etf_code": "159263"},
    "深红利":     {"name": "深红利",     "etf_code": "159905"},
}

TRADITION_START = "20081031"
TRADITION_END   = "20221031"
ROLLING_WINDOW  = 1250
# Supabase 连接字符串（从 Streamlit secrets 读取）
# 优先使用连接池 URL（更稳，通常可规避 IPv6/直连网络问题）
DATABASE_URL = st.secrets.get("database_url_pooler", None) or st.secrets.get("database_url", None)
# 全局变量：存储缩放比例（指数点位 ÷ ETF价格）
SCALING_FACTOR = {}

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ─── 数据库工具 ───────────────────────────────────────────────────────────────
def get_db_connection():
    """获取数据库连接（每次新建，避免缓存已关闭连接）"""
    if not DATABASE_URL:
        return None

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

            ipv4_addr = socket.getaddrinfo(host, None, socket.AF_INET)[0][4][0]

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


def save_to_db(df, etf_code):
    """将数据写入 PostgreSQL，包含缩放比例"""
    conn = get_db_connection()
    if not conn:
        st.warning("无法连接数据库，数据将不被保存")
        return
    
    try:
        cur = conn.cursor()
        # 表已由 Supabase SQL 创建，这里只需插入数据
        scaling_factor = SCALING_FACTOR.get(etf_code, 1.0)
        
        for _, row in df.iterrows():
            cur.execute("""
                INSERT INTO etf_prices (etf_code, date, close, scaling_factor) 
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (etf_code, date) DO UPDATE 
                SET close = EXCLUDED.close, scaling_factor = EXCLUDED.scaling_factor
            """, (etf_code, row['Date'].date(), float(row['Close']), float(scaling_factor)))
        
        conn.commit()
        cur.close()
    except Exception as e:
        st.warning(f"保存数据异常: {e}")
    finally:
        if conn:
            conn.close()


def load_from_db(etf_code):
    """从 PostgreSQL 读取数据，同时加载缩放比例"""
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        df = pd.read_sql(
            "SELECT date AS Date, close AS Close, scaling_factor FROM etf_prices WHERE etf_code=%s ORDER BY date",
            conn, params=(etf_code,)
        )
        if df.empty:
            return None
        
        # 从数据库读取缩放比例，保存到全局
        latest_scaling = df['scaling_factor'].iloc[-1]
        if pd.notna(latest_scaling):
            SCALING_FACTOR[etf_code] = float(latest_scaling)
        
        df['Date'] = pd.to_datetime(df['Date'])
        df['Close'] = df['Close'].astype(float)
        # 只返回 Date 和 Close，scaling_factor 存在全局变量中
        return df[['Date', 'Close']].reset_index(drop=True)
    except Exception as e:
        st.warning(f"读取数据异常: {e}")
        return None
    finally:
        if conn:
            conn.close()


# ─── AkShare 全量拉取（云端无本地文件时使用）────────────────────────────────
def fetch_all_from_akshare(etf_code):
    """拉取 ETF 后复权全量日线数据"""
    etf_df = ak.fund_etf_hist_em(symbol=etf_code, period="daily", adjust="hfq")
    etf_df['日期'] = pd.to_datetime(etf_df['日期'])
    # 返回 Date, Close（ETF收盘价）
    etf_df = etf_df[['日期', '收盘']].rename(columns={'日期': 'Date', '收盘': 'Close'})
    return etf_df.sort_values('Date').reset_index(drop=True)


def update_with_akshare(df, etf_code):
    """增量更新：拉最新数据，计算并保存缩放比例"""
    try:
        new_all = ak.fund_etf_hist_em(symbol=etf_code, period="daily", adjust="hfq")
        new_all['日期'] = pd.to_datetime(new_all['日期'])
        new_all = new_all[['日期', '收盘']].rename(columns={'日期': 'Date', '收盘': 'ETF_Close'})
        
        last_date = df['Date'].max()
        
        # 锚点：计算缩放比例
        merged = pd.merge(df[['Date', 'Close']], new_all, on='Date', how='inner')
        if not merged.empty:
            anchor_row = merged.iloc[-1]
            scaling_factor = anchor_row['Close'] / anchor_row['ETF_Close']
            SCALING_FACTOR[etf_code] = scaling_factor
            print(f"缩放比例 ({etf_code}): {scaling_factor:.4f}")
        else:
            scaling_factor = SCALING_FACTOR.get(etf_code, 1.0)
        
        # 拼接新数据
        new_data = new_all[new_all['Date'] > last_date].copy()
        if not new_data.empty:
            new_data['Close'] = new_data['ETF_Close'] * scaling_factor
            new_data = new_data[['Date', 'Close']]
            df = pd.concat([df, new_data], ignore_index=True)
            print(f"成功拼接 {len(new_data)} 条数据")
        
        save_to_db(df, etf_code)
        return df
    except Exception as e:
        st.warning(f"⚠️ AkShare 更新失败 ({etf_code}): {e}")
        return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_data(etf_code: str):
    """
    优先从数据库加载 → 增量更新 → 写回数据库
    若数据库无数据，则从 AkShare 拉全量
    """
    df = load_from_db(etf_code)
    if df is None:
        # 首次拉取：全量历史
        df = fetch_all_from_akshare(etf_code)
        # 暂时设缩放比例为 1.0，后续用户上传 Excel 时更新
        SCALING_FACTOR[etf_code] = 1.0
        save_to_db(df, etf_code)
    else:
        # 增量更新
        df = update_with_akshare(df, etf_code)
    
    return df


# ─── 核心分析 ─────────────────────────────────────────────────────────────────
def compute_and_plot(df, etf_name, deviation_pct):
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
    
    # 用缩放比例换算成ETF价格（指数点位 ÷ 缩放比例）
    scaling_factor = SCALING_FACTOR.get(etf_name, 1.0) if isinstance(etf_name, str) else 1.0
    
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
def build_comparison(deviation_pct):
    rows = []
    for name, cfg in ETF_CONFIG.items():
        df = load_from_db(cfg['etf_code'])
        if df is None or len(df) < ROLLING_WINDOW + 10:
            rows.append({"标的": name, "ETF代码": cfg['etf_code'],
                         "最新日期": "无数据（请先刷新）",
                         "传统偏离度(%)": None, "滚动偏离度(%)": None,
                         "传统CAGR(%)": None, "滚动CAGR(%)": None})
            continue
        try:
            fig, res = compute_and_plot(df, name, deviation_pct)
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
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna()
        df = df.sort_values('Date').reset_index(drop=True)
        
        return df, f"✅ 成功解析 {len(df)} 条数据"
    except Exception as e:
        return None, f"❌ 解析文件出错: {e}"


def stitch_with_akshare(history_df, etf_code):
    """将历史数据与 AkShare 最新数据拼接，返回 (combined_df, scaling_factor, message)"""
    try:
        # 拉取 AkShare 数据
        new_all = ak.fund_etf_hist_em(symbol=etf_code, period="daily", adjust="hfq")
        new_all['日期'] = pd.to_datetime(new_all['日期'])
        new_all = new_all[['日期', '收盘']].rename(columns={'日期': 'Date', '收盘': 'ETF_Close'})
        
        last_date = history_df['Date'].max()
        
        # 查找锚点计算缩放比例
        merged = pd.merge(history_df[['Date', 'Close']], new_all, on='Date', how='inner')
        if merged.empty:
            return None, 1.0, "❌ 历史数据与 AkShare 无重叠日期，无法计算缩放比例"
        
        anchor_row = merged.iloc[-1]
        scaling_factor = anchor_row['Close'] / anchor_row['ETF_Close']
        
        # 拼接新数据
        new_data = new_all[new_all['Date'] > last_date].copy()
        if not new_data.empty:
            new_data['Close'] = new_data['ETF_Close'] * scaling_factor
            new_data = new_data[['Date', 'Close']]
            combined = pd.concat([history_df, new_data], ignore_index=True)
        else:
            combined = history_df
        
        return combined, scaling_factor, f"✅ 拼接成功，缩放比例: {scaling_factor:.4f}，新增 {len(new_data)} 条数据"
    except Exception as e:
        return None, 1.0, f"❌ 拼接 AkShare 失败: {e}"


# ─── Streamlit UI ─────────────────────────────────────────────────────────────
st.title("📈 ETF 回归估值仪表板")

with st.sidebar:
    st.header("⚙️ 参数设置")
    selected       = st.selectbox("选择标的", list(ETF_CONFIG.keys()))
    deviation_pct  = st.slider("偏离阈值 (%)", 5, 30, 15, 1)

    st.divider()
    if st.button("🔄 更新全部数据", use_container_width=True, type="primary"):
        st.cache_data.clear()
        prog = st.progress(0)
        etf_list = list(ETF_CONFIG.items())
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
                    "SELECT etf_code, COUNT(*) AS 条数, MAX(date) AS 最新日期 "
                    "FROM etf_prices GROUP BY etf_code ORDER BY etf_code", conn)
                conn.close()
                if not summary.empty:
                    st.caption("📊 Supabase 数据库已有数据")
                    st.dataframe(summary, hide_index=True, use_container_width=True)
        except Exception as e:
            st.caption(f"⚠️ 无法查询数据库: {e}")
    else:
        st.warning("⚠️ 未配置 database_url，请先设置 Streamlit secrets")

# ─── 主内容区 ─────────────────────────────────────────────────────────────────
cfg      = ETF_CONFIG[selected]
etf_code = cfg['etf_code']
etf_name = cfg['name']

tab1, tab2, tab3 = st.tabs(["📊 单标的详情", "📋 全市场对比", "⚙️ 数据管理"])

with tab1:
    with st.spinner(f"加载 {etf_name} ({etf_code}) 数据..."):
        try:
            df = get_data(etf_code)
        except Exception as e:
            st.error(f"❌ 数据加载失败：{e}")
            st.stop()

    if df is None or len(df) < ROLLING_WINDOW + 10:
        st.error("数据不足，无法计算回归，请点击「更新全部数据」")
    else:
        try:
            fig, res = compute_and_plot(df, etf_name, deviation_pct)
            st.pyplot(fig)
            plt.close(fig)

            st.divider()
            # 显示指数点位 + ETF价格两行
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
    st.caption("对比数据来自 Supabase，更新请点击侧边栏「更新全部数据」")
    with st.spinner("计算全市场偏离度..."):
        compare_df = build_comparison(deviation_pct)

    if compare_df.empty:
        st.info("数据库暂无数据，请先点击「更新全部数据」")
    else:
        numeric_cols = ["传统偏离度(%)", "滚动偏离度(%)"]
        styled = compare_df.style.background_gradient(
            subset=[c for c in numeric_cols if c in compare_df.columns],
            cmap="RdYlGn_r", vmin=-30, vmax=30
        ).format({
            "传统偏离度(%)": lambda x: f"{x:+.2f}" if pd.notna(x) else "—",
            "滚动偏离度(%)": lambda x: f"{x:+.2f}" if pd.notna(x) else "—",
            "传统CAGR(%)":   lambda x: f"{x:.2f}"  if pd.notna(x) else "—",
            "滚动CAGR(%)":   lambda x: f"{x:.2f}"  if pd.notna(x) else "—",
        })
        st.dataframe(styled, use_container_width=True, hide_index=True)

        plot_df = compare_df.dropna(subset=["传统偏离度(%)", "滚动偏离度(%)"])
        if not plot_df.empty:
            st.subheader("偏离度可视化对比")
            fig2, ax = plt.subplots(figsize=(12, max(4, len(plot_df) * 0.7)))
            x = np.arange(len(plot_df))
            w = 0.35
            ax.barh(x + w/2, plot_df["传统偏离度(%)"], w, color='red',  alpha=0.7, label='传统偏离度')
            ax.barh(x - w/2, plot_df["滚动偏离度(%)"], w, color='blue', alpha=0.7, label='滚动偏离度')
            ax.axvline(0, color='black', linewidth=1)
            ax.axvline( deviation_pct, color='orange', linestyle=':', alpha=0.8)
            ax.axvline(-deviation_pct, color='orange', linestyle=':', alpha=0.8,
                       label=f'±{deviation_pct}%')
            ax.set_yticks(x)
            ax.set_yticklabels(plot_df["标的"], fontsize=11)
            ax.set_xlabel("偏离度 (%)")
            ax.legend(fontsize=10)
            ax.grid(True, axis='x', linestyle=':', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

with tab3:
    st.subheader("📤 上传历史数据并拼接 AkShare")
    
    # 创建两列布局
    col_upload, col_stitch = st.columns(2)
    
    with col_upload:
        st.write("**第1步：上传历史数据文件**")
        uploaded_file = st.file_uploader("选择 Excel 或 CSV 文件", type=['xlsx', 'xls', 'csv'])
        
        if uploaded_file:
            df_uploaded, msg = parse_upload_file(uploaded_file)
            st.info(msg)
            if df_uploaded is not None:
                st.write(f"📊 预览数据 (前10行):")
                st.dataframe(df_uploaded.head(10), use_container_width=True)
    
    with col_stitch:
        st.write("**第2步：选择 ETF 并拼接**")
        selected_etf = st.selectbox("选择要拼接的 ETF", list(ETF_CONFIG.keys()), key="stitch_etf")
        
        if uploaded_file and st.button("🔗 开始拼接 AkShare", use_container_width=True):
            if df_uploaded is not None:
                with st.spinner("正在拼接数据..."):
                    etf_code = ETF_CONFIG[selected_etf]['etf_code']
                    df_combined, scaling_factor, msg = stitch_with_akshare(df_uploaded, etf_code)
                    st.info(msg)
                    
                    if df_combined is not None:
                        st.write(f"✅ 拼接完成! (共 {len(df_combined)} 条数据)")
                        st.dataframe(df_combined.head(10), use_container_width=True)
                        
                        # 保存按钮
                        if st.button("💾 保存到数据库", use_container_width=True, type="primary"):
                            with st.spinner("保存中..."):
                                SCALING_FACTOR[etf_code] = scaling_factor
                                save_to_db(df_combined, etf_code)
                                st.success("✅ 数据已保存到 Supabase！")
                                st.cache_data.clear()
    
    st.divider()
    st.subheader("➕ 新增 ETF")
    
    col_name, col_code = st.columns(2)
    
    with col_name:
        new_etf_name = st.text_input("ETF 名称", placeholder="例如：新etf指数")
    
    with col_code:
        new_etf_code = st.text_input("ETF 代码", placeholder="例如：159999")
    
    if st.button("➕ 添加到系统", use_container_width=True):
        if new_etf_name and new_etf_code:
            ETF_CONFIG[new_etf_name] = {
                "name": new_etf_name,
                "etf_code": new_etf_code
            }
            st.success(f"✅ 已添加 {new_etf_name} ({new_etf_code})，刷新页面生效")
            st.rerun()
        else:
            st.error("❌ 请填写完整的 ETF 名称和代码")
