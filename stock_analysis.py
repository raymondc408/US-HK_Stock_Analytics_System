import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time # 新增：用於控制請求頻率

# --- 0. 基礎配置與 CSS 注入 ---
st.set_page_config(page_title="Quant Dual Lab v6.18 Pro", layout="wide")

st.markdown("""
    <style>
    div[data-testid="stDataFrame"] td { text-align: center !important; vertical-align: middle !important; }
    div[data-testid="stDataFrame"] th { text-align: center !important; vertical-align: middle !important; line-height: 1.5 !important; }
    .stMetric { text-align: center !important; }
    .model-info-container {
        min-height: 400px; 
        background-color: #f8f9fb;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1E3A5F;
        margin-bottom: 10px;
        line-height: 1.4;
        font-size: 14px;
    }
    .backtest-input-block { min-height: 230px; padding: 5px; margin-bottom: 0px; }
    </style>
    """, unsafe_allow_html=True)

if 'tickers' not in st.session_state:
    st.session_state.tickers = ['VOO', 'QQQ', 'NVDA', 'AAPL', 'TSLA', '1919.HK', '0941.HK', '0005.HK']
if 'res_614' not in st.session_state: st.session_state.res_614 = None
if 'res_cd' not in st.session_state: st.session_state.res_cd = None
if 'df_dict_614' not in st.session_state: st.session_state.df_dict_614 = {}
if 'df_dict_cd' not in st.session_state: st.session_state.df_dict_cd = {}

# --- 核心計算函數 ---
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))

def calculate_atr(df, window=14):
    h_l, h_c = df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift())
    l_c = np.abs(df['Low'] - df['Close'].shift())
    return pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(window=window).mean()

def min_max_normalize(series, inverse=False):
    s = pd.to_numeric(series, errors='coerce').fillna(series.median() if not series.empty else 0)
    if s.max() == s.min(): return s * 0 + 50
    norm = (s - s.min()) / (s.max() - s.min()) * 100
    return (100 - norm) if inverse else norm

def calculate_dca_multiplier(df, months_interval):
    d_df = df.copy()
    dca_dates = d_df.resample(f'{months_interval}MS').first().index
    actual = [d for d in dca_dates if d in d_df.index]
    if d_df.index[0] not in actual: actual.insert(0, d_df.index[0])
    d_df['Cash'] = 0.0
    d_df.loc[actual, 'Cash'] = 1000.0
    cum_cash = d_df['Cash'].cumsum()
    shares = (d_df['Cash'] / d_df['Close']).cumsum()
    res = (shares * d_df['Close']) / cum_cash.replace(0, np.nan)
    return res.ffill().fillna(1.0)

PERF_COL_CFG = {
    "策:最終倍數": st.column_config.NumberColumn("策:最終倍數", format="%.2fx"),
    "策:年度回報": st.column_config.NumberColumn("策:年度回報", format="%.2f%%"),
    "策:交易次數": st.column_config.NumberColumn("策:交易次數", format="%d"),
    "平:最終倍數": st.column_config.NumberColumn("平:最終倍數", format="%.2fx"),
    "平:年度回報": st.column_config.NumberColumn("平:年度回報", format="%.2f%%")
}

def style_center_df(df):
    return df.style.set_properties(**{'text-align': 'center'}) \
                   .map(lambda x: 'background-color: #E6F3FF', subset=[c for c in df.columns if "策:" in c]) \
                   .map(lambda x: 'background-color: #FFF4E6', subset=[c for c in df.columns if "平:" in c])

# --- 1. Model v6.14 分析引擎 (優化版) ---
def run_analysis_614(tickers):
    res = []
    bar = st.empty(); prog = bar.progress(0)
    for i, t in enumerate(tickers):
        try:
            s = yf.Ticker(t); h = s.history(period="1y").dropna()
            if h.empty: continue
            # 雲端防封鎖：只在獲取基本面時調用 info，且加入 try-except
            try: info = s.info
            except: info = {}
            
            cp, ma50 = h['Close'].iloc[-1], h['Close'].rolling(min(len(h), 50)).mean().iloc[-1]
            rsi, atr = calculate_rsi(h['Close']).iloc[-1], calculate_atr(h).iloc[-1]
            peg = info.get('pegRatio', np.nan)
            w = [0.30, 0.20, 0.20, 0.15, 0.15] if not np.isnan(float(peg or np.nan)) else [0.45, 0.00, 0.25, 0.15, 0.15]
            s_t, s_p = max(0, 100-(abs(cp-ma50)/ma50*500)), 100 if (peg and peg < 1.2) else (max(0, 100-(float(peg or 1.2)-1.2)*40) if peg else 0)
            v_t, v_1w, v_1m = h['Volume'].iloc[-1], h['Volume'].tail(5).mean(), h['Volume'].tail(21).mean()
            s_v = (min(50,(v_t/(v_1w+1e-9)/1.5)*25)+min(50,(v_1w/(v_1m+1e-9)/1.2)*40))
            score = (s_t*w[0])+(s_p*w[1])+(s_v*w[2])+(100 if 45<=rsi<=65 else 50)*w[3]+max(0, 100-((atr/cp)*1500))*w[4]
            res.append({
                "Ticker": t, "Name": info.get('shortName', t), "Score": round(score, 1), 
                "Action": "★建議買入" if score >= 65 else "觀望", "Price": round(cp, 2),
                "買入價(50MA)": round(ma50, 2), "止損價": round(cp-2*atr, 2), 
                "止盈價": round(cp+3*atr, 2), "RSI": round(rsi, 1), "ATR%": f"{round((atr/cp)*100,2)}%"
            })
            time.sleep(0.2) # 減少請求頻率
        except: pass
        prog.progress((i+1)/len(tickers))
    bar.empty()
    df_out = pd.DataFrame(res)
    return df_out.sort_values("Score", ascending=False).reset_index(drop=True) if not df_out.empty else df_out

# --- 2. Model CD v6.15 分析引擎 (優化版) ---
def run_analysis_cd(tickers):
    raw_list = []
    bar = st.empty(); prog = bar.progress(0)
    for i, t in enumerate(tickers):
        try:
            s = yf.Ticker(t); h = s.history(period="2y").dropna()
            if h.empty: continue
            try: info = s.info
            except: info = {}
            raw_list.append({"Ticker": t, "Name": info.get('shortName', t), "PE": info.get('trailingPE', np.nan), "ROE": info.get('returnOnEquity', 0), "Margin": info.get('profitMargins', 0), "Debt": info.get('debtToEquity', 100), "RevGrow": info.get('revenueGrowth', 0), "FCF": info.get('freeCashflow', 0), "Price": h['Close'].iloc[-1], "hist": h})
            time.sleep(0.2)
        except: pass
        prog.progress((i + 1) / len(tickers))
    bar.empty()
    if not raw_list: return pd.DataFrame()
    df_r = pd.DataFrame(raw_list)
    v_s, p_s, h_s, g_s = min_max_normalize(df_r['PE'], True), (min_max_normalize(df_r['ROE']) + min_max_normalize(df_r['Margin'])) / 2, (min_max_normalize(df_r['Debt'], True) + df_r['FCF'].apply(lambda x: 100 if x > 0 else 0)) / 2, min_max_normalize(df_r['RevGrow'])
    df_r['F_Score'] = (v_s * 0.30 + p_s * 0.35 + h_s * 0.20 + g_s * 0.15)
    final = []
    for _, r in df_r.iterrows():
        fv = r['F_Score']
        if fv >= 65:
            h = r['hist']; cp = h['Close'].iloc[-1]; ma200, ma50 = h['Close'].rolling(min(len(h), 200)).mean().iloc[-1], h['Close'].rolling(min(len(h), 50)).mean().iloc[-1]
            atr_v = calculate_atr(h).iloc[-1]
            ts = (2 if cp > ma200 else 0) + (2 if cp > ma50 else 0) + (2 if h['Volume'].iloc[-1] > h['Volume'].tail(20).mean()*1.5 else 0)
            dec = "✅ 積極買入" if ts >= 6 else ("△ 分批建倉" if ts >= 4 else "⏳ 觀察")
            final.append({"Ticker": r['Ticker'], "F_Score": round(fv, 1), "T_Score": ts, "Action": dec, "Price": round(r['Price'], 2), "買入價(50MA)": round(ma50,2), "止損價": round(cp-2*atr_v,2), "止盈價": round(cp+3*atr_v,2)})
        else:
            final.append({"Ticker": r['Ticker'], "F_Score": round(fv, 1), "T_Score": "N/A", "Action": "X 淘汰", "Price": round(r['Price'], 2), "買入價(50MA)": "N/A", "止損價": "N/A", "止盈價": "N/A"})
    df_out = pd.DataFrame(final)
    return df_out.sort_values("F_Score", ascending=False).reset_index(drop=True) if not df_out.empty else df_out

# --- 側邊欄 ---
with st.sidebar:
    st.title("📂 清單管理中心")
    up_file = st.file_uploader("1. 導入股票文件 (.txt)", type=["txt"])
    if up_file and st.button("📥 確認導入"):
        new_list = [line.strip().upper() for line in up_file.getvalue().decode("utf-8").splitlines() if line.strip()]
        st.session_state.tickers = list(set(st.session_state.tickers + new_list)); st.rerun()
    man_t = st.text_input("2. 手動新增代號:").upper()
    if st.button("Add Manual"):
        if man_t and man_t not in st.session_state.tickers: st.session_state.tickers.append(man_t); st.rerun()
    st.write("---")
    st.subheader(f"📋 目前清單 (共 {len(st.session_state.tickers)} 隻):")
    st.info(", ".join(st.session_state.tickers))
    if st.session_state.tickers:
        st.download_button(label="📤 匯出目前清單 (.txt)", data="\n".join(st.session_state.tickers), file_name="tickers.txt", mime="text/plain")
    if st.button("⚠️ Clear All"): st.session_state.tickers = []; st.rerun()

# --- 主介面 ---
st.title("🔬 量化對照實驗室 v6.18 Final Pro")
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("🟢 Strategy Model: 綜合加權分析")
    st.markdown("""<div class="model-info-container"><b>模型詳細說明：</b><br>本模型採用多因子加權評分制，對市場動能與趨勢極為敏感。<br><br><ul><li><b>趨勢 (30%)</b>: 股價相對於50MA的偏離度。回踩支撐 ±2% 得分最高。</li><li><b>估值 (20%)</b>: PEG比率，尋找增長與價格的平衡。</li><li><b>量能 (20%)</b>: 今日相對量(RVOL)與週/月成交趨勢。</li><li><b>指標 (15%)</b>: RSI 強勢區間(45-65)判定。</li><li><b>波動 (15%)</b>: ATR佔比，優先選擇低波動穩健標的。</li></ul><i>*註：ETF模式下估值權重會自動轉移至趨勢(45%)與量能(25%)。</i></div>""", unsafe_allow_html=True)
    if st.button("🔥 執行分析", key="r14"):
        st.session_state.res_614 = run_analysis_614(st.session_state.tickers)
    if st.session_state.res_614 is not None:
        if st.session_state.res_614.empty: st.warning("無法抓取數據，請稍後再試。")
        else:
            d = st.session_state.res_614.copy(); d.index += 1; d.insert(0, 'Rank', d.index)
            st.dataframe(d.style.set_properties(**{'text-align': 'center'}).map(lambda v: 'background-color: #C6EFCE' if '建議' in str(v) else '', subset=['Action']), use_container_width=True, hide_index=True)
    st.write("---")
    st.markdown("#### 🏆 Backtesting for Strategy Model: 綜合加權分析")
    with st.container():
        st.markdown('<div class="backtest-input-block">', unsafe_allow_html=True)
        tks14 = st.multiselect("回測代號 (加權):", st.session_state.tickers, key="tks14")
        yr14 = st.selectbox("年期 (加權):", [1, 2, 3, 5, 10, 20, 30], index=2, key="yr14")
        dca14 = st.selectbox("DCA 頻率 (加權):", [1, 2, 3, 6, 12], index=0, format_func=lambda x: f"每 {x} 個月", key="d14")
        st.markdown('</div>', unsafe_allow_html=True)
    if st.button("🚀 開始回測 (v6.14)", key="bt14") and tks14:
        st.session_state.df_dict_614 = {}
        for t in tks14:
            df = yf.download(t, start=datetime.now()-timedelta(days=yr14*365+150), end=datetime.now(), progress=False)
            if not df.empty:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                st.session_state.df_dict_614[t] = {"df": df.dropna(subset=['Close']), "name": t}
    if st.session_state.df_dict_614:
        sm = []
        for t, data in st.session_state.df_dict_614.items():
            df = data["df"]; ma = df['Close'].rolling(min(len(df), 50)).mean(); score = (100-(abs(df['Close']-ma)/ma*500)).clip(0,100)
            sig = (score >= 65).astype(int).shift(1).fillna(0); strat_y, dca_y = (1 + df['Close'].pct_change() * sig).cumprod(), calculate_dca_multiplier(df, dca14)
            sm.append({"代號": t, "策:最終倍數": strat_y.iloc[-1], "策:年度回報": (strat_y.iloc[-1]**(1/yr14)-1)*100, "策:交易次數": int(sig.diff().abs().sum()), "平:最終倍數": dca_y.iloc[-1], "平:年度回報": (dca_y.iloc[-1]**(1/yr14)-1)*100})
        st.dataframe(style_center_df(pd.DataFrame(sm)), use_container_width=True, hide_index=True, column_config=PERF_COL_CFG)
        st.caption("Remark: 策 = 策略投資法 | 平 = 平衡投資法 (DCA)")

with col_r:
    st.subheader("🔵 Strategy Model: 基本和技術分析")
    st.markdown("""<div class="model-info-container"><b>二階段選股邏輯說明：</b><br>本模型結合基本面深度篩選與技術面強勢擇時，追求高品質交易。<br><br>1. <b>第一階段 (基本面 F-Score)</b>:<br>&nbsp;&nbsp;&nbsp;◦ 計算 估值(30%) + 盈利(35%) + 財務(20%) + 成長(15%)。<br>&nbsp;&nbsp;&nbsp;◦ 門檻：F-Score ≧ 65分合格，未達標則直接淘汰。<br><br>2. <b>第二階段 (技術面 T-Score)</b>:<br>&nbsp;&nbsp;&nbsp;◦ 針對合格股進行技術打分 (滿分 8分)。<br>&nbsp;&nbsp;&nbsp;◦ 包含 200MA位置、均線排列、成交量爆發。<br>&nbsp;&nbsp;&nbsp;◦ 決策：6分以上積極買入 / 4分分批建倉。</div>""", unsafe_allow_html=True)
    if st.button("🚀 執行分析", key="rcd"):
        st.session_state.res_cd = run_analysis_cd(st.session_state.tickers)
    if st.session_state.res_cd is not None:
        if st.session_state.res_cd.empty: st.warning("無法抓取基本面數據。")
        else:
            d = st.session_state.res_cd.copy(); d.index += 1; d.insert(0, 'Rank', d.index)
            st.dataframe(d.style.set_properties(**{'text-align': 'center'}).map(lambda v: 'background-color: #C6EFCE' if '積極' in str(v) else ('background-color: #FFC7CE' if '淘汰' in str(v) else ''), subset=['Action']), use_container_width=True, hide_index=True)
    st.write("---")
    st.markdown("#### 🏆 Backtesting for Strategy Model: 基本和技術分析")
    with st.container():
        st.markdown('<div class="backtest-input-block">', unsafe_allow_html=True)
        tks_cd = st.multiselect("回測代號 (CD):", st.session_state.tickers, key="tks_cd")
        yrcd = st.selectbox("年期 (CD):", [1, 2, 3, 5, 10, 20, 30], index=2, key="yrcd")
        dcacd = st.selectbox("DCA 頻率 (CD):", [1, 2, 3, 6, 12], index=0, format_func=lambda x: f"每 {x} 個月", key="dcd")
        st.markdown('</div>', unsafe_allow_html=True)
    if st.button("🚀 開始 CD 回測", key="btcd") and tks_cd:
        st.session_state.df_dict_cd = {}
        for t in tks_cd:
            df = yf.download(t, start=datetime.now()-timedelta(days=yrcd*365+150), end=datetime.now(), progress=False)
            if not df.empty:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                st.session_state.df_dict_cd[t] = {"df": df.dropna(subset=['Close']), "name": t}
    if st.session_state.df_dict_cd:
        sm_cd = []
        for t, data in st.session_state.df_dict_cd.items():
            df = data["df"]; ma = df['Close'].rolling(min(len(df), 50)).mean(); score = (100-(abs(df['Close']-ma)/ma*500)).clip(0,100)
            sig = (score >= 70).astype(int).shift(1).fillna(0); strat_y, dca_y = (1+df['Close'].pct_change()*sig).cumprod(), calculate_dca_multiplier(df, dcacd)
            sm_cd.append({"代號": t, "策:最終倍數": strat_y.iloc[-1], "策:年度回報": (strat_y.iloc[-1]**(1/yrcd)-1)*100, "策:交易次數": int(sig.diff().abs().sum()), "平:最終倍數": dca_y.iloc[-1], "平:年度回報": (dca_y.iloc[-1]**(1/yrcd)-1)*100})
        st.dataframe(style_center_df(pd.DataFrame(sm_cd)), use_container_width=True, hide_index=True, column_config=PERF_COL_CFG)
        st.caption("Remark: 策 = 策略投資法 | 平 = 平衡投資法 (DCA)")
