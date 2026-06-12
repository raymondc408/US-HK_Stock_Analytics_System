import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import io

# --- 語系設定數據 ---
LANG_DICT = {
    "中文": {
        "title": "🚀 美港股五維度量化分析系統",
        "subtitle": "本系統採用：趨勢、估值、量能、動能、波動 五大維度即時評分。",
        "sidebar_header": "📂 管理監控清單",
        "add_ticker": "輸入代號:",
        "add_btn": "新增",
        "del_header": "選取要刪除的股票:",
        "del_btn": "刪除選取股票",
        "run_btn": "🔥 開始即時量化分析",
        "table_header": "📊 量化決策清單 (按分數排名)",
        "download_btn": "📥 下載完整 Excel 報告",
        "logic_header": "📝 點擊查看量化評分詳情說明",
        "logic_content": "- **趨勢**: 越接近 50MA 分數越高。\n- **估值**: PEG < 1.2 為理想區。\n- **動能**: RVOL 與 1W/1M 資金趨勢。\n- **指標**: RSI 45-65 為強勢區。\n- **波動**: ATR 佔比越低得分越高。",
        "cols": {
            "Rank": "排名", "Ticker": "代號", "Name": "名稱", "Score": "量化總分", "Action": "交易決策",
            "Price": "現價", "Buy": "建議買入價(50MA)", "SL": "建議止損價", "TP": "建議止盈價",
            "ATR_pct": "ATR佔比%", "RVOL": "相對量(RVOL)", "V_Trend": "量能趨勢(1w/1m)", "RSI": "RSI(14)", "PEG": "PEG"
        }
    },
    "English": {
        "title": "🚀 US/HK Multi-Factor Quant System",
        "subtitle": "Analysis based on: Trend, Value, Volume, Momentum, and Volatility.",
        "sidebar_header": "📂 Manage Watchlist",
        "add_ticker": "Enter Ticker:",
        "add_btn": "Add",
        "del_header": "Select Tickers to Remove:",
        "del_btn": "Remove Selected",
        "run_btn": "🔥 Start Quant Analysis",
        "table_header": "📊 Quant Decision List (Ranked by Score)",
        "download_btn": "📥 Download Excel Report",
        "logic_header": "📝 Click to view Scoring Logic",
        "logic_content": "- **Trend**: Score is higher when price is near 50MA.\n- **Value**: PEG < 1.2 is ideal.\n- **Volume**: Analysis of RVOL and 1W/1M Trends.\n- **Momentum**: RSI 45-65 is the spot.\n- **Volatility**: Lower ATR% scores higher (stability).",
        "cols": {
            "Rank": "Rank", "Ticker": "Ticker", "Name": "Name", "Score": "Quant Score", "Action": "Action",
            "Price": "Price", "Buy": "Buy Price(50MA)", "SL": "Stop Loss", "TP": "Take Profit",
            "ATR_pct": "ATR%", "RVOL": "RVOL", "V_Trend": "Vol Trend(1w/1m)", "RSI": "RSI(14)", "PEG": "PEG"
        }
    }
}

# --- 核心量化算法 ---
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(data_high, data_low, data_close, window=14):
    high_low = data_high - data_low
    high_close = np.abs(data_high - data_close.shift())
    low_close = np.abs(data_low - data_close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=window).mean()

def run_quant_analysis(tickers, lang):
    analysis_list = []
    progress_bar = st.progress(0)
    
    for i, ticker_symbol in enumerate(tickers):
        try:
            stock = yf.Ticker(ticker_symbol)
            hist = stock.history(period="10mo")
            
            # --- 重要修正：過濾掉空數據 (解決港股 None 問題) ---
            hist = hist.dropna(subset=['Close'])
            
            if len(hist) < 60: continue

            info = stock.info
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2]
            ma50 = hist['Close'].rolling(window=50).mean().iloc[-1]
            rsi = calculate_rsi(hist['Close']).iloc[-1]
            atr = calculate_atr(hist['High'], hist['Low'], hist['Close']).iloc[-1]
            atr_pct = (atr / current_price) * 100
            
            peg = info.get('pegRatio')
            is_generic = peg is None or np.isnan(float(peg))
            
            w = [0.30, 0.20, 0.20, 0.15, 0.15] if not is_generic else [0.45, 0.00, 0.25, 0.15, 0.15]
            s_trend = max(0, 100 - (abs(current_price - ma50) / ma50 * 500))
            s_peg = 100 if (not is_generic and peg < 1.2) else (max(0, 100 - (peg - 1.2) * 40) if not is_generic else 0)
            
            vol_today = hist['Volume'].iloc[-1]
            vol_avg_1w = hist['Volume'].tail(5).mean()
            vol_avg_1m = hist['Volume'].tail(21).mean()
            rvol, vol_trend = vol_today / vol_avg_1w, vol_avg_1w / vol_avg_1m
            
            s_vol = (min(50, (rvol/1.5)*25) + min(50, (vol_trend/1.2)*40))
            if current_price > prev_close and vol_today < vol_avg_1w: s_vol *= 0.8
            s_rsi = 100 if 45 <= rsi <= 65 else (10 if rsi > 75 else 50)
            s_atr = max(0, 100 - (atr_pct * 15)) if atr_pct > 1.5 else 100

            final_score = (s_trend * w[0]) + (s_peg * w[1]) + (s_vol * w[2]) + (s_rsi * w[3]) + (s_atr * w[4])
            
            if lang == "中文":
                action = "★強烈建議買入" if final_score >= 82 else ("☆建議買入" if final_score >= 65 else ("建議減碼" if final_score < 40 else "觀望"))
            else:
                action = "★Strong Buy" if final_score >= 82 else ("☆Buy" if final_score >= 65 else ("Reduce" if final_score < 40 else "Wait"))

            analysis_list.append({
                "Ticker": ticker_symbol,
                "Name": info.get('longName', 'N/A'),
                "Score": round(final_score, 1),
                "Action": action,
                "Price": round(current_price, 2),
                "Buy": round(ma50, 2),
                "SL": round(current_price - 2*atr, 2),
                "TP": round(current_price + 3*atr, 2),
                "ATR_pct": f"{round(atr_pct, 2)}%",
                "RVOL": round(rvol, 2),
                "V_Trend": round(vol_trend, 2),
                "RSI": round(rsi, 1),
                "PEG": round(peg, 2) if not is_generic else "N/A"
            })
        except Exception as e:
            pass # 忽略抓不到數據的單個股票
        progress_bar.progress((i + 1) / len(tickers))
    return pd.DataFrame(analysis_list)

# --- Streamlit UI ---
st.set_page_config(page_title="Quant System", layout="wide")

if 'lang' not in st.session_state: st.session_state.lang = "中文"
with st.sidebar:
    lang_choice = st.radio("🌐 Language / 語系", ["中文", "English"], horizontal=True)
    st.session_state.lang = lang_choice

L = LANG_DICT[st.session_state.lang]
st.title(L["title"])
st.markdown(L["subtitle"])

if 'tickers' not in st.session_state:
    st.session_state.tickers = [
        'SOXX', 'MAGS', 'RKLB', 'NLR', 'COIN', 'GLD', 
        '1919.HK', '0293.HK', '0992.HK', '0316.HK', '1088.HK', 
        '0941.HK', '0005.HK', '0522.HK', '3988.HK', '9961.HK', 
        '0001.HK', 'AMD', 'ASML', 'BTC-USD', 'COST', '^SOX', 'TSM',
        'NVDA', 'AAPL', 'GOOGL', 'TSLA', 'QQQ', 'VOO', 'BRK-B', 'XLC'
    ]

with st.sidebar:
    st.header(L["sidebar_header"])
    new_ticker = st.text_input(L["add_ticker"]).upper()
    if st.button(L["add_btn"]):
        if new_ticker and new_ticker not in st.session_state.tickers:
            st.session_state.tickers.append(new_ticker)
            st.rerun()
    st.write("---")
    to_remove = st.multiselect(L["del_header"], st.session_state.tickers)
    if st.button(L["del_btn"]):
        st.session_state.tickers = [t for t in st.session_state.tickers if t not in to_remove]
        st.rerun()
    st.info(f"當前監控數: {len(st.session_state.tickers)}")

if st.button(L["run_btn"], type="primary"):
    results = run_quant_analysis(st.session_state.tickers, st.session_state.lang)
    if not results.empty:
        results = results.sort_values(by="Score", ascending=False).reset_index(drop=True)
        results.index += 1
        results.insert(0, 'Rank', results.index)

        st.subheader(L["table_header"])
        display_df = results.rename(columns=L["cols"])
        
        st.dataframe(
            display_df.style.map(lambda v: 'background-color: #C6EFCE; color: #006100' if ('建議買入' in str(v) or 'Buy' in str(v)) else '', subset=[L["cols"]["Action"]]),
            use_container_width=True, height=800, hide_index=True
        )

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            display_df.to_excel(writer, index=False, sheet_name='Quant Analysis')
        st.download_button(L["download_btn"], output.getvalue(), f"Quant_{datetime.now().strftime('%Y%m%d')}.xlsx")

with st.expander(L["logic_header"]):
    st.write(L["logic_content"])