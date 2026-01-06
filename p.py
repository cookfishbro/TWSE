import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(page_title="TWSE-Alpha Core (Live)", layout="wide", page_icon="📈")

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    .stMetric { background-color: #1e293b; padding: 10px; border-radius: 5px; border: 1px solid #334155; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem; color: #f8fafc; }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem; color: #94a3b8; }
    .css-1d391kg { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

TICKER_MAP = {
    # 半導體 / Foundry & IDM
    "2330.TW": "台積電", "2303.TW": "聯電", "5347.TWO": "世界先進", "3711.TW": "日月光投控",
    "6770.TW": "力積電", "2342.TW": "茂矽",
    
    # IC設計 / IP / Design
    "2454.TW": "聯發科", "3034.TW": "聯詠", "2379.TW": "瑞昱", "3443.TW": "創意", 
    "3661.TW": "世芯-KY", "3035.TW": "智原", "3529.TWO": "力旺", "5274.TW": "信驊",
    "3227.TWO": "原相", "8299.TWO": "群聯", "4961.TW": "天鈺", "8016.TW": "矽創",
    
    # AI / 伺服器 / 組裝 (AI Hardware)
    "2317.TW": "鴻海", "2382.TW": "廣達", "3231.TW": "緯創", "2356.TW": "英業達",
    "6669.TW": "緯穎", "2376.TW": "技嘉", "2357.TW": "華碩", "2301.TW": "光寶科",
    "2324.TW": "仁寶", "3706.TW": "神達", "2353.TW": "宏碁", "2395.TW": "研華",

    # PCB / 被動元件 / 零組件 (Components)
    "3037.TW": "欣興", "2313.TW": "華通", "3044.TW": "健鼎", "2368.TW": "金像電",
    "6274.TW": "台燿", "2327.TW": "國巨", "2492.TW": "華新科", "2308.TW": "台達電",
    "2383.TW": "台光電", "6213.TW": "聯茂", "2456.TW": "奇力新", "4958.TW": "臻鼎-KY",

    # 光學 / 高價 (Optical)
    "3008.TW": "大立光", "3406.TW": "玉晶光", "3362.TW": "先進光", "3504.TW": "揚明光",

    # 記憶體 (Memory)
    "2408.TW": "南亞科", "2344.TW": "華邦電", "2337.TW": "旺宏", "2451.TW": "創見", "8271.TW": "宇瞻",

    # 重電 / 綠能 (Heavy Electric & Green Energy)
    "1513.TW": "中興電", "1519.TW": "華城", "1503.TW": "士電", "1504.TW": "東元",
    "1605.TW": "華新", "9958.TW": "世紀鋼", "1514.TW": "亞力", "6806.TW": "森崴能源",

    # 航運 / 航空 (Shipping & Aviation)
    "2603.TW": "長榮", "2609.TW": "陽明", "2615.TW": "萬海", "2637.TW": "慧洋-KY",
    "2618.TW": "長榮航", "2610.TW": "華航", "2606.TW": "裕民", "2605.TW": "新興",

    # 金融 (Financials)
    "2881.TW": "富邦金", "2882.TW": "國泰金", "2891.TW": "中信金", "2886.TW": "兆豐金",
    "2884.TW": "玉山金", "2892.TW": "第一金", "2880.TW": "華南金", "2885.TW": "元大金",
    "2883.TW": "開發金", "2887.TW": "台新金", "5880.TW": "合庫金", "5871.TW": "中租-KY",
    "2890.TW": "永豐金", "2801.TW": "彰銀",

    # 傳產 / 內需 / 電信 (Old Economy & Consumer)
    "2002.TW": "中鋼", "1101.TW": "台泥", "1301.TW": "台塑", "1303.TW": "南亞",
    "1326.TW": "台化", "6505.TW": "台塑化", "1216.TW": "統一", "2912.TW": "統一超",
    "2412.TW": "中華電", "3045.TW": "台灣大", "4904.TW": "遠傳", "9910.TW": "豐泰",
    "9904.TW": "寶成", "1476.TW": "儒鴻", "1402.TW": "遠東新",

    # 生技 / 營建 / 休閒 (Others)
    "1795.TW": "美時", "4147.TW": "中裕", "6446.TW": "藥華藥", "2501.TW": "國建",
    "2542.TW": "興富發", "5522.TW": "遠雄", "9921.TW": "巨大", "9914.TW": "美利達",
    "8926.TWO": "台汽電", "9907.TW": "統一實"
}

SECTORS = {
    "半導體": ["2330.TW", "2303.TW", "5347.TWO", "3711.TW", "6770.TW", "2342.TW"],
    "IC設計/IP": ["2454.TW", "3034.TW", "2379.TW", "3443.TW", "3661.TW", "3035.TW", "3529.TWO", "5274.TW", "3227.TWO", "8299.TWO", "4961.TW", "8016.TW"],
    "AI/電腦硬體": ["2317.TW", "2382.TW", "3231.TW", "2356.TW", "6669.TW", "2376.TW", "2357.TW", "2301.TW", "2324.TW", "3706.TW", "2353.TW", "2395.TW"],
    "PCB/零組件": ["3037.TW", "2313.TW", "3044.TW", "2368.TW", "6274.TW", "2327.TW", "2492.TW", "2308.TW", "2383.TW", "6213.TW", "2456.TW", "4958.TW"],
    "重電/綠能": ["1513.TW", "1519.TW", "1503.TW", "1504.TW", "1605.TW", "9958.TW", "1514.TW", "6806.TW"],
    "高價/光學": ["3008.TW", "3406.TW", "5274.TW", "3661.TW", "3362.TW", "3504.TW"],
    "記憶體": ["2408.TW", "2344.TW", "2337.TW", "2451.TW", "8271.TW"],
    "航運/航空": ["2603.TW", "2609.TW", "2615.TW", "2637.TW", "2618.TW", "2610.TW", "2606.TW", "2605.TW"],
    "金融": ["2881.TW", "2882.TW", "2891.TW", "2886.TW", "2884.TW", "2892.TW", "2880.TW", "2885.TW", "2883.TW", "2887.TW", "5880.TW", "5871.TW", "2890.TW", "2801.TW"],
    "傳產/電信": ["2002.TW", "1101.TW", "1301.TW", "1303.TW", "1326.TW", "6505.TW", "1216.TW", "2912.TW", "2412.TW", "3045.TW", "4904.TW", "9910.TW", "9904.TW", "1476.TW", "1402.TW"],
    "生技/營建/休閒": ["1795.TW", "4147.TW", "6446.TW", "2501.TW", "2542.TW", "5522.TW", "9921.TW", "9914.TW", "8926.TWO", "9907.TW"]
}

# ==========================================
# DATA ENGINE (YFINANCE)
# ==========================================
@st.cache_data(ttl=3600)
def fetch_real_data(ticker, interval="1d", period="6mo"):
    """Fetches real price data from Yahoo Finance with dynamic interval"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            return None, []
        
        # --- Technical Indicators ---
        # 1. MA20 (Simple Moving Average)
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # 2. RSI (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 3. MACD (12, 26, 9)
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal_Line']

        # 4. Bollinger Bands (20, 2)
        std20 = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['MA20'] + (std20 * 2)
        df['BB_Lower'] = df['MA20'] - (std20 * 2)

        # --- SIMULATE CHIP DATA ---
        # Simulation Logic adjusted for data length
        np.random.seed(42) 
        price_change = df['Close'].pct_change().fillna(0)
        
        # Scale volumes based on interval (Weekly/Monthly have larger volumes)
        vol_scale = 1
        if interval == '1wk': vol_scale = 5
        if interval == '1mo': vol_scale = 20

        noise_f = np.random.normal(0, 1000 * vol_scale, len(df))
        df['Foreign_Net'] = (price_change * 50000 * vol_scale) + noise_f
        
        noise_i = np.random.normal(0, 500 * vol_scale, len(df))
        df['ITC_Net'] = (price_change * 20000 * vol_scale) + noise_i
        
        noise_d = np.random.normal(0, 800 * vol_scale, len(df))
        df['Dealer_Net'] = (price_change * 30000 * -0.5 * vol_scale) + noise_d 
        
        # Get News 
        try:
            news = stock.news
        except Exception:
            news = []
        
        return df, news
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None, []

def fetch_taiex_sentiment():
    """Fetches TAIEX Index Data for Market Thermometer"""
    try:
        twii = yf.Ticker("^TWII")
        hist = twii.history(period="3mo")
        if hist.empty:
            return 50, "Neutral", 0.0
        
        close = hist['Close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        change_pct = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100
        status = "Risk On" if current_rsi > 50 else "Risk Off"
        
        return round(current_rsi, 1), status, round(change_pct, 2)
    except:
        return 50, "Neutral", 0.0

def calculate_ics(row):
    """Institutional Conviction Score"""
    f_score = (row['Foreign_Net'] / 1000) * 0.45
    i_score = (row['ITC_Net'] / 500) * 0.35
    d_score = (row['Dealer_Net'] / 800) * 0.2
    
    if row['Dealer_Net'] > 1000 and row['Foreign_Net'] < 0:
        d_score = -d_score * 1.5
        
    return round(f_score + i_score + d_score, 2)

def get_signal(ics, rsi, price, ma20, macd, signal_line):
    trend_score = 0
    if price > ma20: trend_score += 1
    if macd > signal_line: trend_score += 1
    
    # Combined Signal Logic
    if ics > 1.5 and trend_score == 2 and rsi < 75:
        return "強力買進", "#10b981", 3 
    elif ics > 0.5 and trend_score >= 1:
        return "逢低佈局", "#10b981", 2
    elif ics < -1.5:
        return "強力賣出", "#f43f5e", -2 
    elif rsi > 80:
        return "超買警戒", "#f43f5e", -1
    elif ics < 0.5 and trend_score == 0:
        return "避開", "#f43f5e", -2
    else:
        return "觀望", "#fbbf24", 0 

def get_broker_data_simulated(foreign_net):
    buy_skew = 1.0 if foreign_net > 0 else 0.5
    sell_skew = 1.5 if foreign_net < 0 else 1.0
    noise = lambda: np.random.uniform(0.8, 1.2)
    
    data = [
        {"Broker": "摩根士丹利", "Net": int(1500 * buy_skew * noise()), "Type": "Buy"},
        {"Broker": "高盛", "Net": int(1200 * buy_skew * noise()), "Type": "Buy"},
        {"Broker": "凱基-信義", "Net": int(-800 * sell_skew * noise()), "Type": "Sell"},
        {"Broker": "群益-大安", "Net": int(-600 * sell_skew * noise()), "Type": "Sell"},
        {"Broker": "散戶", "Net": int(-400 * noise()), "Type": "Sell"}
    ]
    return pd.DataFrame(data)

# ==========================================
# UI LAYOUT
# ==========================================
def main():
    # Sidebar
    st.sidebar.title("TWSE-Alpha Core")
    st.sidebar.caption("Source: Yahoo Finance (Real Price)")
    
    selected_sector = st.sidebar.selectbox("選擇板塊", ["全部"] + list(SECTORS.keys()))
    
    if selected_sector == "全部":
        current_tickers = TICKER_MAP.keys()
    else:
        current_tickers = SECTORS[selected_sector]
        
    st.title("TWSE 法人籌碼決策系統 v3.1")
    st.caption(f"數據日期: {datetime.now().strftime('%Y-%m-%d')} | 包含 MACD, 布林通道, K線圖 (日/週/月切換)")
    
    if st.button("🔄 更新數據", type="primary"):
        st.cache_data.clear()

    # --- 1. DATA FETCHING LOOP (Default Daily for Watchlist) ---
    watchlist_data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_tickers = len(current_tickers)
    
    for i, ticker in enumerate(current_tickers):
        status_text.text(f"正在分析 {ticker} ...")
        # Default to daily for scanning
        df, _ = fetch_real_data(ticker, interval="1d", period="6mo")
        if df is not None:
            last_row = df.iloc[-1]
            ics = calculate_ics(last_row)
            signal_text, signal_color, score = get_signal(
                ics, last_row['RSI'], last_row['Close'], last_row['MA20'], 
                last_row['MACD'], last_row['Signal_Line']
            )
            
            watchlist_data.append({
                "Ticker": ticker,
                "Name": TICKER_MAP[ticker],
                "Price": f"{last_row['Close']:.1f}",
                "ICS": ics,
                "Signal": signal_text,
                "Color": signal_color,
                "Score": score,
                "RSI": last_row['RSI'],
                "MA20": last_row['MA20'],
                "MACD": last_row['MACD'],
                "BB_Lower": last_row['BB_Lower'],
                "BB_Upper": last_row['BB_Upper'],
                "RawPrice": last_row['Close'],
                "Foreign_Net": last_row['Foreign_Net']
            })
        progress_bar.progress((i + 1) / total_tickers)
    
    progress_bar.empty()
    status_text.empty()
    
    if not watchlist_data:
        st.error("無法獲取數據，請檢查網絡連接。")
        return

    # --- 2. TOP SECTION: THERMOMETER & MULTI-STRATEGY RECS ---
    taiex_rsi, taiex_status, taiex_chg = fetch_taiex_sentiment()
    
    c1, c2, c3 = st.columns([1, 1, 4])
    with c1:
        st.metric("加權指數狀態", taiex_status, f"{taiex_chg}%")
    with c2:
        st.metric("大盤 RSI", f"{taiex_rsi}")
        
    st.markdown("### 🎯 AI 多策略選股雷達 (Top 5)")
    
    # Strategy Filters
    # 1. Institutional Strong Buy (ICS Desc)
    strat_inst = sorted(watchlist_data, key=lambda x: x['ICS'], reverse=True)[:5]
    
    # 2. Bottom Fishing (RSI < 45, Price < BB_Lower * 1.05) -> Upside Potential
    strat_low = [x for x in watchlist_data if x['RSI'] < 45 and float(x['RawPrice']) < x['BB_Lower'] * 1.05]
    strat_low = sorted(strat_low, key=lambda x: x['ICS'], reverse=True)[:5]
    
    # 3. Momentum (MACD > 0, Price > MA20)
    strat_mom = [x for x in watchlist_data if x['MACD'] > 0 and float(x['RawPrice']) > float(x['MA20'])]
    strat_mom = sorted(strat_mom, key=lambda x: x['RSI'], reverse=True)[:5]

    col_s1, col_s2, col_s3 = st.columns(3)

    def render_card(col, title, items, color_border):
        with col:
            st.markdown(f"#### {title}")
            if not items:
                st.info("無符合標的")
            for rec in items:
                st.markdown(f"""
                <div style="background-color: #1e293b; padding: 12px; margin-bottom: 10px; border-radius: 8px; border-left: 5px solid {color_border};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin:0; color: #f8fafc; font-size: 1em;">{rec['Name']} <span style="color:#94a3b8; font-size:0.8em;">{rec['Ticker']}</span></h4>
                        <span style="color: {color_border}; font-weight: bold;">{rec['Price']}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                        <span style="background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px; font-size: 0.75em;">ICS: {rec['ICS']}</span>
                        <span style="color: #cbd5e1; font-size: 0.75em;">RSI: {rec['RSI']:.1f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    render_card(col_s1, "👑 法人重倉股", strat_inst, "#10b981") # Green
    render_card(col_s2, "⚓ 超跌反彈股 (布林下軌)", strat_low, "#3b82f6") # Blue
    render_card(col_s3, "🚀 MACD 動能股", strat_mom, "#f59e0b") # Orange
    
    st.divider()

    # --- 3. MAIN SECTION: WATCHLIST & DETAILS ---
    col_list, col_detail = st.columns([1, 2])
    
    with col_list:
        st.subheader(f"監控清單 ({selected_sector})")
        watch_df = pd.DataFrame(watchlist_data)
        st.dataframe(
            watch_df[['Ticker', 'Name', 'Price', 'ICS', 'Signal']],
            use_container_width=True,
            height=800, # Taller for list
            hide_index=True
        )
    
    with col_detail:
        options = [f"{d['Ticker']} {d['Name']}" for d in watchlist_data]
        selected_ticker_display = st.selectbox("選擇詳細分析標的", options)
        sel_ticker = selected_ticker_display.split(" ")[0]
        
        # --- TIMEFRAME SELECTOR ---
        st.write(" ")
        tf_col, _ = st.columns([2, 2])
        with tf_col:
            selected_tf_label = st.radio("K線週期", ["日線 (Daily)", "週線 (Weekly)", "月線 (Monthly)"], horizontal=True, index=0)
        
        # Map label to API args
        tf_map = {
            "日線 (Daily)": {"interval": "1d", "period": "1y"},
            "週線 (Weekly)": {"interval": "1wk", "period": "2y"},
            "月線 (Monthly)": {"interval": "1mo", "period": "5y"}
        }
        api_args = tf_map[selected_tf_label]
        
        # Re-fetch chart data based on user selection
        sel_data_point = next((item for item in watchlist_data if item["Ticker"] == sel_ticker), None)
        df, news = fetch_real_data(sel_ticker, interval=api_args["interval"], period=api_args["period"])
        
        if df is not None:
            # CHART 1: Candlestick & Indicators
            fig = go.Figure()
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                name="K線", increasing_line_color='#ef4444', decreasing_line_color='#10b981'
            ))
            
            # MA20 & BB
            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="20MA", line=dict(color='#f59e0b', width=1)))
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name="BB上軌", line=dict(color='#cbd5e1', width=0.5, dash='dot')))
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name="BB下軌", line=dict(color='#cbd5e1', width=0.5, dash='dot')))

            fig.update_layout(
                title=f"{TICKER_MAP[sel_ticker]} 技術分析 ({selected_tf_label})",
                yaxis=dict(title="股價", color="#e2e8f0"),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                height=600, # Increased height as requested
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Sub-Chart: MACD
            st.caption("MACD 指標")
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name="MACD柱狀", marker_color='gray'))
            fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="DIF", line=dict(color='#3b82f6')))
            fig_macd.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name="MACD", line=dict(color='#f59e0b')))
            fig_macd.update_layout(height=180, margin=dict(t=0, b=0, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), showlegend=False)
            st.plotly_chart(fig_macd, use_container_width=True)

            # SUB-SECTION: Broker Map (Simulated based on fetched data) & News
            st.markdown("---")
            sub_c1, sub_c2 = st.columns(2)
            
            with sub_c1:
                st.markdown("#### 🛡️ 券商分點多空對決 (模擬)")
                if not df.empty:
                    # Use last row from the detail DF for simulation context
                    broker_df = get_broker_data_simulated(df.iloc[-1]['Foreign_Net'])
                    fig_broker = px.bar(broker_df, x="Net", y="Broker", orientation='h', 
                                      color="Net", 
                                      color_continuous_scale=["#f43f5e", "#10b981"],
                                      text_auto=True)
                    fig_broker.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e2e8f0'),
                        height=300,
                        showlegend=False,
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig_broker, use_container_width=True)
            
            with sub_c2:
                st.markdown("#### 📰 相關新聞")
                with st.container(height=300):
                    if news:
                        count = 0
                        for item in news:
                            if count >= 4: break
                            title = item.get('title', '無標題')
                            link = item.get('link', '#')
                            publisher = item.get('publisher', '未知來源')
                            try:
                                t_ts = item.get('providerPublishTime', 0)
                                time_str = datetime.fromtimestamp(t_ts).strftime('%m-%d %H:%M')
                            except:
                                time_str = ""
                            
                            st.markdown(f"**[{title}]({link})**")
                            st.caption(f"{publisher} | {time_str}")
                            st.divider()
                            count += 1
                    else:
                        st.info("暫無相關新聞")

if __name__ == "__main__":
    main()