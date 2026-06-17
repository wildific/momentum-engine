"""🚀 Momentum Engine v4"""
import io, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import date, timedelta

from indices import NSE_INDICES, INDEX_CATEGORIES
from engine import (fetch_yahoo, fetch_index, parse_csv, ma,
                    generate_signals, run_backtest, detect_regime, run_monte_carlo,
                    fetch_crypto_yahoo, fetch_coingecko, fetch_binance,
                    is_crypto_ticker, COINGECKO_IDS,
                    fetch_ohlcv, run_turtle_backtest, turtle_signals, compute_atr)

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Momentum Engine", page_icon="🚀",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#0D1117;}
.main .block-container{padding-top:1rem;max-width:1500px;}
[data-testid="stSidebar"]{background:#010409!important;border-right:1px solid #21262D;}
[data-testid="stSidebar"] label{color:#7D8590!important;font-size:12px!important;}
.stButton>button{background:linear-gradient(135deg,#00D4AA,#0D8A70);color:#0D1117;
  font-weight:700;border:none;border-radius:8px;padding:10px 0;width:100%;font-size:14px;}
.stButton>button:hover{opacity:.88;}
.stDownloadButton>button{background:#161B22!important;color:#00D4AA!important;
  border:1px solid #00D4AA!important;border-radius:8px!important;font-weight:600!important;}
.stTabs [data-baseweb="tab-list"]{background:#161B22;border-radius:10px;padding:4px;border:1px solid #21262D;}
.stTabs [data-baseweb="tab"]{color:#7D8590;font-weight:600;border-radius:7px;padding:6px 12px;}
.stTabs [aria-selected="true"]{background:#0D1117!important;color:#00D4AA!important;}
.mc{background:#161B22;border:1px solid #21262D;border-radius:10px;padding:14px 18px;margin-bottom:8px;}
.ml{font-size:10px;color:#7D8590;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:3px;}
.mv{font-family:'JetBrains Mono',monospace;font-size:17px;font-weight:700;color:#E6EDF3;word-break:break-word;line-height:1.3;}
.mv.pos{color:#3FB950;}.mv.neg{color:#F85149;}.mv.neu{color:#58A6FF;}
.sh{font-size:11px;font-weight:700;color:#00D4AA;text-transform:uppercase;letter-spacing:2px;
    padding:6px 0 8px;border-bottom:1px solid #21262D;margin-bottom:14px;margin-top:4px;}
.pill-bull{background:#0D4429;color:#3FB950;padding:3px 12px;border-radius:20px;font-size:12px;font-weight:700;display:inline-block;}
.pill-neutral{background:#1C2A3A;color:#58A6FF;padding:3px 12px;border-radius:20px;font-size:12px;font-weight:700;display:inline-block;}
.pill-bear{background:#3D1616;color:#F85149;padding:3px 12px;border-radius:20px;font-size:12px;font-weight:700;display:inline-block;}
.sim-card{background:#161B22;border:1px solid #21262D;border-radius:10px;padding:16px;margin-bottom:10px;}
.sim-label{color:#7D8590;font-size:11px;text-transform:uppercase;letter-spacing:1px;}
.sim-val{font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;color:#E6EDF3;}
</style>
""", unsafe_allow_html=True)

PLOT = dict(paper_bgcolor="#0D1117",plot_bgcolor="#0D1117",
            font=dict(family="Inter",color="#7D8590",size=12),
            xaxis=dict(gridcolor="#21262D",linecolor="#30363D",zeroline=False),
            yaxis=dict(gridcolor="#21262D",linecolor="#30363D",zeroline=False),
            legend=dict(bgcolor="#161B22",bordercolor="#30363D",borderwidth=1),
            margin=dict(l=50,r=20,t=40,b=40))

def mcard(label, value, pos_good=True):
    try:
        v = float(str(value).replace(",","").replace("₹","").replace("%","").replace("+",""))
        cls = ("pos" if v>=0 else "neg") if pos_good else ("neg" if v>=0 else "pos")
    except Exception:
        cls = "neu"
    sign = "+" if cls=="pos" and not str(value).startswith("-") else ""
    st.markdown(f'<div class="mc"><div class="ml">{label}</div>'
                f'<div class="mv {cls}">{sign}{value}</div></div>', unsafe_allow_html=True)

def sh(t): st.markdown(f'<div class="sh">{t}</div>', unsafe_allow_html=True)
def fmt(v, d=2, pre="", suf=""):
    try: return f"{pre}{float(v):,.{d}f}{suf}"
    except: return str(v)

# Always-available ra_map (also redefined inside run block for clarity)
ra_map = {
    "Scale Exposure":                  "Scale Exposure",
    "Exit per Strategy, No New Entry": "Exit per Strategy",
    "Keep Stocks, No New Entry":       "Keep Stocks No Entry",
    "Exit All, No New Entry":          "Exit All No Entry",
}


# ── EXCEL ─────────────────────────────────────────────────────
def build_excel(results, cfg):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        wb  = w.book
        hf  = wb.add_format({"bold":True,"bg_color":"#161B22","font_color":"#00D4AA","border":1})
        pf  = wb.add_format({"font_color":"#3FB950","bold":True})
        nf  = wb.add_format({"font_color":"#F85149","bold":True})
        rf2 = wb.add_format({"font_color":"#E6EDF3"})
        ws  = wb.add_worksheet("Summary")
        ws.set_column("A:A",34); ws.set_column("B:B",22)
        ws.write(0,0,"Momentum Engine Backtest Report",
                 wb.add_format({"bold":True,"font_size":14,"font_color":"#00D4AA"}))
        ws.write(1,0,f"Generated: {date.today()}",rf2)
        r=3; ws.write(r,0,"CONFIGURATION",hf); r+=1
        for k,v in cfg.items():
            ws.write(r,0,k,rf2); ws.write(r,1,str(v),rf2); r+=1
        r+=1; ws.write(r,0,"PERFORMANCE METRICS",hf); r+=1
        m=results["metrics"]
        for k in [x for x in m if not x.startswith("_")]:
            ws.write(r,0,k,rf2)
            try:
                fv=float(m[k]); ws.write(r,1,fv,pf if fv>=0 else nf)
            except:
                ws.write(r,1,str(m[k]),rf2)
            r+=1
        eq=results["equity"].reset_index(); eq.columns=["Date","Portfolio Value"]
        eq.to_excel(w,sheet_name="Equity Curve",index=False)
        mo=m.get("_monthly",pd.Series())
        if not mo.empty:
            try:
                mo.index=pd.to_datetime(mo.index.to_timestamp() if hasattr(mo.index,"to_timestamp") else mo.index)
                dfm=pd.DataFrame({"y":mo.index.year,"m":mo.index.month,"r":mo.values})
                piv=dfm.pivot(index="y",columns="m",values="r")
                piv.columns=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][:len(piv.columns)]
                piv.to_excel(w,sheet_name="Monthly Returns")
            except Exception: pass
        an=m.get("_annual",pd.Series())
        if not an.empty:
            try:
                an.index=pd.to_datetime(an.index.to_timestamp() if hasattr(an.index,"to_timestamp") else an.index)
                pd.DataFrame({"Year":an.index.year,"Return(%)":(an.values*100).round(2)}).to_excel(w,sheet_name="Annual Returns",index=False)
            except Exception: pass
        if not results["trades"].empty:
            results["trades"].to_excel(w,sheet_name="Trade Log",index=False)
        dd=m.get("_drawdown",pd.Series())
        if not dd.empty:
            ddf=dd.reset_index(); ddf.columns=["Date","Drawdown(%)"]
            ddf["Drawdown(%)"]=ddf["Drawdown(%)"]*100
            ddf.to_excel(w,sheet_name="Drawdown",index=False)
    return buf.getvalue()


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚀 Momentum Engine")
    st.caption("v4 — NSE Momentum Backtester")
    st.markdown("---")

    sh("Strategy Type")
    strategy_type = st.selectbox("Strategy",
        ["Momentum", "Turtle Trading"],
        help="Momentum: MA crossover + rank-based entry\nTurtle Trading: Donchian breakout + ATR position sizing")
    st.markdown("---")

    sh("Universe")
    cat      = st.selectbox("Category", list(INDEX_CATEGORIES.keys()), label_visibility="collapsed")
    idx_name = st.selectbox("Index", INDEX_CATEGORIES[cat])
    idx_info = NSE_INDICES.get(idx_name, {})
    # Crypto data source (shown only for Crypto category)
    crypto_source = "Yahoo Finance"
    if cat == "Crypto":
        crypto_source = st.selectbox("Crypto Data Source", [
            "Yahoo Finance",
            "CoinGecko (free, slower)",
            "Binance (USDT pairs)",
        ], help="Yahoo Finance: easiest, covers major pairs\nCoinGecko: 10k+ coins, free API\nBinance: best OHLCV quality, USDT pairs only")

        if crypto_source == "CoinGecko (free, slower)":
            st.info("📡 CoinGecko: uses coin IDs (bitcoin, ethereum). Rate limit: 30 req/min — may be slow for large universes.")
        elif crypto_source == "Binance (USDT pairs)":
            st.info("📡 Binance: converts tickers to USDT pairs (BTC-USD → BTCUSDT). Fastest & most accurate.")

    csv_prices = st.file_uploader("📄 Upload Price CSV", type=["csv"])
    csv_index  = st.file_uploader("📄 Upload Index/Benchmark CSV", type=["csv"])
    if csv_prices: st.success("Price CSV ✓")
    if csv_index:  st.success("Index CSV ✓")
    st.markdown("---")

    sh("Date Range & Rebalance")
    c1,c2 = st.columns(2)
    with c1: start_dt = st.date_input("Start", value=date.today()-timedelta(days=5*365), label_visibility="collapsed")
    with c2: end_dt   = st.date_input("End",   value=date.today(), label_visibility="collapsed")
    rebal = st.selectbox("Rebalance Frequency", ["Weekly","Monthly","Quarterly","Yearly"])
    rebal_day = "Friday"
    if rebal == "Weekly":
        rebal_day = st.selectbox("Rebalance Day", ["Monday","Tuesday","Wednesday","Thursday","Friday"], index=4)
    st.markdown("---")

    _is_turtle = (strategy_type == "Turtle Trading")

    sh("Moving Averages")
    if _is_turtle:
        st.caption("⚠️ Not used in Turtle Trading — entry/exit based on Donchian breakout")
    ma_type    = st.selectbox("MA Type", ["EMA","SMA"], disabled=_is_turtle)
    st.caption("Entry: price > MA(fast) AND MA(fast) > MA(slow)")
    entry_fast = st.number_input("Fast MA period", 5,  500, 50,  5, disabled=_is_turtle)
    entry_slow = st.number_input("Slow MA period", 10, 500, 200, 10, disabled=_is_turtle)
    st.caption("Exit: price < MA(exit)")
    exit_ma_type = st.selectbox("Exit MA Type", ["EMA","SMA"], key="exit_ma_type_sel", disabled=_is_turtle)
    exit_ma_p    = st.number_input("Exit MA period", 5, 500, 20, 5, disabled=_is_turtle)
    st.caption("Extra chart MAs (comma-separated)")
    extra_ma_raw = st.text_input("e.g. 20,50,200", value="20,50,200", disabled=_is_turtle)
    st.markdown("---")

    sh("Ranking Criteria")
    if _is_turtle:
        st.caption("⚠️ Not used in Turtle Trading — position sized by ATR")
    rank_by = st.selectbox("Rank by", ["Momentum","Sharpe Ratio","Return %","Low Volatility"], disabled=_is_turtle)
    lb_opts = st.multiselect("Lookback days", [21,30,60,90,120,180,252], default=[60,90,120,252], disabled=_is_turtle)
    if not lb_opts: lb_opts = [60,90,120,252]
    wcols = st.columns(min(len(lb_opts),4))
    raw_w = {}
    for i,lb in enumerate(lb_opts):
        with wcols[i%4]:
            raw_w[lb] = st.number_input(f"{lb}d",0.0,1.0,round(1/len(lb_opts),2),0.05,key=f"w{lb}", disabled=_is_turtle)
    tw = sum(raw_w.values()) or 1
    mom_w = {k: v/tw for k,v in raw_w.items()}
    st.markdown("---")

    sh("Correlation Filter")
    use_corr     = st.toggle("Filter correlated stocks", value=False, disabled=_is_turtle)
    corr_thresh  = st.slider("Max correlation threshold", 0.1, 1.0, 0.7, 0.05, disabled=not use_corr or _is_turtle)
    corr_window  = st.slider("Correlation lookback (days)", 20, 120, 60, 10, disabled=not use_corr or _is_turtle)
    st.markdown("---")

    sh("Portfolio & Allocation")
    if _is_turtle:
        st.caption("⚠️ Position sizing handled by ATR risk % above")
    top_n      = st.slider("Stocks to hold", 1, 40, 10, disabled=_is_turtle)
    ex_rank    = st.slider("Exit rank threshold", top_n, 60, min(top_n*2,20), disabled=_is_turtle)
    capital    = st.number_input("Initial Portfolio (₹)", value=1_000_000, step=100_000, format="%d")
    alloc_mode = st.selectbox("Allocation Mode", ["Reinvestment","SIP","Fixed"], disabled=_is_turtle)
    sip_amount = 0
    if alloc_mode == "SIP" and not _is_turtle:
        sip_amount = st.number_input("SIP Amount/period (₹)", value=50_000, step=10_000, format="%d")
    sizing  = st.selectbox("Position Sizing", ["Equal Weight","Inverse Volatility"], disabled=_is_turtle)
    txn     = st.slider("Transaction Cost (%)", 0.0, 1.0, 0.1) / 100
    slip    = st.slider("Slippage (%)", 0.0, 1.0, 0.1) / 100
    rf_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 6.5) / 100
    st.markdown("---")

    sh("Regime Detection")
    use_regime   = st.toggle("Enable Regime Filter", value=True)
    st.caption("BULL = price > Short MA > Long MA")
    reg_fast     = st.number_input("Short MA period (e.g. 50)", 10, 500, 50,  10, disabled=not use_regime)
    reg_slow     = st.number_input("Long MA period  (e.g. 200)", 10, 500, 200, 10, disabled=not use_regime)
    reg_ma_type  = st.selectbox("Regime MA type", ["EMA","SMA"], disabled=not use_regime)
    vol_thresh   = st.slider("Bull vol threshold (%)", 10, 50, 20, disabled=not use_regime) / 100
    if reg_fast >= reg_slow:
        st.warning(f"⚠️ Short MA ({reg_fast}) ≥ Long MA ({reg_slow}). They will be auto-swapped.")
    regime_action = "Scale Exposure"
    if use_regime:
        regime_action = st.selectbox("Bear Regime Action", [
            "Scale Exposure",
            "Exit per Strategy, No New Entry",
            "Keep Stocks, No New Entry",
            "Exit All, No New Entry",
        ])
    st.markdown("---")

    sh("Monte Carlo")
    mc_method = st.selectbox("Method", ["Bootstrap","Parametric","Trade Shuffle"])
    mc_n      = st.select_slider("Simulations", [200,500,1000,2000,5000], value=1000)
    st.markdown("---")

    # ── Turtle Parameters (shown only when Turtle selected) ───
    turtle_system     = 1
    turtle_atr_window = 20
    turtle_risk_pct   = 1.0
    turtle_stop_mult  = 2.0
    turtle_max_units  = 4
    turtle_pyramid    = True
    turtle_pyr_step   = 0.5
    turtle_trail_win  = None

    if strategy_type == "Turtle Trading":
        sh("Turtle Trading Parameters")
        turtle_system = st.selectbox("System",
            [1, 2],
            format_func=lambda x: f"System {x} — {'20' if x==1 else '55'}-day breakout",
            help="System 1: 20-day breakout (more trades)\nSystem 2: 55-day breakout (longer trends)")
        turtle_atr_window = st.number_input("ATR Window (days)", 5, 50, 20, 5)
        turtle_risk_pct   = st.slider("Risk per unit (% of equity)", 0.25, 3.0, 1.0, 0.25) / 100
        turtle_stop_mult  = st.slider("Stop loss (× ATR)", 0.5, 5.0, 2.0, 0.5)
        turtle_pyramid    = st.toggle("Enable Pyramiding", value=True)
        if turtle_pyramid:
            turtle_max_units = st.slider("Max units per market", 1, 6, 4)
            turtle_pyr_step  = st.slider("Pyramid step (× ATR)", 0.25, 1.0, 0.5, 0.25)
        trail_label = "10" if turtle_system == 1 else "20"
        st.caption(f"Trailing exit: {trail_label}-day lowest low (auto)")
        st.markdown("---")

    run_btn = st.button("▶  RUN BACKTEST", use_container_width=True)


# ── HEADER ────────────────────────────────────────────────────
st.markdown("## 🚀 Momentum Engine")
st.caption("NSE Momentum Scanner · Backtester · Monte Carlo · Regime Detection · Simulator")

# ── LANDING ───────────────────────────────────────────────────
if not run_btn and "results" not in st.session_state:
    st.markdown("""
    <div style="background:#161B22;border:1px solid #21262D;border-radius:16px;
                padding:48px;text-align:center;margin-top:24px">
        <div style="font-size:52px;margin-bottom:16px">⚡</div>
        <div style="font-size:22px;font-weight:700;color:#E6EDF3;margin-bottom:8px">
            Configure your strategy in the sidebar, then hit RUN BACKTEST</div>
        <div style="color:#7D8590;font-size:14px;max-width:560px;margin:0 auto 28px">
            Pick an NSE index or upload your own CSV. Yahoo Finance loads automatically.
        </div>
        <div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap">
            <span style="background:#0D4429;color:#3FB950;padding:6px 14px;border-radius:20px;font-size:12px;font-weight:700">📊 20+ Metrics</span>
            <span style="background:#0C2D6B;color:#58A6FF;padding:6px 14px;border-radius:20px;font-size:12px;font-weight:700">🎲 Monte Carlo</span>
            <span style="background:#2D1140;color:#BC8CFF;padding:6px 14px;border-radius:20px;font-size:12px;font-weight:700">🌡️ Regime (4 modes)</span>
            <span style="background:#3D2000;color:#D29922;padding:6px 14px;border-radius:20px;font-size:12px;font-weight:700">🔗 Correlation Filter</span>
            <span style="background:#0D4429;color:#3FB950;padding:6px 14px;border-radius:20px;font-size:12px;font-weight:700">🧪 Period Simulator</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    sh("📄 CSV TEMPLATES")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("**Price CSV (Wide)**")
        wide = pd.DataFrame({"date":["2022-01-03","2022-01-04","2022-01-05"],
            "RELIANCE":[2450.0,2480.0,2510.0],"TCS":[3200.0,3190.0,3230.0],
            "HDFCBANK":[1620.0,1635.0,1618.0],"INFY":[1750.0,1768.0,1745.0]})
        st.dataframe(wide, use_container_width=True, hide_index=True)
        st.download_button("📥 Wide template", wide.to_csv(index=False), "price_wide.csv","text/csv")
    with c2:
        st.markdown("**Price CSV (Long)**")
        long = pd.DataFrame({"date":["2022-01-03","2022-01-03","2022-01-04","2022-01-04"],
            "symbol":["RELIANCE","TCS","RELIANCE","TCS"],"close":[2450.0,3200.0,2480.0,3190.0]})
        st.dataframe(long, use_container_width=True, hide_index=True)
        st.download_button("📥 Long template", long.to_csv(index=False), "price_long.csv","text/csv")
    with c3:
        st.markdown("**Index / Benchmark CSV**")
        idx_t = pd.DataFrame({"date":["2022-01-03","2022-01-04","2022-01-05"],
            "close":[17625.70,17805.25,17812.40]})
        st.dataframe(idx_t, use_container_width=True, hide_index=True)
        st.download_button("📥 Index template", idx_t.to_csv(index=False), "index.csv","text/csv")

    st.markdown("---")
    sh("📋 INDEX CONSTITUENTS")
    sc1,sc2 = st.columns(2)
    with sc1: sel_cat = st.selectbox("Category", list(INDEX_CATEGORIES.keys()), key="lc")
    with sc2: sel_idx = st.selectbox("Index", INDEX_CATEGORIES[sel_cat], key="li")
    members = NSE_INDICES.get(sel_idx,{}).get("components",[])
    if members:
        df_m = pd.DataFrame({"Symbol":[m.replace(".NS","") for m in members],"Yahoo Ticker":members})
        st.dataframe(df_m, use_container_width=True, hide_index=True)
        st.download_button(f"📥 {sel_idx} constituents",
            df_m.to_csv(index=False), f"{sel_idx.replace(' ','_')}_constituents.csv","text/csv")
    st.stop()


# ── RUN ───────────────────────────────────────────────────────
# Skip re-running backtest if results cached and button not pressed
if not run_btn and "results" in st.session_state:
    results   = st.session_state["results"]
    mc        = st.session_state["mc"]
    signals   = st.session_state["signals"]
    prices    = st.session_state["prices"]
    benchmark = st.session_state.get("benchmark", pd.Series(dtype=float))
    regime_df = st.session_state.get("regime_df", None)
    try:
        extra_periods = [int(x.strip()) for x in extra_ma_raw.split(",") if x.strip().isdigit()]
    except Exception:
        extra_periods = []
else:
  prog = st.progress(0,"Starting…")
  try:
    prog.progress(5,"Loading prices…")
    if csv_prices:
        csv_prices.seek(0)
        prices  = parse_csv(csv_prices)
        tickers = list(prices.columns)
    else:
        tickers = idx_info.get("components",[])
        if not tickers:
            st.error("No pre-loaded components. Upload a Price CSV.")
            prog.empty(); st.stop()

        # Detect crypto universe
        _is_crypto = (cat == "Crypto" or
                      any(is_crypto_ticker(t) for t in tickers[:3]))

        if _is_crypto:
            if crypto_source == "CoinGecko (free, slower)":
                # Convert tickers to CoinGecko IDs
                coin_ids = []
                for t in tickers:
                    base = t.replace("-USD","").replace("-INR","").replace("-USDT","").upper()
                    cg_id = COINGECKO_IDS.get(base, base.lower())
                    coin_ids.append(cg_id)
                vs_currency = "inr" if any("INR" in t for t in tickers) else "usd"
                prices = fetch_coingecko(coin_ids, str(start_dt), str(end_dt), vs_currency)
                # Rename columns back to original tickers
                if not prices.empty:
                    rename_map = {cg.upper(): tk for cg, tk in zip(coin_ids, tickers)}
                    prices.columns = [rename_map.get(c, c) for c in prices.columns]

            elif crypto_source == "Binance (USDT pairs)":
                # Convert to USDT pairs: BTC-USD -> BTCUSDT
                binance_syms = []
                for t in tickers:
                    base = t.replace("-USD","").replace("-INR","").replace("-USDT","").upper()
                    binance_syms.append(f"{base}USDT")
                prices = fetch_binance(binance_syms, str(start_dt), str(end_dt))
                # Rename back to original tickers
                if not prices.empty:
                    rename_map = {b: t for b, t in zip(binance_syms, tickers)}
                    prices.columns = [rename_map.get(c, c) for c in prices.columns]

            else:  # Yahoo Finance for crypto
                prices = fetch_crypto_yahoo(tickers, str(start_dt), str(end_dt))
        else:
            prices = fetch_yahoo(tickers, str(start_dt), str(end_dt))

    if prices.empty:
        st.error("No price data returned.")
        prog.empty(); st.stop()
    prices = prices.dropna(how="all",axis=1).dropna(how="all",axis=0)
    prog.progress(25,f"Loaded {len(prices.columns)} stocks · {len(prices)} days")

    # Store ticker map in session state for edits
    if "ticker_map" not in st.session_state or st.session_state.get("ticker_map_idx") != idx_name:
        orig = idx_info.get("components",[])
        st.session_state["ticker_map"] = {t.replace(".NS",""): t for t in orig}
        st.session_state["ticker_map_idx"] = idx_name

    prog.progress(30,"Fetching benchmark…")
    if csv_index:
        csv_index.seek(0)
        bm_df = parse_csv(csv_index)
        benchmark = bm_df.iloc[:,0].rename("Index")
    else:
        bm_ticker = idx_info.get("index_ticker","^NSEI")
        benchmark = fetch_index(bm_ticker, str(start_dt), str(end_dt))
        # Retry with Nifty 50 if primary fails
        if benchmark.empty or benchmark.dropna().empty:
            benchmark = fetch_index("^NSEI", str(start_dt), str(end_dt))
        # Last resort: equal-weight portfolio of loaded stocks as benchmark
        if benchmark.empty or benchmark.dropna().empty:
            benchmark = prices.mean(axis=1).rename("Index")

    regime_df = None
    if use_regime and not benchmark.empty:
        prog.progress(38,"Detecting regime…")
        regime_df = detect_regime(benchmark, reg_fast, reg_slow, reg_ma_type, vol_thresh=vol_thresh)

    # Map regime_action label to engine parameter (used by both strategies)
    ra_map = {
        "Scale Exposure":                  "Scale Exposure",
        "Exit per Strategy, No New Entry": "Exit per Strategy",
        "Keep Stocks, No New Entry":       "Keep Stocks No Entry",
        "Exit All, No New Entry":          "Exit All No Entry",
    }

    if strategy_type == "Turtle Trading":
        prog.progress(48,"Fetching OHLCV for Turtle…")
        ohlcv_dict = fetch_ohlcv(tickers, str(start_dt), str(end_dt))
        if not ohlcv_dict:
            st.error("No OHLCV data returned. Try Yahoo Finance or a wider date range.")
            prog.empty(); st.stop()

        prog.progress(62,"Running Turtle backtest…")
        results = run_turtle_backtest(
            ohlcv_dict, benchmark,
            capital=float(capital),
            system=turtle_system,
            atr_window=turtle_atr_window,
            risk_pct=turtle_risk_pct,
            stop_atr_mult=turtle_stop_mult,
            max_units_per_market=turtle_max_units if turtle_pyramid else 1,
            pyramid_atr_step=turtle_pyr_step,
            allow_pyramid=turtle_pyramid,
            txn=txn, slip=slip, rf=rf_rate,
            universe=idx_info.get("components",[]) if not csv_prices else None,
        )
        # Prices for charting
        prices   = pd.DataFrame({s: df["close"] for s, df in ohlcv_dict.items()})
        signals  = results.get("turtle_signals", {})

    else:
        prog.progress(48,"Computing signals…")
        signals = generate_signals(prices, mom_w, entry_fast, entry_slow,
                                   exit_ma_p, ma_type, top_n, ex_rank,
                                   rank_by=rank_by, rf=rf_rate,
                                   exit_ma_type=exit_ma_type)

        prog.progress(62,"Running backtest…")
        results = run_backtest(
            prices, signals, benchmark, regime_df,
            float(capital), rebal, top_n, ex_rank,
            txn, slip, sizing, rf_rate,
            allocation_mode=alloc_mode, sip_amount=float(sip_amount),
            rebal_day=rebal_day,
            use_corr_filter=use_corr, corr_threshold=corr_thresh, corr_window=corr_window,
            regime_action=ra_map.get(regime_action,"Scale Exposure"),
            universe=(idx_info.get("components",[])
                      if not csv_prices else []),
        )

    prog.progress(82,f"Monte Carlo ({mc_n} sims)…")
    mc = run_monte_carlo(results["equity"], results["trades"],
                         n_sim=mc_n, method=mc_method,
                         capital=float(capital), rf=rf_rate)

    try:
        extra_periods = [int(x.strip()) for x in extra_ma_raw.split(",") if x.strip().isdigit()]
    except Exception:
        extra_periods = []

    prog.progress(100,"Done ✅"); prog.empty()

    # Cache in session state so simulator tab doesn't lose data on widget interaction
    st.session_state["results"]   = results
    st.session_state["mc"]        = mc
    st.session_state["signals"]   = signals
    st.session_state["prices"]    = prices
    st.session_state["benchmark"] = benchmark
    st.session_state["regime_df"] = regime_df
    st.session_state["equity"]    = results["equity"]
    st.session_state["trades"]    = results["trades"]
    st.session_state["extra_periods"] = extra_periods

  except Exception as e:
    prog.empty()
    st.error(f"❌ {e}")
    import traceback; st.code(traceback.format_exc())
    st.stop()

m  = results["metrics"]
eq = results["equity"]
bm = results["benchmark"]
tr = results["trades"]
mo = m.get("_monthly",pd.Series())
an = m.get("_annual", pd.Series())
dd = m.get("_drawdown",pd.Series())

# ── HEADLINE ──────────────────────────────────────────────────
st.markdown("---")
hc = st.columns(7)
for col,(lbl,val,suf,pg) in zip(hc,[
    ("CAGR",          fmt(m.get("CAGR (%)",0)),       "%", True),
    ("Index CAGR",    fmt(m.get("Index CAGR (%)",0)), "%", True),
    ("Max Drawdown",  fmt(m.get("Max Drawdown (%)",0)),"%",False),
    ("Sharpe",        fmt(m.get("Sharpe Ratio",0),3), "",  True),
    ("Sortino",       fmt(m.get("Sortino Ratio",0),3),"",  True),
    ("Total P&L",     ("₹"+f"{m.get('Total P&L (₹)',0)/1e5:.1f}"+"L") if abs(m.get('Total P&L (₹)',0))>=1e5 else f"₹{m.get('Total P&L (₹)',0):,.0f}","",True),
    ("Total Trades",  str(m.get("Total Trades",0)),   "",  True),
]):
    with col: mcard(lbl, f"{val}{suf}", pg)

# Strategy summary bar
dl = f" ({rebal_day})" if rebal=="Weekly" else ""
regime_badge = ""
if regime_df is not None and not regime_df.empty:
    cr  = str(regime_df["regime"].iloc[-1]).upper()
    cls = {"BULL":"pill-bull","NEUTRAL":"pill-neutral","BEAR":"pill-bear"}.get(cr,"pill-neutral")
    ra_display = ra_map.get(regime_action, regime_action)
    regime_badge = (f'&nbsp;&nbsp;|&nbsp;&nbsp;Latest Regime: <span class="{cls}">{cr}</span>'
                   + (f'&nbsp;<span style="color:#7D8590;font-size:11px">[Bear: {ra_display}]</span>'
                      if use_regime else ''))

st.markdown(f"""
<div style="background:#161B22;border:1px solid #21262D;border-radius:8px;
            padding:10px 16px;margin-bottom:12px;font-size:13px;color:#8B949E;
            display:flex;flex-wrap:wrap;gap:8px;align-items:center;">
  <span>📅 <b style="color:#E6EDF3">{rebal}{dl}</b></span>
  <span style="color:#30363D">|</span>
  <span>📈 Entry: <b style="color:#E6EDF3">price &gt; {ma_type}({entry_fast}) &gt; {ma_type}({entry_slow})</b></span>
  <span style="color:#30363D">|</span>
  <span>📉 Exit: <b style="color:#E6EDF3">price &lt; {exit_ma_type}({exit_ma_p})</b></span>
  <span style="color:#30363D">|</span>
  <span>🏆 Rank: <b style="color:#E6EDF3">{rank_by}</b></span>
  <span style="color:#30363D">|</span>
  <span>💰 <b style="color:#E6EDF3">{alloc_mode}</b> · <b style="color:#E6EDF3">{sizing}</b></span>
  {regime_badge}
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ── TABS ──────────────────────────────────────────────────────
t1,t2,t3,t4,t5,t6,t7,t8,t9,t10 = st.tabs([
    "📊 Overview","📈 Equity","📅 Returns","📉 Drawdown",
    "🌡️ Regime","🎲 Monte Carlo","📋 Trades",
    "💼 Portfolio","📋 Constituents","🧪 Simulator"])


# ── OVERVIEW ─────────────────────────────────────────────────
with t1:
    l,r = st.columns(2)
    with l:
        sh("RETURNS")
        for k,pg in [("Total Return (%)",True),("CAGR (%)",True),("Index CAGR (%)",True),
                     ("Avg Annual Return (%)",True),("Median Annual Return (%)",True),
                     ("Avg Monthly Return (%)",True)]:
            mcard(k, fmt(m.get(k,0)), pg)
    with r:
        sh("RISK & P&L")
        for k,pg in [("Annual Volatility (%)",False),("Max Drawdown (%)",False),
                     ("Sharpe Ratio",True),("Sortino Ratio",True),
                     ("Realized P&L (₹)",True),("Unrealized P&L (₹)",True),("Total P&L (₹)",True),
                     ("Total Invested (₹)",True)]:
            mcard(k, fmt(m.get(k,0)), pg)
    st.markdown("---")
    l2,r2 = st.columns(2)
    with l2:
        sh("MONTHLY / YEARLY")
        for k,pg in [("Positive Months (%)",True),("Avg +ve Month (%)",True),
                     ("Avg -ve Month (%)",False),("Positive Years (%)",True),
                     ("Avg +ve Year (%)",True),("Avg -ve Year (%)",False)]:
            mcard(k, fmt(m.get(k,0)), pg)
    with r2:
        sh("TRADES")
        for k,pg in [("Total Trades",True),("Positive Trades (%)",True),
                     ("Avg Monthly Churn",True)]:
            mcard(k, fmt(m.get(k,0)), pg)


# ── EQUITY ────────────────────────────────────────────────────
with t2:
    if not eq.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq.index,y=eq,name="Strategy",
            line=dict(color="#00D4AA",width=2),fill="tozeroy",fillcolor="rgba(0,212,170,0.06)"))
        if not bm.empty:
            fig.add_trace(go.Scatter(x=bm.index,y=bm,name=f"Benchmark ({idx_name})",
                line=dict(color="#7D8590",width=1.5,dash="dot")))
        if extra_periods and not prices.empty:
            for p in extra_periods:
                try:
                    ma_avg = ma(prices,p,ma_type).mean(axis=1).reindex(eq.index)
                    ratio  = eq.mean()/ma_avg.mean() if ma_avg.mean()!=0 else 1
                    fig.add_trace(go.Scatter(x=ma_avg.index,y=ma_avg*ratio,
                        name=f"{ma_type}{p}(avg)",line=dict(width=1,dash="dot"),opacity=0.5))
                except Exception: pass
        fig.update_layout(title="Portfolio vs Benchmark",yaxis_title="Value (₹)",height=460,**PLOT)
        st.plotly_chart(fig, use_container_width=True)
        if len(eq)>260:
            rs=eq.pct_change().dropna().rolling(252).apply(
                lambda x:(x.mean()*252-rf_rate)/(x.std()*np.sqrt(252)+1e-9),raw=True)
            fig2=go.Figure(go.Scatter(x=rs.index,y=rs,line=dict(color="#BC8CFF",width=1.5)))
            fig2.add_hline(y=1,line_dash="dash",line_color="#3FB950",annotation_text="Sharpe=1")
            fig2.add_hline(y=0,line_dash="dash",line_color="#F85149")
            fig2.update_layout(title="Rolling 252d Sharpe",height=260,**PLOT)
            st.plotly_chart(fig2, use_container_width=True)


# ── RETURNS ───────────────────────────────────────────────────
with t3:
    if not mo.empty:
        try:
            mo.index=pd.to_datetime(mo.index.to_timestamp() if hasattr(mo.index,"to_timestamp") else mo.index)
            dfm=pd.DataFrame({"y":mo.index.year,"m":mo.index.month,"r":mo.values*100})
            piv=dfm.pivot(index="y",columns="m",values="r")
            piv.columns=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][:len(piv.columns)]
            fig_h=px.imshow(piv,color_continuous_scale=[[0,"#8B1A1A"],[0.5,"#161B22"],[1,"#145A32"]],
                color_continuous_midpoint=0,text_auto=".1f",aspect="auto")
            fig_h.update_layout(title="Monthly Returns (%)",height=340,
                paper_bgcolor="#0D1117",plot_bgcolor="#0D1117",
                font=dict(color="#7D8590"),margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig_h, use_container_width=True)
        except Exception as e:
            st.warning(f"Heatmap: {e}")
    if not an.empty:
        try:
            an.index=pd.to_datetime(an.index.to_timestamp() if hasattr(an.index,"to_timestamp") else an.index)
            fig_a=go.Figure(go.Bar(x=[str(d.year) for d in an.index],y=(an.values*100).round(2),
                marker_color=["#3FB950" if v>=0 else "#F85149" for v in an.values],
                text=[f"{v*100:.1f}%" for v in an.values],textposition="outside"))
            fig_a.update_layout(title="Annual Returns (%)",showlegend=False,height=320,**PLOT)
            st.plotly_chart(fig_a, use_container_width=True)
        except Exception: pass


# ── DRAWDOWN ─────────────────────────────────────────────────
with t4:
    if not dd.empty:
        fig_dd=go.Figure(go.Scatter(x=dd.index,y=dd*100,fill="tozeroy",
            fillcolor="rgba(248,81,73,0.12)",line=dict(color="#F85149",width=1.5)))
        fig_dd.add_hline(y=0,line_color="#30363D")
        fig_dd.update_layout(title="Drawdown (%)",yaxis_title="Drawdown (%)",height=380,**PLOT)
        st.plotly_chart(fig_dd, use_container_width=True)
        dc1,dc2,dc3=st.columns(3)
        with dc1: mcard("Max Drawdown",fmt(dd.min()*100),False)
        with dc2: mcard("Avg Drawdown",fmt(dd.mean()*100),False)
        with dc3: mcard("Days below -10%",str((dd<-0.10).sum()),False)


# ── REGIME ────────────────────────────────────────────────────
with t5:
    if regime_df is not None and not regime_df.empty:
        cr  = str(regime_df["regime"].iloc[-1]).upper()
        ce  = regime_df["exposure"].iloc[-1]
        cls = {"BULL":"pill-bull","NEUTRAL":"pill-neutral","BEAR":"pill-bear"}.get(cr,"pill-neutral")
        st.markdown(f'**Current:** <span class="{cls}">{cr}</span> · Exposure `{ce*100:.0f}%` · Action: `{regime_action}`',
                    unsafe_allow_html=True)
        fig_r=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.7,0.3],vertical_spacing=0.06)
        if not bm.empty:
            fig_r.add_trace(go.Scatter(x=bm.index,y=bm,name="Benchmark",
                line=dict(color="#00D4AA",width=1.5)),row=1,col=1)
        colors={"BULL":"#238636","NEUTRAL":"#1F6FEB","BEAR":"#DA3633"}
        rc=regime_df["regime"]; prev,st0=rc.iloc[0],regime_df.index[0]
        for dt2,reg in rc.items():
            if reg!=prev:
                fig_r.add_vrect(x0=st0,x1=dt2,fillcolor=colors.get(prev.upper(),"#161B22"),
                    opacity=0.12,layer="below",line_width=0,row=1,col=1)
                st0,prev=dt2,reg
        fig_r.add_vrect(x0=st0,x1=regime_df.index[-1],
            fillcolor=colors.get(prev.upper(),"#161B22"),opacity=0.12,layer="below",line_width=0,row=1,col=1)
        fig_r.add_trace(go.Scatter(x=regime_df.index,y=regime_df["exposure"]*100,
            name="Exposure %",line=dict(color="#D29922",width=1.5),
            fill="tozeroy",fillcolor="rgba(210,153,34,0.1)"),row=2,col=1)
        fig_r.update_layout(height=500,paper_bgcolor="#0D1117",plot_bgcolor="#0D1117",
            font=dict(color="#7D8590"),legend=dict(bgcolor="#161B22"),
            margin=dict(l=50,r=20,t=40,b=40))
        fig_r.update_xaxes(gridcolor="#21262D"); fig_r.update_yaxes(gridcolor="#21262D")
        st.plotly_chart(fig_r, use_container_width=True)
        rc2=regime_df["regime"].value_counts()
        fig_p=go.Figure(go.Pie(labels=rc2.index,values=rc2.values,hole=0.5,
            marker=dict(colors=["#238636","#1F6FEB","#DA3633"])))
        fig_p.update_layout(title="Regime Distribution",height=280,
            paper_bgcolor="#0D1117",font=dict(color="#7D8590"))
        st.plotly_chart(fig_p, use_container_width=True)
    else:
        st.info("Enable Regime Detection in the sidebar.")


# ── MONTE CARLO ───────────────────────────────────────────────
with t6:
    if mc.get("n_sim",0)>0:
        sh(f"MONTE CARLO — {mc['n_sim']:,} SIMS · {mc_method.upper()}")
        x=list(range(mc["n"]))
        fig_mc=go.Figure()
        for lo,hi,clr in [(5,95,"rgba(0,212,170,0.07)"),(25,75,"rgba(0,212,170,0.13)")]:
            fig_mc.add_trace(go.Scatter(x=x+x[::-1],
                y=list(mc["pcts"][hi])+list(mc["pcts"][lo][::-1]),
                fill="toself",fillcolor=clr,line=dict(width=0),name=f"{lo}–{hi}th pct"))
        fig_mc.add_trace(go.Scatter(x=x,y=mc["pcts"][50],name="Median",line=dict(color="#00D4AA",width=2)))
        fig_mc.add_trace(go.Scatter(x=x[:len(eq)],y=eq.values,name="Actual",
            line=dict(color="#FFFFFF",width=1.5,dash="dot")))
        fig_mc.update_layout(title="Simulated Equity Paths",yaxis_title="Value (₹)",height=420,**PLOT)
        st.plotly_chart(fig_mc, use_container_width=True)
        ml,mr=st.columns(2)
        with ml:
            sh("STATS")
            for k,v in mc["stats"].items():
                pg=any(w in k for w in ["Profit","CAGR","95th","Median Final"])
                mcard(k, fmt(float(v),0 if "₹" in k else 2), pg)
        with mr:
            fig_d=go.Figure(go.Histogram(x=mc["final"],nbinsx=60,
                marker=dict(color="#00D4AA",opacity=0.7,line=dict(width=0))))
            fig_d.add_vline(x=float(capital),line_dash="dash",line_color="#F85149",annotation_text="Initial")
            fig_d.add_vline(x=float(np.median(mc["final"])),line_dash="dash",line_color="#3FB950",annotation_text="Median")
            fig_d.update_layout(title="Final Value Distribution",height=300,**PLOT,showlegend=False)
            st.plotly_chart(fig_d, use_container_width=True)
            fig_c=go.Figure(go.Histogram(x=mc["cagr_dist"],nbinsx=50,
                marker=dict(color="#BC8CFF",opacity=0.7,line=dict(width=0))))
            fig_c.add_vline(x=float(m.get("CAGR (%)",0)),line_dash="dash",line_color="#00D4AA",annotation_text="Actual")
            fig_c.update_layout(title="Simulated CAGR (%)",height=260,**PLOT)
            st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.info("Not enough data for Monte Carlo.")


# ── TRADES ────────────────────────────────────────────────────
with t7:
    if not tr.empty:
        sells=tr[tr["action"].isin(["SELL","TRIM"])].copy() if "action" in tr.columns else tr.copy()
        if not sells.empty and "pnl" in sells.columns:
            fig_pnl=go.Figure(go.Bar(
                x=sells["date"].astype(str) if "date" in sells.columns else sells.index.astype(str),
                y=sells["pnl"],
                marker_color=["#3FB950" if p>=0 else "#F85149" for p in sells["pnl"]]))
            fig_pnl.update_layout(title="P&L per Trade (Sells)",yaxis_title="P&L (₹)",height=280,**PLOT)
            st.plotly_chart(fig_pnl, use_container_width=True)
        show_tr=tr.copy()
        if "date" in show_tr.columns:
            show_tr["date"]=pd.to_datetime(show_tr["date"]).dt.date
        st.dataframe(show_tr, use_container_width=True, height=400)
    else:
        st.info("No trades. Try reducing MA periods (e.g. fast=50, slow=200).")


# ── TURTLE SIGNALS CHART (shown only in Turtle mode) ─────────
if strategy_type == "Turtle Trading" and signals:
    with t2:  # inject into Equity tab
        st.markdown("---")
        sh("📊 DONCHIAN CHANNEL — SELECT SYMBOL")
        turtle_sym = st.selectbox("Symbol", list(signals.keys()), key="turtle_sym_sel")
        if turtle_sym and turtle_sym in signals:
            sig_df = signals[turtle_sym]
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(x=sig_df.index, y=sig_df["close"],
                name="Price", line=dict(color="#00D4AA", width=1.5)))
            fig_t.add_trace(go.Scatter(x=sig_df.index, y=sig_df["don_high"],
                name=f"Entry ({20 if turtle_system==1 else 55}d high)",
                line=dict(color="#3FB950", width=1, dash="dot")))
            fig_t.add_trace(go.Scatter(x=sig_df.index, y=sig_df["don_trail"],
                name=f"Trail exit ({10 if turtle_system==1 else 20}d low)",
                line=dict(color="#F85149", width=1, dash="dot")))
            # Mark entries
            entries = sig_df[sig_df["entry_signal"]]
            if not entries.empty:
                fig_t.add_trace(go.Scatter(x=entries.index, y=entries["close"],
                    mode="markers", name="Entry signal",
                    marker=dict(color="#3FB950", size=8, symbol="triangle-up")))
            fig_t.update_layout(title=f"Donchian Channel — {turtle_sym}",
                yaxis_title="Price", height=400, **PLOT)
            st.plotly_chart(fig_t, use_container_width=True)

            # ATR chart
            fig_atr = go.Figure(go.Scatter(x=sig_df.index, y=sig_df["atr"],
                line=dict(color="#D29922", width=1.5), name="ATR"))
            fig_atr.update_layout(title="ATR (Volatility)", height=220,
                yaxis_title="ATR", **PLOT)
            st.plotly_chart(fig_atr, use_container_width=True)

# ── PORTFOLIO ─────────────────────────────────────────────────
with t8:
    sh("CURRENT HOLDINGS")
    fh=results.get("final_holdings",{}); fpx=results.get("final_prices",pd.Series())
    if fh:
        rows=[]
        for sym,sh_ct in fh.items():
            p=float(fpx[sym]) if sym in fpx.index else 0
            ep=0  # not stored separately here
            rows.append({"Symbol":sym,"Shares":round(sh_ct,4),
                "Last Price (₹)":round(p,2),"Value (₹)":round(sh_ct*p,2)})
        dfh=pd.DataFrame(rows).sort_values("Value (₹)",ascending=False)
        tv=dfh["Value (₹)"].sum()
        dfh["Weight (%)"]=((dfh["Value (₹)"]/tv*100) if tv else 0).round(2)
        hl,hr=st.columns([2,1])
        with hl: st.dataframe(dfh, use_container_width=True)
        with hr:
            if not dfh.empty:
                fig_pie=go.Figure(go.Pie(labels=dfh["Symbol"],values=dfh["Value (₹)"],
                    hole=0.45,marker=dict(colors=px.colors.qualitative.Dark24)))
                fig_pie.update_layout(height=300,paper_bgcolor="#0D1117",
                    font=dict(color="#7D8590"),margin=dict(l=10,r=10,t=20,b=10))
                st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No holdings at end of backtest.")

    st.markdown("---")
    sh("⚡ LIVE SCANNER")
    if strategy_type == "Turtle Trading":
        # Turtle scanner — show current breakout status
        if isinstance(signals, dict) and signals:
            scan_rows = []
            last_dt = prices.index[-1] if not prices.empty else None
            for sym, sig_df in signals.items():
                if sig_df.empty or last_dt not in sig_df.index:
                    continue
                row = sig_df.loc[last_dt]
                scan_rows.append({
                    "Symbol":     sym,
                    "Price":      round(float(row["close"]),2),
                    "ATR":        round(float(row["atr"]),2) if not pd.isna(row["atr"]) else 0,
                    "Entry Level":round(float(row["don_high"]),2) if not pd.isna(row["don_high"]) else 0,
                    "Trail Stop": round(float(row["don_trail"]),2) if not pd.isna(row["don_trail"]) else 0,
                    "Signal":     "🟢 BUY" if row["entry_signal"] else ("🔴 EXIT" if row["exit_trail"] else "⏳ HOLD"),
                })
            if scan_rows:
                df_ts = pd.DataFrame(scan_rows).sort_values("Signal")
                st.dataframe(df_ts, use_container_width=True, height=380)
    else:
        ls=signals["filtered"].iloc[-1].dropna().sort_values(ascending=False)
        lr=signals["ranks"].iloc[-1].dropna().sort_values()
        le=signals["entry_filter"].iloc[-1]
        lx=signals["exit"].iloc[-1]
        if ls.empty:
            st.warning("No stocks pass entry filter. Try smaller MA periods.")
        else:
            df_scan=pd.DataFrame({
                "Rank":     lr.reindex(ls.index).fillna(999).astype(int).values,
                "Symbol":   ls.index,
                "Score":    ls.values.round(4),
                "Entry":    le.reindex(ls.index).map({True:"✅",False:"❌"}).fillna("❌").values,
                "Exit":     lx.reindex(ls.index).map({True:"🚪",False:"—"}).fillna("—").values,
                "Status":   ["⭐ HOLD" if i<top_n else "" for i in range(len(ls))],
            }).head(30)
            st.dataframe(df_scan.set_index("Rank"), use_container_width=True, height=380)
    st.caption(f"As of {prices.index[-1].date()} · {len(prices.columns)} stocks")


# ── CONSTITUENTS (with ticker editor) ─────────────────────────
with t9:
    sh(f"INDEX CONSTITUENTS — {idx_name}")
    members = idx_info.get("components",[])

    st.caption("✏️ Fix any ticker that failed to load — edit the Yahoo Ticker column and click Apply.")

    tmap = st.session_state.get("ticker_map", {})

    if members:
        rows_c = []
        for sym_ns in members:
            sym = sym_ns.replace(".NS","")
            current_ticker = tmap.get(sym, sym_ns)
            has_data = sym in prices.columns
            rows_c.append({
                "Symbol":         sym,
                "Yahoo Ticker":   current_ticker,
                "Data Loaded":    "✅" if has_data else "❌ No data",
            })
        df_c = pd.DataFrame(rows_c)

        edited = st.data_editor(
            df_c,
            column_config={
                "Symbol":       st.column_config.TextColumn("Symbol", disabled=True),
                "Yahoo Ticker": st.column_config.TextColumn("Yahoo Ticker (editable)"),
                "Data Loaded":  st.column_config.TextColumn("Status", disabled=True),
            },
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            key="constituent_editor",
        )

        if st.button("🔄 Apply Ticker Changes & Refetch", use_container_width=True):
            new_tickers = edited["Yahoo Ticker"].tolist()
            syms        = edited["Symbol"].tolist()
            new_map     = {s: t for s,t in zip(syms, new_tickers)}
            st.session_state["ticker_map"] = new_map
            with st.spinner("Refetching with updated tickers…"):
                new_px = fetch_yahoo(new_tickers, str(start_dt), str(end_dt))
            if not new_px.empty:
                st.success(f"Loaded {len(new_px.columns)} stocks with updated tickers. Re-run the backtest.")
                st.dataframe(new_px.tail(5), use_container_width=True)
            else:
                st.error("No data returned with updated tickers.")

        st.download_button(f"📥 {idx_name} constituents",
            df_c.to_csv(index=False), f"{idx_name.replace(' ','_')}_constituents.csv","text/csv")
    else:
        st.info("No pre-loaded list. Stocks from uploaded CSV:")
        if not prices.empty:
            st.dataframe(pd.DataFrame({"Symbol":list(prices.columns)}), use_container_width=True)


# ── SIMULATOR ─────────────────────────────────────────────────
with t10:
    if strategy_type == "Turtle Trading":
        sh("🧪 DAILY TURTLE SIMULATOR")
        st.caption("Step through every trading day — open positions, ATR stops, daily P&L.")

        if eq.empty:
            st.info("Run the backtest first.")
        else:
            all_sim_dates = eq.index.tolist()
            yrs_t  = sorted({d.year for d in all_sim_dates})
            yopts_t= ["All"] + [str(y) for y in yrs_t]

            if "tsim_year" not in st.session_state:  st.session_state["tsim_year"] = "All"
            if "tsim_date" not in st.session_state:  st.session_state["tsim_date"] = None

            tc1, tc2 = st.columns([1, 3])
            with tc1:
                t_year = st.selectbox("Filter by year", yopts_t,
                    index=yopts_t.index(st.session_state["tsim_year"])
                          if st.session_state["tsim_year"] in yopts_t else 0,
                    key="tsim_year_box")
                st.session_state["tsim_year"] = t_year

            t_dates = [d for d in all_sim_dates if d.year==int(t_year)] if t_year!="All" else all_sim_dates
            if not t_dates:
                st.info("No data for selected year.")
            else:
                t_labels = [d.strftime("%Y-%m-%d (%a)") for d in t_dates]
                prev_td  = st.session_state.get("tsim_date")
                def_idx  = t_labels.index(prev_td) if prev_td in t_labels else len(t_labels)-1
                with tc2:
                    sel_t_label = st.select_slider("Select Date", options=t_labels,
                        value=t_labels[def_idx], key="tsim_date_slider")
                st.session_state["tsim_date"] = sel_t_label
                sel_dt = pd.Timestamp(sel_t_label[:10])

                # Trades to/on date
                tr_to   = tr[pd.to_datetime(tr["date"]) <= sel_dt] if not tr.empty and "date" in tr.columns else pd.DataFrame()
                tr_today= tr[pd.to_datetime(tr["date"]) == sel_dt] if not tr.empty and "date" in tr.columns else pd.DataFrame()
                realized_to = float(tr_to[tr_to["action"]=="SELL"]["pnl"].sum()) if not tr_to.empty and "pnl" in tr_to.columns else 0.0

                port_to = float(eq.loc[sel_dt]) if sel_dt in eq.index else 0.0
                eq_to   = eq[:sel_dt]
                dd_to   = ((eq_to - eq_to.cummax()) / eq_to.cummax()).min() * 100 if not eq_to.empty else 0.0

                # Reconstruct open positions from trade log
                open_pos = {}
                if not tr_to.empty and "symbol" in tr_to.columns:
                    for sym in tr_to["symbol"].unique():
                        b = tr_to[(tr_to["symbol"]==sym) & (tr_to["action"].isin(["BUY","ADD"]))]
                        s = tr_to[(tr_to["symbol"]==sym) & (tr_to["action"]=="SELL")]
                        net_sh = b["shares"].sum() - (s["shares"].sum() if not s.empty else 0)
                        if net_sh > 0.001:
                            avg_entry = (b["price"]*b["shares"]).sum() / b["shares"].sum()
                            curr_px   = 0.0
                            atr_now   = 0.0
                            if isinstance(signals, dict) and sym in signals:
                                sdf = signals[sym]
                                nearest = sdf[sdf.index <= sel_dt]
                                if not nearest.empty:
                                    curr_px = float(nearest["close"].iloc[-1])
                                    atr_now = float(nearest["atr"].iloc[-1]) if not pd.isna(nearest["atr"].iloc[-1]) else 0
                            mkt = net_sh * curr_px
                            cost= net_sh * avg_entry
                            open_pos[sym] = {
                                "units":     len(b),
                                "shares":    round(net_sh, 4),
                                "avg_entry": round(avg_entry, 2),
                                "curr_price":round(curr_px, 2),
                                "mkt_value": round(mkt, 2),
                                "unrealised":round(mkt - cost, 2),
                                "atr":       round(atr_now, 2),
                                "stop_price":round(avg_entry - turtle_stop_mult * atr_now, 2),
                            }

                unrealized_to = sum(p["unrealised"] for p in open_pos.values())
                sc1,sc2,sc3,sc4,sc5 = st.columns(5)
                with sc1: mcard("Portfolio Value",  f"₹{port_to:,.0f}", True)
                with sc2: mcard("Open Positions",   str(len(open_pos)), True)
                with sc3: mcard("Realized P&L",     f"₹{realized_to:,.0f}", True)
                with sc4: mcard("Unrealized P&L",   f"₹{unrealized_to:,.0f}", True)
                with sc5: mcard("Max DD to date",   f"{dd_to:.1f}%", False)
                st.markdown("---")

                h_col, t_col = st.columns([3, 2])
                with h_col:
                    sh("OPEN POSITIONS")
                    if open_pos:
                        df_op = pd.DataFrame([
                            {"Symbol":sym,"Units":v["units"],"Shares":v["shares"],
                             "Avg Entry (₹)":v["avg_entry"],"Current (₹)":v["curr_price"],
                             "Mkt Value (₹)":v["mkt_value"],"Unrealized P&L (₹)":v["unrealised"],
                             "ATR":v["atr"],"Stop (₹)":v["stop_price"]}
                            for sym,v in open_pos.items()
                        ]).sort_values("Mkt Value (₹)", ascending=False)
                        def _cpnl(val):
                            return f"color:{'#3FB950' if val>=0 else '#F85149'}"
                        st.dataframe(df_op.style.map(_cpnl, subset=["Unrealized P&L (₹)"]),
                                     use_container_width=True, hide_index=True)
                    else:
                        st.info("No open positions on this date.")

                with t_col:
                    sh("TRADES TODAY")
                    if not tr_today.empty:
                        cols_show = [c for c in ["action","symbol","price","shares","value","pnl","reason"] if c in tr_today.columns]
                        def _cact(val):
                            return ("color:#3FB950" if val in ("BUY","ADD") else ("color:#F85149" if val=="SELL" else ""))
                        st.dataframe(tr_today[cols_show].style.map(_cact, subset=["action"]),
                                     use_container_width=True, hide_index=True)
                    else:
                        st.info("No trades on this date.")

                st.markdown("---")
                sh("EQUITY TO DATE")
                if not eq_to.empty:
                    fig_s = go.Figure()
                    fig_s.add_trace(go.Scatter(x=eq_to.index, y=eq_to,
                        line=dict(color="#00D4AA",width=2), fill="tozeroy",
                        fillcolor="rgba(0,212,170,0.06)", name="Portfolio"))
                    if sel_dt in eq_to.index:
                        fig_s.add_trace(go.Scatter(x=[sel_dt], y=[eq_to[sel_dt]],
                            mode="markers", marker=dict(color="#D29922",size=12,symbol="diamond"),
                            name="Selected"))
                    fig_s.update_layout(height=280, **PLOT)
                    st.plotly_chart(fig_s, use_container_width=True)

    else:
        # ── MOMENTUM SIMULATOR ────────────────────────────────
        sh("🧪 PERIOD-BY-PERIOD SIMULATOR")
        st.caption("Step through each rebalance period to see holdings, trades, P&L.")

        snapshots = results.get("period_snapshots", [])
        if not snapshots:
            st.info("No rebalance periods captured. Run the backtest first.")
        else:
            snap_dates   = [s["date"] for s in snapshots]
            snap_labels  = [d.strftime("%Y-%m-%d (%a)") for d in snap_dates]
            years_avail  = sorted({d.year for d in snap_dates})
            year_options = ["All"] + [str(y) for y in years_avail]

            if "sim_year" not in st.session_state:  st.session_state["sim_year"] = "All"
            if "sim_label" not in st.session_state: st.session_state["sim_label"] = None

            sim_c1, sim_c2 = st.columns([1, 3])
            with sim_c1:
                sel_year = st.selectbox("Filter by year", year_options,
                    index=year_options.index(st.session_state["sim_year"])
                          if st.session_state["sim_year"] in year_options else 0,
                    key="sim_year_box")
                st.session_state["sim_year"] = sel_year

            filtered_snaps = [(i,s) for i,s in enumerate(snapshots)
                              if s["date"].year==int(sel_year)] if sel_year!="All"                              else list(enumerate(snapshots))

            if not filtered_snaps:
                st.info("No periods in selected year.")
            else:
                f_labels  = [snapshots[i]["date"].strftime("%Y-%m-%d (%a)") for i,_ in filtered_snaps]
                prev_lbl  = st.session_state.get("sim_label")
                def_idx   = f_labels.index(prev_lbl) if prev_lbl in f_labels else len(f_labels)//2
                with sim_c2:
                    sel_label = st.select_slider("Select Rebalance Period", options=f_labels,
                        value=f_labels[def_idx], key="sim_period_slider")
                st.session_state["sim_label"] = sel_label

                sel_idx_s = f_labels.index(sel_label)
                orig_idx, snap = filtered_snaps[sel_idx_s]
                snap_dt = snap["date"]

                tr_to_date  = tr[pd.to_datetime(tr["date"]) <= snap_dt] if not tr.empty and "date" in tr.columns else pd.DataFrame()
                realized_pnl= float(tr_to_date[tr_to_date["action"].isin(["SELL","TRIM"])]["pnl"].sum()) if not tr_to_date.empty and "pnl" in tr_to_date.columns else 0.0
                unrealized_pnl = sum(v["unrealised"] for v in snap["holdings"].values())

                eq_to_date = eq[:snap_dt]
                max_dd_to  = ((eq_to_date - eq_to_date.cummax()) / eq_to_date.cummax()).min() * 100 if not eq_to_date.empty else 0.0

                sc1,sc2,sc3,sc4,sc5 = st.columns(5)
                with sc1: mcard("Portfolio Value", f"₹{snap['portfolio_val']:,.0f}", True)
                with sc2: mcard("Cash",            f"₹{snap['cash']:,.0f}", True)
                with sc3: mcard("Realized P&L",    f"₹{realized_pnl:,.0f}", True)
                with sc4: mcard("Unrealized P&L",  f"₹{unrealized_pnl:,.0f}", True)
                with sc5: mcard("Max DD to date",  f"{max_dd_to:.1f}%", False)

                reg_c = snap.get("regime","—")
                cls   = {"BULL":"pill-bull","NEUTRAL":"pill-neutral","BEAR":"pill-bear"}.get(reg_c,"pill-neutral")
                action_info = f" · Action: <b>{regime_action}</b>" if use_regime else ""
                st.markdown(f'Regime: <span class="{cls}">{reg_c}</span>{action_info}', unsafe_allow_html=True)
                st.markdown("---")

                h_col, t_col = st.columns([3, 2])
                with h_col:
                    sh("HOLDINGS ON THIS DATE")
                    if snap["holdings"]:
                        df_snap = pd.DataFrame([
                            {"Symbol":sym,"Shares":round(v["shares"],2),
                             "Cost Basis/sh (₹)":v["entry_price"],
                             "Current Price (₹)":v["curr_price"],
                             "Cost Value (₹)":v.get("cost_value",round(v["shares"]*v["entry_price"],2)),
                             "Market Value (₹)":v["mkt_value"],
                             "Unrealized P&L (₹)":v["unrealised"],
                             "Weight (%)":v["weight_pct"]}
                            for sym,v in snap["holdings"].items()
                        ]).sort_values("Market Value (₹)", ascending=False)
                        def color_pnl(val):
                            return f"color:{'#3FB950' if val>=0 else '#F85149'}"
                        st.dataframe(df_snap.style.map(color_pnl, subset=["Unrealized P&L (₹)"]),
                                     use_container_width=True, hide_index=True)
                    else:
                        st.info("No holdings on this date.")

                with t_col:
                    sh("TRADES ON THIS DATE")
                    trades_today = [r for r in (snap.get("trades_today") or []) if r.get("date")==snap_dt]
                    if trades_today:
                        df_tt = pd.DataFrame(trades_today)[[c for c in ["action","symbol","price","shares","value","pnl"] if c in pd.DataFrame(trades_today).columns]]
                        def color_action(val):
                            return "color:#3FB950" if val=="BUY" else ("color:#F85149" if val in ("SELL","TRIM") else "")
                        st.dataframe(df_tt.style.map(color_action, subset=["action"]),
                                     use_container_width=True, hide_index=True)
                    else:
                        st.info("No trades on this rebalance date.")

                st.markdown("---")
                sh("EQUITY CURVE UP TO THIS DATE")
                eq_slice = eq[:snap_dt]
                if not eq_slice.empty:
                    fig_s = go.Figure()
                    fig_s.add_trace(go.Scatter(x=eq_slice.index, y=eq_slice,
                        line=dict(color="#00D4AA",width=2), fill="tozeroy",
                        fillcolor="rgba(0,212,170,0.06)", name="Portfolio"))
                    if snap_dt in eq_slice.index:
                        fig_s.add_trace(go.Scatter(x=[snap_dt], y=[eq_slice[snap_dt]],
                            mode="markers", marker=dict(color="#D29922",size=12,symbol="diamond"),
                            name="Selected date"))
                    fig_s.update_layout(height=280, **PLOT)
                    st.plotly_chart(fig_s, use_container_width=True)

# ── EXPORT ────────────────────────────────────────────────────
st.markdown("---")
dl_label = f" ({rebal_day})" if rebal=="Weekly" else ""
cfg_export = {
    "Index":idx_name,"Start":str(start_dt),"End":str(end_dt),
    "Rebalance":f"{rebal}{dl_label}","MA Type":ma_type,
    "Entry":f"price>{ma_type}({entry_fast}) AND {ma_type}({entry_fast})>{ma_type}({entry_slow})",
    "Exit":f"price<{exit_ma_type}({exit_ma_p})",
    "Rank by":rank_by,"Top N":top_n,"Exit Rank":ex_rank,
    "Sizing":sizing,"Allocation":alloc_mode,
    "SIP Amt":f"₹{sip_amount:,}" if alloc_mode=="SIP" else "N/A",
    "Portfolio":f"₹{capital:,.0f}",
    "Corr Filter":f"On (thresh={corr_thresh})" if use_corr else "Off",
    "Regime Action":regime_action if use_regime else "Off",
    "Monte Carlo":f"{mc_n} sims · {mc_method}",
}
try:
    xl=build_excel(results,cfg_export)
    st.download_button("📥 Download Full Excel Report",data=xl,
        file_name=f"momentum_{idx_name.replace(' ','_')}_{date.today()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True)
except Exception as e:
    st.warning(f"Excel export error: {e}")
