"""
🚀 Momentum Engine v3 — Streamlit Cloud
MA-based entry/exit · Reinvestment/SIP/Fixed · Sharpe/Return/Vol ranking
"""
import io, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import date, timedelta

from indices import NSE_INDICES, INDEX_CATEGORIES
from engine import (
    fetch_yahoo, fetch_index, parse_csv, ma,
    generate_signals, run_backtest, detect_regime, run_monte_carlo,
)

warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────
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
.stTabs [data-baseweb="tab"]{color:#7D8590;font-weight:600;border-radius:7px;padding:6px 14px;}
.stTabs [aria-selected="true"]{background:#0D1117!important;color:#00D4AA!important;}
.mc{background:#161B22;border:1px solid #21262D;border-radius:10px;padding:14px 18px;margin-bottom:8px;}
.ml{font-size:10px;color:#7D8590;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:3px;}
.mv{font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700;color:#E6EDF3;}
.mv.pos{color:#3FB950;}.mv.neg{color:#F85149;}.mv.neu{color:#58A6FF;}
.sh{font-size:11px;font-weight:700;color:#00D4AA;text-transform:uppercase;letter-spacing:2px;
    padding:6px 0 8px;border-bottom:1px solid #21262D;margin-bottom:14px;margin-top:4px;}
.pill-bull{background:#0D4429;color:#3FB950;padding:3px 12px;border-radius:20px;font-size:12px;font-weight:700;display:inline-block;}
.pill-neutral{background:#1C2A3A;color:#58A6FF;padding:3px 12px;border-radius:20px;font-size:12px;font-weight:700;display:inline-block;}
.pill-bear{background:#3D1616;color:#F85149;padding:3px 12px;border-radius:20px;font-size:12px;font-weight:700;display:inline-block;}
.info-box{background:#161B22;border:1px solid #21262D;border-radius:10px;padding:16px 20px;margin-bottom:12px;}
</style>
""", unsafe_allow_html=True)

PLOT = dict(
    paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
    font=dict(family="Inter", color="#7D8590", size=12),
    xaxis=dict(gridcolor="#21262D", linecolor="#30363D", zeroline=False),
    yaxis=dict(gridcolor="#21262D", linecolor="#30363D", zeroline=False),
    legend=dict(bgcolor="#161B22", bordercolor="#30363D", borderwidth=1),
    margin=dict(l=50,r=20,t=40,b=40),
)

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


# ── EXCEL BUILDER ─────────────────────────────────────────────
def build_excel(results, cfg):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        wb  = w.book
        hf  = wb.add_format({"bold":True,"bg_color":"#161B22","font_color":"#00D4AA","border":1})
        pf  = wb.add_format({"font_color":"#3FB950","bold":True})
        nf  = wb.add_format({"font_color":"#F85149","bold":True})
        rf2 = wb.add_format({"font_color":"#E6EDF3"})

        ws = wb.add_worksheet("Summary")
        ws.set_column("A:A",34); ws.set_column("B:B",22)
        ws.write(0,0,"Momentum Engine Backtest Report",
                 wb.add_format({"bold":True,"font_size":14,"font_color":"#00D4AA"}))
        ws.write(1,0,f"Generated: {date.today()}",rf2)
        r=3; ws.write(r,0,"CONFIGURATION",hf); r+=1
        for k,v in cfg.items():
            ws.write(r,0,k,rf2); ws.write(r,1,str(v),rf2); r+=1
        r+=1; ws.write(r,0,"PERFORMANCE METRICS",hf); r+=1
        m = results["metrics"]
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
    st.caption("NSE Momentum Scanner & Backtester v3")
    st.markdown("---")

    sh("Universe")
    cat      = st.selectbox("Category", list(INDEX_CATEGORIES.keys()), label_visibility="collapsed")
    idx_name = st.selectbox("Index", INDEX_CATEGORIES[cat])
    idx_info = NSE_INDICES.get(idx_name, {})

    csv_prices = st.file_uploader("📄 Upload Price CSV", type=["csv"],
        help="Wide: date+symbol columns  |  Long: date,symbol,close")
    csv_index  = st.file_uploader("📄 Upload Index CSV (optional)", type=["csv"],
        help="Two columns: date, close — for custom benchmark")

    if csv_prices: st.success("Price CSV loaded ✓")
    if csv_index:  st.success("Index CSV loaded ✓")
    st.markdown("---")

    sh("Date Range & Rebalance")
    c1,c2 = st.columns(2)
    with c1: start_dt = st.date_input("Start", value=date.today()-timedelta(days=5*365), label_visibility="collapsed")
    with c2: end_dt   = st.date_input("End",   value=date.today(), label_visibility="collapsed")
    rebal = st.selectbox("Rebalance Frequency", ["Weekly","Monthly","Quarterly","Yearly"])
    st.markdown("---")

    sh("Moving Averages (Entry & Exit)")
    ma_type = st.selectbox("MA Type", ["EMA","SMA"])

    st.caption("Entry Condition")
    st.markdown("*Price > MA(fast)  AND  MA(fast) > MA(slow)*")
    entry_fast = st.number_input("Fast MA period", min_value=5,  max_value=500, value=50,  step=5)
    entry_slow = st.number_input("Slow MA period", min_value=10, max_value=500, value=200, step=10)

    st.caption("Exit Condition")
    st.markdown("*Price < Exit MA*")
    exit_ma_p  = st.number_input("Exit MA period", min_value=5, max_value=500, value=20, step=5)

    st.caption("Extra display MAs on chart (comma-separated)")
    extra_ma_raw = st.text_input("e.g. 50,100,200", value="20,50,200")
    st.markdown("---")

    sh("Ranking Criteria")
    rank_by = st.selectbox("Rank stocks by",
        ["Momentum", "Sharpe Ratio", "Return %", "Low Volatility"])
    st.caption("Lookback Windows & Weights")
    lb_opts = st.multiselect("Lookback days", [21,30,60,90,120,180,252], default=[60,90,120,252])
    if not lb_opts: lb_opts = [60,90,120,252]
    wcols = st.columns(min(len(lb_opts),4))
    raw_w = {}
    for i,lb in enumerate(lb_opts):
        with wcols[i%4]:
            raw_w[lb] = st.number_input(f"{lb}d",0.0,1.0,round(1/len(lb_opts),2),0.05,key=f"w{lb}")
    tw = sum(raw_w.values()) or 1
    mom_w = {k: v/tw for k,v in raw_w.items()}
    st.markdown("---")

    sh("Portfolio & Allocation")
    top_n   = st.slider("Stocks to hold", 1, 40, 10)
    ex_rank = st.slider("Exit rank threshold", top_n, 60, min(top_n*2, 20))
    capital = st.number_input("Initial Portfolio (₹)", value=1_000_000, step=100_000, format="%d")

    alloc_mode = st.selectbox("Allocation Mode",
        ["Reinvestment","SIP","Fixed"],
        help="Reinvestment=compound full portfolio | SIP=add fixed amount each period | Fixed=only invest original capital")
    sip_amount = 0
    if alloc_mode == "SIP":
        sip_amount = st.number_input("SIP Amount per period (₹)", value=50_000, step=10_000, format="%d")

    sizing  = st.selectbox("Position Sizing", ["Inverse Volatility","Equal Weight"])
    txn     = st.slider("Transaction Cost (%)", 0.0, 1.0, 0.1) / 100
    slip    = st.slider("Slippage (%)", 0.0, 1.0, 0.1) / 100
    rf_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 6.5) / 100
    st.markdown("---")

    sh("Regime Detection")
    use_regime  = st.toggle("Enable Regime Filter", value=True)
    reg_fast    = st.number_input("Regime fast MA", 20, 200, 50, 10, disabled=not use_regime)
    reg_slow    = st.number_input("Regime slow MA", 50, 500, 200, 10, disabled=not use_regime)
    reg_ma_type = st.selectbox("Regime MA type", ["EMA","SMA"], disabled=not use_regime)
    vol_thresh  = st.slider("Bull vol threshold (%)", 10, 50, 20, disabled=not use_regime) / 100
    st.markdown("---")

    sh("Monte Carlo")
    mc_method = st.selectbox("Method", ["Bootstrap","Parametric","Trade Shuffle"])
    mc_n      = st.select_slider("Simulations", [200,500,1000,2000,5000], value=1000)
    st.markdown("---")

    run_btn = st.button("▶  RUN BACKTEST", use_container_width=True)


# ── HEADER ────────────────────────────────────────────────────
st.markdown("## 🚀 Momentum Engine")
st.caption("NSE Momentum Scanner · Backtester · Monte Carlo · Regime Detection")

# ── LANDING ───────────────────────────────────────────────────
if not run_btn:
    st.markdown("""
    <div style="background:#161B22;border:1px solid #21262D;border-radius:16px;
                padding:48px;text-align:center;margin-top:24px">
        <div style="font-size:52px;margin-bottom:16px">⚡</div>
        <div style="font-size:22px;font-weight:700;color:#E6EDF3;margin-bottom:8px">
            Configure your strategy in the sidebar, then hit RUN BACKTEST</div>
        <div style="color:#7D8590;font-size:14px;max-width:560px;margin:0 auto 28px">
            Pick an NSE index or upload your own CSV. Yahoo Finance data loads automatically.
        </div>
        <div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap">
            <span style="background:#0D4429;color:#3FB950;padding:6px 14px;border-radius:20px;font-size:12px;font-weight:700">📊 20+ Metrics</span>
            <span style="background:#0C2D6B;color:#58A6FF;padding:6px 14px;border-radius:20px;font-size:12px;font-weight:700">🎲 Monte Carlo</span>
            <span style="background:#2D1140;color:#BC8CFF;padding:6px 14px;border-radius:20px;font-size:12px;font-weight:700">🌡️ Regime Detection</span>
            <span style="background:#3D2000;color:#D29922;padding:6px 14px;border-radius:20px;font-size:12px;font-weight:700">📈 EMA/SMA Entry+Exit</span>
            <span style="background:#0D4429;color:#3FB950;padding:6px 14px;border-radius:20px;font-size:12px;font-weight:700">💰 SIP / Fixed / Reinvest</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    sh("📄 CSV TEMPLATES — DOWNLOAD & USE")

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("**Price CSV (Wide)**")
        st.caption("One column per stock, rows = dates")
        wide = pd.DataFrame({
            "date":     ["2022-01-03","2022-01-04","2022-01-05"],
            "RELIANCE": [2450.0,2480.0,2510.0],
            "TCS":      [3200.0,3190.0,3230.0],
            "HDFCBANK": [1620.0,1635.0,1618.0],
            "INFY":     [1750.0,1768.0,1745.0],
        })
        st.dataframe(wide, use_container_width=True, hide_index=True)
        st.download_button("📥 Download Wide template",
            wide.to_csv(index=False), "price_wide.csv", "text/csv")

    with c2:
        st.markdown("**Price CSV (Long)**")
        st.caption("Three columns: date, symbol, close")
        long = pd.DataFrame({
            "date":   ["2022-01-03","2022-01-03","2022-01-04","2022-01-04"],
            "symbol": ["RELIANCE","TCS","RELIANCE","TCS"],
            "close":  [2450.0,3200.0,2480.0,3190.0],
        })
        st.dataframe(long, use_container_width=True, hide_index=True)
        st.download_button("📥 Download Long template",
            long.to_csv(index=False), "price_long.csv", "text/csv")

    with c3:
        st.markdown("**Index / Benchmark CSV**")
        st.caption("Two columns: date, close (for custom benchmark)")
        idx_tmpl = pd.DataFrame({
            "date":  ["2022-01-03","2022-01-04","2022-01-05"],
            "close": [17625.70,17805.25,17812.40],
        })
        st.dataframe(idx_tmpl, use_container_width=True, hide_index=True)
        st.download_button("📥 Download Index template",
            idx_tmpl.to_csv(index=False), "index_benchmark.csv", "text/csv")

    st.markdown("---")
    sh("📋 INDEX CONSTITUENTS")
    sel_cat  = st.selectbox("View category", list(INDEX_CATEGORIES.keys()), key="view_cat")
    sel_idx  = st.selectbox("View index",    INDEX_CATEGORIES[sel_cat], key="view_idx")
    members  = NSE_INDICES.get(sel_idx,{}).get("components",[])
    if members:
        df_mem = pd.DataFrame({
            "Symbol": [m.replace(".NS","") for m in members],
            "Yahoo Ticker": members,
        })
        st.dataframe(df_mem, use_container_width=True, hide_index=True)
        st.download_button(f"📥 Download {sel_idx} constituents",
            df_mem.to_csv(index=False), f"{sel_idx.replace(' ','_')}_constituents.csv","text/csv")
    else:
        st.info("Upload a CSV for this index — constituent list not pre-loaded for large indices.")

    st.stop()


# ── RUN BACKTEST ──────────────────────────────────────────────
prog = st.progress(0, "Starting…")
try:
    prog.progress(5, "Loading prices…")
    if csv_prices:
        csv_prices.seek(0)
        prices  = parse_csv(csv_prices)
        tickers = list(prices.columns)
    else:
        tickers = idx_info.get("components", [])
        if not tickers:
            st.error("This index has no pre-loaded components. Please upload a Price CSV.")
            prog.empty(); st.stop()
        prices = fetch_yahoo(tickers, str(start_dt), str(end_dt))

    if prices.empty:
        st.error("No price data returned. Check date range or try uploading a CSV.")
        prog.empty(); st.stop()

    prices = prices.dropna(how="all",axis=1).dropna(how="all",axis=0)
    prog.progress(25, f"Loaded {len(prices.columns)} stocks · {len(prices)} days")

    prog.progress(30, "Fetching benchmark…")
    if csv_index:
        csv_index.seek(0)
        bm_df     = parse_csv(csv_index)
        benchmark = bm_df.iloc[:,0].rename("Index")
    else:
        bm_ticker = idx_info.get("index_ticker","^NSEI")
        benchmark = fetch_index(bm_ticker, str(start_dt), str(end_dt))

    regime_df = None
    if use_regime and not benchmark.empty:
        prog.progress(38, "Detecting regime…")
        regime_df = detect_regime(benchmark, reg_fast, reg_slow, reg_ma_type, vol_thresh=vol_thresh)

    prog.progress(48, "Computing signals…")
    signals = generate_signals(
        prices, mom_w, entry_fast, entry_slow,
        exit_ma_p, ma_type, top_n, ex_rank,
        rank_by=rank_by, rf=rf_rate,
    )

    prog.progress(62, "Running backtest…")
    results = run_backtest(
        prices, signals, benchmark, regime_df,
        float(capital), rebal, top_n, ex_rank,
        txn, slip, sizing, rf_rate,
        allocation_mode=alloc_mode, sip_amount=float(sip_amount),
    )

    prog.progress(82, f"Monte Carlo ({mc_n} sims)…")
    mc = run_monte_carlo(results["equity"], results["trades"],
                         n_sim=mc_n, method=mc_method,
                         capital=float(capital), rf=rf_rate)

    try:
        extra_periods = [int(x.strip()) for x in extra_ma_raw.split(",") if x.strip().isdigit()]
    except Exception:
        extra_periods = []

    prog.progress(100,"Done ✅"); prog.empty()

except Exception as e:
    prog.empty()
    st.error(f"❌ Error: {e}")
    import traceback; st.code(traceback.format_exc())
    st.stop()

m  = results["metrics"]
eq = results["equity"]
bm = results["benchmark"]
tr = results["trades"]
mo = m.get("_monthly", pd.Series())
an = m.get("_annual",  pd.Series())
dd = m.get("_drawdown",pd.Series())

# ── HEADLINE METRICS ─────────────────────────────────────────
st.markdown("---")
hc = st.columns(7)
for col,(lbl,val,suf,pg) in zip(hc,[
    ("CAGR",           fmt(m.get("CAGR (%)",0)),        "%", True),
    ("Index CAGR",     fmt(m.get("Index CAGR (%)",0)),   "%", True),
    ("Max Drawdown",   fmt(m.get("Max Drawdown (%)",0)), "%", False),
    ("Sharpe",         fmt(m.get("Sharpe Ratio",0),3),   "",  True),
    ("Sortino",        fmt(m.get("Sortino Ratio",0),3),  "",  True),
    ("Total P&L",      f"₹{m.get('Total P&L (₹)',0):,.0f}", "", True),
    ("Total Trades",   str(m.get("Total Trades",0)),     "",  True),
]):
    with col: mcard(lbl, f"{val}{suf}", pg)

# Regime + allocation badges
badge_cols = st.columns([2,2,3])
with badge_cols[0]:
    if regime_df is not None and not regime_df.empty:
        cr  = str(regime_df["regime"].iloc[-1]).upper()
        cls = {"BULL":"pill-bull","NEUTRAL":"pill-neutral","BEAR":"pill-bear"}.get(cr,"pill-neutral")
        st.markdown(f'Regime: <span class="{cls}">{cr}</span> · Exposure: `{regime_df["exposure"].iloc[-1]*100:.0f}%`', unsafe_allow_html=True)
with badge_cols[1]:
    st.markdown(f'Allocation: `{alloc_mode}` · Sizing: `{sizing}`')
with badge_cols[2]:
    st.markdown(f'Entry: price > {ma_type}{entry_fast} > {ma_type}{entry_slow} · Exit: price < {ma_type}{exit_ma_p} · Rank by: `{rank_by}`')

st.markdown("---")

# ── TABS ─────────────────────────────────────────────────────
t1,t2,t3,t4,t5,t6,t7,t8,t9 = st.tabs([
    "📊 Overview","📈 Equity","📅 Returns","📉 Drawdown",
    "🌡️ Regime","🎲 Monte Carlo","📋 Trades","💼 Portfolio","📋 Constituents"])


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
                     ("Realized P&L (₹)",True),("Total P&L (₹)",True),
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
        sh("STRATEGY CONFIG")
        st.markdown(f"""
        <div class="info-box">
        <b style="color:#00D4AA">Entry:</b> price &gt; {ma_type}({entry_fast}) AND {ma_type}({entry_fast}) &gt; {ma_type}({entry_slow})<br>
        <b style="color:#00D4AA">Exit:</b> price &lt; {ma_type}({exit_ma_p})<br>
        <b style="color:#00D4AA">Rank by:</b> {rank_by}<br>
        <b style="color:#00D4AA">Allocation:</b> {alloc_mode}{f" · ₹{sip_amount:,}/period" if alloc_mode=="SIP" else ""}<br>
        <b style="color:#00D4AA">Lookbacks:</b> {lb_opts}
        </div>
        """, unsafe_allow_html=True)


# ── EQUITY CURVE ─────────────────────────────────────────────
with t2:
    if not eq.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq.index, y=eq, name="Strategy",
            line=dict(color="#00D4AA",width=2),
            fill="tozeroy", fillcolor="rgba(0,212,170,0.06)"))
        if not bm.empty:
            fig.add_trace(go.Scatter(x=bm.index, y=bm,
                name=f"Benchmark ({idx_name})",
                line=dict(color="#7D8590",width=1.5,dash="dot")))

        # Extra MAs on equity (mean across portfolio)
        if extra_periods and not prices.empty:
            for p in extra_periods:
                try:
                    ma_avg = ma(prices, p, ma_type).mean(axis=1).reindex(eq.index)
                    # Scale to portfolio range for visual
                    ratio = eq.mean() / ma_avg.mean() if ma_avg.mean() != 0 else 1
                    fig.add_trace(go.Scatter(x=ma_avg.index, y=ma_avg*ratio,
                        name=f"{ma_type}{p} (avg)",
                        line=dict(width=1,dash="dot"), opacity=0.5))
                except Exception: pass

        fig.update_layout(title="Portfolio vs Benchmark", yaxis_title="Value (₹)",
                          height=460, **PLOT)
        st.plotly_chart(fig, use_container_width=True)

        if len(eq) > 260:
            rs = eq.pct_change().dropna().rolling(252).apply(
                lambda x: (x.mean()*252-rf_rate)/(x.std()*np.sqrt(252)+1e-9), raw=True)
            fig2 = go.Figure(go.Scatter(x=rs.index, y=rs,
                line=dict(color="#BC8CFF",width=1.5), name="Rolling Sharpe"))
            fig2.add_hline(y=1, line_dash="dash", line_color="#3FB950", annotation_text="Sharpe=1")
            fig2.add_hline(y=0, line_dash="dash", line_color="#F85149")
            fig2.update_layout(title="Rolling 252d Sharpe", height=260, **PLOT)
            st.plotly_chart(fig2, use_container_width=True)


# ── RETURNS ───────────────────────────────────────────────────
with t3:
    if not mo.empty:
        try:
            mo.index = pd.to_datetime(mo.index.to_timestamp() if hasattr(mo.index,"to_timestamp") else mo.index)
            dfm = pd.DataFrame({"y":mo.index.year,"m":mo.index.month,"r":mo.values*100})
            piv = dfm.pivot(index="y",columns="m",values="r")
            piv.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][:len(piv.columns)]
            fig_h = px.imshow(piv,
                color_continuous_scale=[[0,"#8B1A1A"],[0.5,"#161B22"],[1,"#145A32"]],
                color_continuous_midpoint=0, text_auto=".1f", aspect="auto")
            fig_h.update_layout(title="Monthly Returns Heatmap (%)", height=340,
                paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
                font=dict(color="#7D8590"), margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig_h, use_container_width=True)
        except Exception as e:
            st.warning(f"Heatmap error: {e}")

    if not an.empty:
        try:
            an.index = pd.to_datetime(an.index.to_timestamp() if hasattr(an.index,"to_timestamp") else an.index)
            fig_a = go.Figure(go.Bar(
                x=[str(d.year) for d in an.index],
                y=(an.values*100).round(2),
                marker_color=["#3FB950" if v>=0 else "#F85149" for v in an.values],
                text=[f"{v*100:.1f}%" for v in an.values], textposition="outside"))
            fig_a.update_layout(title="Annual Returns (%)", yaxis_title="Return (%)",
                                showlegend=False, height=320, **PLOT)
            st.plotly_chart(fig_a, use_container_width=True)
        except Exception: pass


# ── DRAWDOWN ─────────────────────────────────────────────────
with t4:
    if not dd.empty:
        fig_dd = go.Figure(go.Scatter(x=dd.index, y=dd*100, fill="tozeroy",
            fillcolor="rgba(248,81,73,0.12)", line=dict(color="#F85149",width=1.5)))
        fig_dd.add_hline(y=0, line_color="#30363D")
        fig_dd.update_layout(title="Drawdown (%)", yaxis_title="Drawdown (%)",
                             height=380, **PLOT)
        st.plotly_chart(fig_dd, use_container_width=True)
        dc1,dc2,dc3 = st.columns(3)
        with dc1: mcard("Max Drawdown", fmt(dd.min()*100), False)
        with dc2: mcard("Avg Drawdown", fmt(dd.mean()*100), False)
        with dc3: mcard("Days below -10%", str((dd<-0.10).sum()), False)


# ── REGIME ────────────────────────────────────────────────────
with t5:
    if regime_df is not None and not regime_df.empty:
        cr  = str(regime_df["regime"].iloc[-1]).upper()
        ce  = regime_df["exposure"].iloc[-1]
        cls = {"BULL":"pill-bull","NEUTRAL":"pill-neutral","BEAR":"pill-bear"}.get(cr,"pill-neutral")
        st.markdown(f'**Current:** <span class="{cls}">{cr}</span> · Exposure `{ce*100:.0f}%`',
                    unsafe_allow_html=True)

        fig_r = make_subplots(rows=2,cols=1,shared_xaxes=True,
                              row_heights=[0.7,0.3],vertical_spacing=0.06)
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
    if mc.get("n_sim",0) > 0:
        sh(f"MONTE CARLO — {mc['n_sim']:,} SIMULATIONS · {mc_method.upper()}")
        x = list(range(mc["n"]))
        fig_mc = go.Figure()
        for lo,hi,clr in [(5,95,"rgba(0,212,170,0.07)"),(25,75,"rgba(0,212,170,0.13)")]:
            fig_mc.add_trace(go.Scatter(x=x+x[::-1],
                y=list(mc["pcts"][hi])+list(mc["pcts"][lo][::-1]),
                fill="toself",fillcolor=clr,line=dict(width=0),name=f"{lo}–{hi}th pct"))
        fig_mc.add_trace(go.Scatter(x=x,y=mc["pcts"][50],name="Median",
            line=dict(color="#00D4AA",width=2)))
        fig_mc.add_trace(go.Scatter(x=x[:len(eq)],y=eq.values,name="Actual",
            line=dict(color="#FFFFFF",width=1.5,dash="dot")))
        fig_mc.update_layout(title="Simulated Equity Paths",yaxis_title="Value (₹)",
                             height=420,**PLOT)
        st.plotly_chart(fig_mc, use_container_width=True)

        ml,mr=st.columns(2)
        with ml:
            sh("STATS")
            for k,v in mc["stats"].items():
                pg = any(w in k for w in ["Profit","CAGR","95th","Median Final"])
                mcard(k, fmt(float(v), 0 if "₹" in k else 2), pg)
        with mr:
            fig_d=go.Figure(go.Histogram(x=mc["final"],nbinsx=60,
                marker=dict(color="#00D4AA",opacity=0.7,line=dict(width=0))))
            fig_d.add_vline(x=float(capital),line_dash="dash",line_color="#F85149",
                annotation_text="Initial Capital")
            fig_d.add_vline(x=float(np.median(mc["final"])),line_dash="dash",
                line_color="#3FB950",annotation_text="Median")
            fig_d.update_layout(title="Final Value Distribution",height=300,**PLOT,showlegend=False)
            st.plotly_chart(fig_d, use_container_width=True)
            fig_c=go.Figure(go.Histogram(x=mc["cagr_dist"],nbinsx=50,
                marker=dict(color="#BC8CFF",opacity=0.7,line=dict(width=0))))
            fig_c.add_vline(x=float(m.get("CAGR (%)",0)),line_dash="dash",
                line_color="#00D4AA",annotation_text="Actual CAGR")
            fig_c.update_layout(title="Simulated CAGR (%)",height=260,**PLOT)
            st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.info("Not enough data for Monte Carlo.")


# ── TRADES ────────────────────────────────────────────────────
with t7:
    if not tr.empty:
        sells = tr[tr["action"]=="SELL"].copy() if "action" in tr.columns else tr.copy()
        if not sells.empty and "pnl" in sells.columns:
            fig_pnl=go.Figure(go.Bar(
                x=sells["date"].astype(str) if "date" in sells.columns else sells.index.astype(str),
                y=sells["pnl"],
                marker_color=["#3FB950" if p>=0 else "#F85149" for p in sells["pnl"]]))
            fig_pnl.update_layout(title="P&L per Trade",yaxis_title="P&L (₹)",height=280,**PLOT)
            st.plotly_chart(fig_pnl, use_container_width=True)
        show_tr = tr.copy()
        if "date" in show_tr.columns:
            show_tr["date"]=pd.to_datetime(show_tr["date"]).dt.date
        st.dataframe(show_tr, use_container_width=True, height=400)
    else:
        st.info("No trades recorded. Check entry MA periods — if EMA(fast) > EMA(slow) is rarely true, try smaller periods (e.g. fast=50, slow=200 instead of 100/200).")


# ── PORTFOLIO & SCANNER ───────────────────────────────────────
with t8:
    sh("CURRENT HOLDINGS")
    fh  = results.get("final_holdings",{})
    fpx = results.get("final_prices",pd.Series())
    if fh:
        rows=[]
        for sym,sh_ct in fh.items():
            p = float(fpx[sym]) if sym in fpx.index else 0
            rows.append({"Symbol":sym,"Shares":round(sh_ct,4),
                         "Last Price (₹)":round(p,2),"Value (₹)":round(sh_ct*p,2)})
        dfh=pd.DataFrame(rows).sort_values("Value (₹)",ascending=False)
        tv=dfh["Value (₹)"].sum()
        dfh["Weight (%)"]=((dfh["Value (₹)"]/tv*100) if tv else 0).round(2)
        hl,hr=st.columns([2,1])
        with hl: st.dataframe(dfh,use_container_width=True)
        with hr:
            if not dfh.empty:
                fig_pie=go.Figure(go.Pie(labels=dfh["Symbol"],values=dfh["Value (₹)"],
                    hole=0.45,marker=dict(colors=px.colors.qualitative.Dark24)))
                fig_pie.update_layout(height=300,paper_bgcolor="#0D1117",
                    font=dict(color="#7D8590"),margin=dict(l=10,r=10,t=20,b=10))
                st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No holdings at end of backtest period.")

    st.markdown("---")
    sh("⚡ LIVE MOMENTUM SCANNER")
    ls=signals["filtered"].iloc[-1].dropna().sort_values(ascending=False)
    lr=signals["ranks"].iloc[-1].dropna().sort_values()
    le=signals["entry_filter"].iloc[-1]
    lx=signals["exit"].iloc[-1]
    if ls.empty:
        st.warning("No stocks pass the entry filter on the last date. Try reducing MA periods.")
    else:
        df_scan=pd.DataFrame({
            "Rank":      lr.reindex(ls.index).fillna(999).astype(int).values,
            "Symbol":    ls.index,
            "Score":     ls.values.round(4),
            "Entry ✅":  le.reindex(ls.index).map({True:"✅",False:"❌"}).fillna("❌").values,
            "Exit 🚪":   lx.reindex(ls.index).map({True:"🚪 EXIT",False:"—"}).fillna("—").values,
            "Status":    ["⭐ HOLD" if i<top_n else "" for i in range(len(ls))],
        }).head(30)
        st.dataframe(df_scan.set_index("Rank"),use_container_width=True,height=380)
    st.caption(f"As of {prices.index[-1].date()} · {len(prices.columns)} stocks")


# ── CONSTITUENTS TAB ─────────────────────────────────────────
with t9:
    sh(f"INDEX CONSTITUENTS — {idx_name}")
    members = idx_info.get("components",[])
    if csv_prices:
        st.info("Using uploaded CSV — constituent list below shows the index definition.")
    if members:
        df_c=pd.DataFrame({
            "Symbol":       [m.replace(".NS","") for m in members],
            "Yahoo Ticker": members,
            "In Universe":  ["✅" if m.replace(".NS","") in prices.columns else "❌ (no data)" for m in members],
        })
        st.dataframe(df_c,use_container_width=True,height=500)
        st.download_button(f"📥 Download {idx_name} constituents",
            df_c.to_csv(index=False),f"{idx_name.replace(' ','_')}_constituents.csv","text/csv")
    else:
        st.info("No pre-loaded constituent list for this index. Upload a Price CSV with your chosen stocks.")
        if not prices.empty:
            df_from_csv=pd.DataFrame({"Symbol":list(prices.columns)})
            st.dataframe(df_from_csv,use_container_width=True)


# ── EXPORT ────────────────────────────────────────────────────
st.markdown("---")
cfg_export={
    "Index": idx_name,"Start":str(start_dt),"End":str(end_dt),
    "Rebalance":rebal,"MA Type":ma_type,
    "Entry":f"price>{ma_type}({entry_fast}) AND {ma_type}({entry_fast})>{ma_type}({entry_slow})",
    "Exit":f"price<{ma_type}({exit_ma_p})",
    "Rank by":rank_by,"Top N":top_n,"Exit Rank":ex_rank,
    "Allocation":alloc_mode,
    "SIP Amount":f"₹{sip_amount:,}" if alloc_mode=="SIP" else "N/A",
    "Portfolio (₹)":f"₹{capital:,.0f}","Sizing":sizing,
    "Lookbacks":str(lb_opts),
    "Regime":"On" if use_regime else "Off",
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
