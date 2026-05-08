"""
Momentum Engine Core v4
Fixes: equal weight sizing, index CAGR, correlation filter
New: regime action modes, per-period simulator data, ticker override support
"""
import io, warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════

def fetch_yahoo(tickers, start, end):
    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False, threads=True)
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["Close"]]
            close.columns = [tickers[0].replace(".NS", "")]
        close.columns = [str(c).replace(".NS", "") for c in close.columns]
        close.index = pd.to_datetime(close.index)
        return close.sort_index().apply(pd.to_numeric, errors="coerce")
    except Exception as e:
        warnings.warn(f"fetch_yahoo error: {e}")
        return pd.DataFrame()


def fetch_index(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
        if df.empty:
            return pd.Series(dtype=float, name="Index")
        s = df["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s = s.squeeze()
        s.index = pd.to_datetime(s.index)
        s.name = "Index"
        return s.sort_index()
    except Exception:
        return pd.Series(dtype=float, name="Index")


def parse_csv(uploaded):
    content = uploaded.read()
    df = pd.read_csv(io.BytesIO(content))
    cols = [c.lower().strip() for c in df.columns]
    if "symbol" in cols and "close" in cols:
        df.columns = cols
        df["date"] = pd.to_datetime(df["date"])
        df = df.pivot(index="date", columns="symbol", values="close")
    else:
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index(df.columns[0])
        df.index.name = "date"
        df.columns = [c.strip().replace(".NS", "") for c in df.columns]
    return df.sort_index().apply(pd.to_numeric, errors="coerce").dropna(how="all")


# ═══════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════

def sma(prices, period):
    return prices.rolling(window=period, min_periods=max(5, period // 4)).mean()

def ema(prices, period):
    return prices.ewm(span=period, adjust=False, min_periods=max(5, period // 4)).mean()

def ma(prices, period, ma_type="EMA"):
    return ema(prices, period) if str(ma_type).upper() == "EMA" else sma(prices, period)

def ann_vol(prices, window=20):
    return prices.pct_change().rolling(window, min_periods=5).std() * np.sqrt(252)

def rolling_return(prices, window):
    return prices.pct_change(periods=int(window))

def rolling_sharpe(prices, window=60, rf=0.065):
    dr = prices.pct_change()
    mu = dr.rolling(window, min_periods=10).mean() * 252
    sd = dr.rolling(window, min_periods=10).std() * np.sqrt(252)
    return (mu - rf) / sd.replace(0, np.nan)


# ═══════════════════════════════════════════════════════════════
# RANK SCORE
# ═══════════════════════════════════════════════════════════════

def compute_rank_score(prices, rank_by="Momentum", weights=None, rf=0.065):
    if weights is None:
        weights = {60: 0.25, 90: 0.25, 120: 0.25, 252: 0.25}
    v = ann_vol(prices, 20).replace(0, np.nan)
    if rank_by == "Sharpe Ratio":
        max_w = max(weights.keys())
        return rolling_sharpe(prices, window=max_w, rf=rf)
    elif rank_by == "Return %":
        score = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for w, wt in weights.items():
            score = score + rolling_return(prices, w).fillna(0) * wt
        return score
    elif rank_by == "Low Volatility":
        return -v
    else:  # Momentum (risk-adjusted)
        score = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for w, wt in weights.items():
            r = rolling_return(prices, w).fillna(0).divide(v).fillna(0)
            score = score + r * wt
        return score


# ═══════════════════════════════════════════════════════════════
# CORRELATION FILTER
# ═══════════════════════════════════════════════════════════════

def apply_correlation_filter(candidates, prices, corr_window=60, corr_threshold=0.7, top_n=10):
    """
    From ranked candidates list, greedily select stocks where pairwise
    correlation < corr_threshold. Returns up to top_n uncorrelated stocks.
    candidates: list of symbols ordered best-rank first
    """
    if not candidates or corr_threshold >= 1.0:
        return candidates[:top_n]

    recent = prices[candidates].tail(corr_window).pct_change().dropna()
    if recent.empty or len(recent) < 10:
        return candidates[:top_n]

    corr = recent.corr()
    selected = []
    for sym in candidates:
        if len(selected) >= top_n:
            break
        if not selected:
            selected.append(sym)
            continue
        # Check correlation with all already-selected stocks
        max_corr = max(abs(corr.loc[sym, s]) for s in selected if sym in corr.index and s in corr.columns)
        if max_corr < corr_threshold:
            selected.append(sym)
    return selected


# ═══════════════════════════════════════════════════════════════
# ENTRY / EXIT
# ═══════════════════════════════════════════════════════════════

def entry_filter(prices, entry_fast_ma, entry_slow_ma):
    return (prices > entry_fast_ma) & (entry_fast_ma > entry_slow_ma)

def exit_filter(prices, exit_ma_df):
    return prices < exit_ma_df


# ═══════════════════════════════════════════════════════════════
# SIGNALS
# ═══════════════════════════════════════════════════════════════

def generate_signals(prices, mom_weights, entry_fast_p, entry_slow_p,
                     exit_ma_p, ma_type, top_n, exit_rank,
                     rank_by="Momentum", rf=0.065):
    all_periods = sorted({int(entry_fast_p), int(entry_slow_p), int(exit_ma_p)})
    mas = {p: ma(prices, p, ma_type) for p in all_periods}
    ef  = entry_filter(prices, mas[int(entry_fast_p)], mas[int(entry_slow_p)])
    xf  = exit_filter(prices, mas[int(exit_ma_p)])
    score    = compute_rank_score(prices, rank_by=rank_by, weights=mom_weights, rf=rf)
    filtered = score.where(ef)
    ranks    = filtered.rank(axis=1, ascending=False, method="first")
    return {"score": score, "filtered": filtered, "ranks": ranks,
            "exit": xf, "entry_filter": ef, "mas": mas}


# ═══════════════════════════════════════════════════════════════
# REGIME
# ═══════════════════════════════════════════════════════════════

def detect_regime(index_series, fast=50, slow=200, ma_type="EMA", vol_thresh=0.20):
    df = index_series.to_frame("price").copy()
    fn = ema if str(ma_type).upper() == "EMA" else sma
    df["fast"] = fn(df[["price"]], fast).squeeze()
    df["slow"] = fn(df[["price"]], slow).squeeze()
    df["rv"]   = df["price"].pct_change().rolling(20, min_periods=5).std() * np.sqrt(252)
    regimes, exposures = [], []
    for _, row in df.iterrows():
        if pd.isna(row["fast"]) or pd.isna(row["slow"]):
            regimes.append("NEUTRAL"); exposures.append(0.5)
        elif row["price"] > row["fast"] > row["slow"] and row["rv"] < vol_thresh:
            regimes.append("BULL");    exposures.append(1.0)
        elif row["price"] > row["slow"]:
            regimes.append("NEUTRAL"); exposures.append(0.5)
        else:
            regimes.append("BEAR");    exposures.append(0.2)
    df["regime"]   = regimes
    df["exposure"] = exposures
    return df


def get_regime_info(regime_df, dt):
    """Returns (regime_str, exposure_float) for a given date."""
    if regime_df is None:
        return "BULL", 1.0
    if dt in regime_df.index:
        row = regime_df.loc[dt]
    else:
        prior = regime_df.index[regime_df.index <= dt]
        if not len(prior):
            return "NEUTRAL", 0.5
        row = regime_df.loc[prior[-1]]
    return str(row["regime"]).upper(), float(row["exposure"])


# ═══════════════════════════════════════════════════════════════
# POSITION SIZING  — FIXED EQUAL WEIGHT
# ═══════════════════════════════════════════════════════════════

def _equal_weights(symbols):
    """Strict equal weight — each stock gets exactly 1/N of portfolio."""
    n = len(symbols)
    return {s: 1.0 / n for s in symbols} if n > 0 else {}

def _inv_vol_weights(symbols, vol_df, dt):
    row = vol_df.loc[dt, [s for s in symbols if s in vol_df.columns]].dropna()
    row = row.replace(0, np.nan).dropna()
    if row.empty:
        return _equal_weights(symbols)
    inv = 1.0 / row
    tot = inv.sum()
    return {s: float(inv.get(s, 0) / tot) for s in symbols}


# ═══════════════════════════════════════════════════════════════
# REBALANCE DATES
# ═══════════════════════════════════════════════════════════════

_WEEK_DAY    = {"Monday":"W-MON","Tuesday":"W-TUE","Wednesday":"W-WED",
                "Thursday":"W-THU","Friday":"W-FRI"}
_FREQ_OTHER  = {"Monthly":"BME","Quarterly":"QE","Yearly":"YE"}

def _rebal_dates(index, freq, rebal_day="Friday"):
    offset = _WEEK_DAY.get(rebal_day,"W-FRI") if freq=="Weekly" else _FREQ_OTHER.get(freq,"BME")
    dates  = pd.date_range(index[0], index[-1], freq=offset)
    out = set()
    for d in dates:
        fwd = index[index >= d]
        if len(fwd): out.add(fwd[0])
    return out


# ═══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# regime_action:
#   "Scale Exposure"        — original behaviour (exposure multiplier)
#   "Exit per Strategy"     — honour exit signals, NO new entries
#   "Keep Stocks No Entry"  — hold current, NO new entries or rebalance buys
#   "Exit All No Entry"     — sell everything, NO new entries
# ═══════════════════════════════════════════════════════════════

def run_backtest(
    prices, signals, benchmark, regime_df=None,
    capital=1_000_000, rebal="Weekly", top_n=10, exit_rank=15,
    txn=0.001, slip=0.001, sizing="Equal Weight", rf=0.065,
    allocation_mode="Reinvestment", sip_amount=50_000,
    rebal_day="Friday",
    use_corr_filter=False, corr_threshold=0.7, corr_window=60,
    regime_action="Scale Exposure",
):
    prices  = prices.sort_index().dropna(how="all", axis=1).dropna(how="all", axis=0)
    ranks   = signals["ranks"].reindex(prices.index)
    xf      = signals["exit"].reindex(prices.index)
    fs      = signals["filtered"].reindex(prices.index)
    vols    = ann_vol(prices, 20).reindex(prices.index)
    rdates  = _rebal_dates(prices.index, rebal, rebal_day)

    holdings       = {}   # sym -> shares
    epx            = {}   # sym -> avg entry price
    edt            = {}   # sym -> entry date
    cash           = float(capital)
    total_invested = float(capital)
    eq_rows, tr_rows = [], []

    # Per-period snapshot for simulator tab
    period_snapshots = []

    for dt in prices.index:
        px = prices.loc[dt]

        # SIP top-up
        if allocation_mode == "SIP" and dt in rdates:
            cash           += float(sip_amount)
            total_invested += float(sip_amount)

        # Mark-to-market
        port_val = cash + sum(
            holdings[s] * float(px[s]) for s in holdings
            if s in px.index and not pd.isna(px[s])
        )
        eq_rows.append({"date": dt, "value": port_val})

        # ── Regime action
        regime_str, exposure = get_regime_info(regime_df, dt)

        if regime_action == "Exit All No Entry" and regime_str == "BEAR":
            # Liquidate everything, stop
            for sym in list(holdings):
                if sym not in px.index or pd.isna(px[sym]): continue
                spx  = float(px[sym]) * (1 - slip)
                proc = holdings[sym] * spx * (1 - txn)
                pnl  = proc - holdings[sym] * epx.get(sym, spx)
                cash += proc
                tr_rows.append({"date":dt,"action":"SELL","symbol":sym,
                    "price":round(spx,2),"shares":round(holdings[sym],4),
                    "value":round(proc,2),"pnl":round(pnl,2),
                    "held_days":(dt-edt[sym]).days if sym in edt else 0})
                holdings.pop(sym,None); epx.pop(sym,None); edt.pop(sym,None)
            continue  # no new entries

        if regime_action == "Keep Stocks No Entry" and regime_str == "BEAR":
            continue  # hold existing, no exits, no new entries

        # ── Exit signals (strategy-based)
        for sym in list(holdings):
            exit_triggered = bool(xf.loc[dt, sym]) if sym in xf.columns else False
            rank_too_low   = (float(ranks.loc[dt, sym]) > exit_rank) if sym in ranks.columns else False
            if (exit_triggered or rank_too_low) and sym in px.index and not pd.isna(px[sym]):
                spx  = float(px[sym]) * (1 - slip)
                proc = holdings[sym] * spx * (1 - txn)
                pnl  = proc - holdings[sym] * epx.get(sym, spx)
                cash += proc
                tr_rows.append({"date":dt,"action":"SELL","symbol":sym,
                    "price":round(spx,2),"shares":round(holdings[sym],4),
                    "value":round(proc,2),"pnl":round(pnl,2),
                    "held_days":(dt-edt[sym]).days if sym in edt else 0})
                holdings.pop(sym,None); epx.pop(sym,None); edt.pop(sym,None)

        if regime_action == "Exit per Strategy" and regime_str == "BEAR":
            continue  # exits processed above, no new entries

        # ── Rebalance
        if dt in rdates:
            today_r  = ranks.loc[dt].dropna()
            eligible = fs.loc[dt].dropna().index.tolist()

            # Rank by score
            ranked_candidates = today_r[today_r.index.isin(eligible)].sort_values().index.tolist()

            # Correlation filter
            if use_corr_filter and len(ranked_candidates) > top_n:
                top = apply_correlation_filter(
                    ranked_candidates, prices, corr_window, corr_threshold, top_n)
            else:
                top = ranked_candidates[:top_n]

            # Sell stocks no longer in top
            for sym in [s for s in list(holdings) if s not in top]:
                if sym not in px.index or pd.isna(px[sym]): continue
                spx  = float(px[sym]) * (1 - slip)
                proc = holdings[sym] * spx * (1 - txn)
                pnl  = proc - holdings[sym] * epx.get(sym, spx)
                cash += proc
                tr_rows.append({"date":dt,"action":"SELL","symbol":sym,
                    "price":round(spx,2),"shares":round(holdings[sym],4),
                    "value":round(proc,2),"pnl":round(pnl,2),
                    "held_days":(dt-edt[sym]).days if sym in edt else 0})
                holdings.pop(sym,None); epx.pop(sym,None); edt.pop(sym,None)

            if not top:
                continue

            # Portfolio value post-sells
            pv_now = cash + sum(
                holdings[s] * float(px[s]) for s in holdings
                if s in px.index and not pd.isna(px[s])
            )

            # Investable amount
            if allocation_mode == "Reinvestment":
                invest = pv_now * (exposure if regime_action == "Scale Exposure" else 1.0)
            elif allocation_mode == "SIP":
                invest = pv_now * (exposure if regime_action == "Scale Exposure" else 1.0)
            else:  # Fixed
                invest = min(float(capital), pv_now)

            if invest <= 0:
                continue

            # ── POSITION SIZING — strictly enforced
            if sizing == "Equal Weight":
                # Each stock gets exactly invest/top_n
                wmap = _equal_weights(top)
            else:
                wmap = _inv_vol_weights(top, vols, dt)

            # Buy / top-up each stock
            for sym in top:
                if sym not in px.index or pd.isna(px[sym]): continue
                bpx       = float(px[sym]) * (1 + slip)
                cost_per  = bpx * (1 + txn)
                target_val= invest * wmap.get(sym, 0.0)   # target ₹ value
                target_sh = target_val / bpx if bpx > 0 else 0.0

                current_val = holdings.get(sym, 0.0) * bpx
                delta_val   = target_val - current_val
                delta_sh    = delta_val / bpx if bpx > 0 else 0.0

                if delta_sh > 0.01:   # need to buy more
                    cost = delta_sh * cost_per
                    cost = min(cost, cash)
                    sh   = cost / cost_per
                    if sh <= 0: continue
                    cash -= sh * cost_per
                    if sym in holdings:
                        old = holdings[sym]; op = epx.get(sym, bpx)
                        holdings[sym] = old + sh
                        epx[sym] = (old * op + sh * bpx) / holdings[sym]
                    else:
                        holdings[sym] = sh; epx[sym] = bpx; edt[sym] = dt
                    tr_rows.append({"date":dt,"action":"BUY","symbol":sym,
                        "price":round(bpx,2),"shares":round(sh,4),
                        "value":round(sh*cost_per,2),"pnl":0,"held_days":0})
                elif delta_sh < -0.01:  # need to trim
                    trim_sh  = abs(delta_sh)
                    sell_px  = float(px[sym]) * (1 - slip)
                    proceeds = trim_sh * sell_px * (1 - txn)
                    pnl_trim = proceeds - trim_sh * epx.get(sym, sell_px)
                    cash    += proceeds
                    holdings[sym] = max(0.0, holdings[sym] - trim_sh)
                    if holdings[sym] < 0.001:
                        holdings.pop(sym,None); epx.pop(sym,None); edt.pop(sym,None)
                    tr_rows.append({"date":dt,"action":"TRIM","symbol":sym,
                        "price":round(sell_px,2),"shares":round(trim_sh,4),
                        "value":round(proceeds,2),"pnl":round(pnl_trim,2),"held_days":0})

            # ── Snapshot for simulator
            snap_holdings = {}
            for sym, sh_count in holdings.items():
                if sym in px.index and not pd.isna(px[sym]):
                    curr_px   = float(px[sym])
                    mkt_val   = sh_count * curr_px
                    unreal    = mkt_val - sh_count * epx.get(sym, curr_px)
                    snap_holdings[sym] = {
                        "shares":      round(sh_count, 4),
                        "entry_price": round(epx.get(sym, curr_px), 2),
                        "curr_price":  round(curr_px, 2),
                        "mkt_value":   round(mkt_val, 2),
                        "unrealised":  round(unreal, 2),
                        "weight_pct":  0,  # filled below
                    }
            total_mkt = sum(v["mkt_value"] for v in snap_holdings.values())
            for sym in snap_holdings:
                snap_holdings[sym]["weight_pct"] = round(
                    snap_holdings[sym]["mkt_value"] / total_mkt * 100 if total_mkt else 0, 2)

            period_snapshots.append({
                "date":          dt,
                "portfolio_val": round(port_val, 2),
                "cash":          round(cash, 2),
                "holdings":      snap_holdings,
                "regime":        regime_str,
                "trades_today":  [r for r in tr_rows if r["date"] == dt],
            })

    equity = pd.DataFrame(eq_rows).set_index("date")["value"]
    trades = pd.DataFrame(tr_rows) if tr_rows else pd.DataFrame()

    bm = benchmark.reindex(equity.index).ffill()
    bm_curve = (bm / bm.iloc[0] * capital) if (not bm.empty and bm.iloc[0] != 0) else pd.Series(dtype=float)

    # Compute benchmark CAGR properly
    bm_cagr = 0.0
    bm_clean = bm.dropna()
    if len(bm_clean) > 5 and bm_clean.iloc[0] != 0:
        nyrs_bm = (bm_clean.index[-1] - bm_clean.index[0]).days / 365.25
        bm_cagr = ((bm_clean.iloc[-1] / bm_clean.iloc[0]) ** (1 / max(nyrs_bm, 0.01)) - 1) * 100
    # Fallback: use normalised benchmark curve
    if bm_cagr == 0.0 and not bm_curve.empty:
        bm_c2 = bm_curve.dropna()
        if len(bm_c2) > 5 and bm_c2.iloc[0] != 0:
            nyrs_bm = (bm_c2.index[-1] - bm_c2.index[0]).days / 365.25
            bm_cagr = ((bm_c2.iloc[-1] / bm_c2.iloc[0]) ** (1 / max(nyrs_bm, 0.01)) - 1) * 100

    metrics = compute_metrics(equity, bm, trades, rf, capital, total_invested)
    metrics["Index CAGR (%)"] = round(bm_cagr, 2)

    return {
        "equity":           equity,
        "benchmark":        bm_curve,
        "trades":           trades,
        "metrics":          metrics,
        "final_holdings":   holdings,
        "final_prices":     prices.iloc[-1] if not prices.empty else pd.Series(),
        "total_invested":   total_invested,
        "period_snapshots": period_snapshots,
    }


# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════

def compute_metrics(equity, benchmark, trades, rf, capital, total_invested=None):
    if equity.empty or len(equity) < 5:
        return {}
    if total_invested is None:
        total_invested = capital

    eq   = equity.dropna()
    dr   = eq.pct_change().dropna()
    nyrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.01)
    tot  = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (1 + tot) ** (1 / nyrs) - 1

    bm = benchmark.reindex(eq.index).ffill().dropna()
    if len(bm) > 2 and bm.iloc[0] != 0:
        nyrs_bm  = (bm.index[-1] - bm.index[0]).days / 365.25
        bm_cagr  = ((bm.iloc[-1] / bm.iloc[0]) ** (1 / max(nyrs_bm, 0.01)) - 1) * 100
    else:
        bm_cagr = 0.0

    rm  = eq.cummax()
    dds = (eq - rm) / rm
    mdd = dds.min()

    sharpe  = (dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0
    down    = dr[dr < 0]
    sortino = ((dr.mean() * 252 - rf) / (down.std() * np.sqrt(252))) \
              if len(down) > 1 and down.std() > 0 else 0.0

    mo = eq.resample("ME").last().pct_change().dropna()
    an = eq.resample("YE").last().pct_change().dropna()

    if not trades.empty and "pnl" in trades.columns and "action" in trades.columns:
        sells = trades[trades["action"].isin(["SELL","TRIM"])]
        nt    = len(sells)
        pos   = int((sells["pnl"] > 0).sum())
        real  = float(sells["pnl"].sum())
        try:
            churn = trades.groupby(
                pd.to_datetime(trades["date"]).dt.to_period("M")).size().mean()
        except Exception:
            churn = 0.0
    else:
        nt = pos = 0; real = churn = 0.0

    def pct(s, mask=None):
        try:
            x = s[mask] if mask is not None else s
            return round(float(x.mean()) * 100, 2) if not x.empty else 0.0
        except Exception:
            return 0.0

    return {
        "CAGR (%)":                round(cagr * 100, 2),
        "Index CAGR (%)":          round(bm_cagr, 2),
        "Total Return (%)":        round(tot * 100, 2),
        "Max Drawdown (%)":        round(mdd * 100, 2),
        "Sharpe Ratio":            round(sharpe, 3),
        "Sortino Ratio":           round(sortino, 3),
        "Annual Volatility (%)":   round(float(dr.std()) * np.sqrt(252) * 100, 2),
        "Realized P&L (₹)":        round(real, 0),
        "Total P&L (₹)":           round(float(eq.iloc[-1] - total_invested), 0),
        "Total Invested (₹)":      round(total_invested, 0),
        "Total Trades":            nt,
        "Positive Trades (%)":     round(pos / nt * 100, 1) if nt else 0.0,
        "Avg Monthly Churn":       round(float(churn), 1),
        "Avg Monthly Return (%)":  pct(mo),
        "Positive Months (%)":     round(float((mo > 0).mean()) * 100, 1) if not mo.empty else 0.0,
        "Avg +ve Month (%)":       pct(mo, mo > 0),
        "Avg -ve Month (%)":       pct(mo, mo < 0),
        "Avg Annual Return (%)":   pct(an),
        "Median Annual Return (%)":round(float(an.median()) * 100, 2) if not an.empty else 0.0,
        "Positive Years (%)":      round(float((an > 0).mean()) * 100, 1) if not an.empty else 0.0,
        "Avg +ve Year (%)":        pct(an, an > 0),
        "Avg -ve Year (%)":        pct(an, an < 0),
        "_monthly": mo, "_annual": an, "_drawdown": dds,
    }


# ═══════════════════════════════════════════════════════════════
# MONTE CARLO
# ═══════════════════════════════════════════════════════════════

def run_monte_carlo(equity, trades=None, n_sim=1000, method="Bootstrap",
                    capital=1_000_000, rf=0.065, seed=42):
    np.random.seed(seed)
    dr = equity.pct_change().dropna().values
    n  = len(dr)
    if n < 10:
        return {"paths":np.array([]),"pcts":{},"final":np.array([]),
                "ret_dist":np.array([]),"cagr_dist":np.array([]),
                "dd_dist":np.array([]),"n_sim":0,"n":n,"initial":capital,"stats":{}}

    if method == "Parametric":
        sim = np.random.normal(dr.mean(), dr.std(), (n_sim, n))
    elif method == "Trade Shuffle" and trades is not None and not trades.empty:
        col = (trades[trades["action"].isin(["SELL","TRIM"])]["pnl"]
               if "action" in trades.columns else pd.Series([0.0]))
        pvals = col.dropna().values
        if len(pvals) == 0: pvals = np.array([0.0])
        sim = np.random.choice(pvals, (n_sim, n), replace=True) / capital
    else:
        sim = dr[np.random.randint(0, n, (n_sim, n))]

    paths = np.cumprod(1 + np.clip(sim, -0.99, 5), axis=1) * capital
    final = paths[:, -1]
    nyrs  = n / 252
    ret   = (final / capital - 1) * 100
    cagr  = ((final / capital) ** (1 / max(nyrs, 0.01)) - 1) * 100
    rm    = np.maximum.accumulate(paths, axis=1)
    dd    = ((paths - rm) / np.where(rm == 0, 1, rm)).min(axis=1) * 100
    pcts  = {p: np.percentile(paths, p, axis=0) for p in [5,25,50,75,95]}
    v95   = float(np.percentile(ret, 5))
    cv95  = float(ret[ret <= v95].mean()) if (ret <= v95).any() else v95

    return {
        "paths":paths,"pcts":pcts,"final":final,
        "ret_dist":ret,"cagr_dist":cagr,"dd_dist":dd,
        "n_sim":n_sim,"n":n,"initial":capital,
        "stats":{
            "Median Final Value (₹)":    round(float(np.median(final)),0),
            "5th Pct Final Value (₹)":   round(float(np.percentile(final,5)),0),
            "95th Pct Final Value (₹)":  round(float(np.percentile(final,95)),0),
            "Probability of Profit (%)": round(float((final>capital).mean()*100),1),
            "Prob of >20% Loss (%)":     round(float((final<capital*0.8).mean()*100),1),
            "Median CAGR (%)":           round(float(np.median(cagr)),2),
            "VaR 95% (%)":               round(v95,2),
            "CVaR 95% (%)":              round(cv95,2),
            "Median Max Drawdown (%)":   round(float(np.median(dd)),2),
            "Worst-case Drawdown (%)":   round(float(dd.min()),2),
        },
    }
