""""
Momentum Engine — Core Logic
Data · Indicators (SMA+EMA) · Signals · Backtest · Monte Carlo · Regime
"""
import io
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


# ── DATA ─────────────────────────────────────────────────────────────────────

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
            close = raw[["Close"]] if len(tickers) == 1 else raw
            if "Close" in raw.columns:
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


# ── INDICATORS ───────────────────────────────────────────────────────────────

def sma(prices, period):
    return prices.rolling(window=period, min_periods=max(1, period // 2)).mean()

def ema(prices, period):
    return prices.ewm(span=period, adjust=False, min_periods=max(1, period // 2)).mean()

def ma(prices, period, ma_type="EMA"):
    return ema(prices, period) if str(ma_type).upper() == "EMA" else sma(prices, period)

def vol(prices, window=20):
    return prices.pct_change().rolling(window, min_periods=5).std() * np.sqrt(252)

def momentum_score(prices, weights, risk_adj=True):
    v = vol(prices, 20).replace(0, np.nan)
    score = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for w, wt in weights.items():
        r = prices.pct_change(periods=int(w)).fillna(0)
        if risk_adj:
            r = r.divide(v).fillna(0)
        score = score + r * wt
    return score


# ── REGIME ───────────────────────────────────────────────────────────────────

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

def get_exposure(regime_df, dt):
    if regime_df is None:
        return 1.0
    if dt in regime_df.index:
        return float(regime_df.at[dt, "exposure"])
    prior = regime_df.index[regime_df.index <= dt]
    return float(regime_df.at[prior[-1], "exposure"]) if len(prior) else 0.5


# ── SIGNALS ──────────────────────────────────────────────────────────────────

def generate_signals(prices, mom_weights, entry_fast, entry_slow,
                     exit_ma_p, exit_ret_thresh, ma_type, top_n, exit_rank):
    periods = sorted({int(entry_fast), int(entry_slow), int(exit_ma_p)})
    mas = {p: ma(prices, p, ma_type) for p in periods}

    ef = (prices > mas[int(entry_fast)]) & (mas[int(entry_fast)] > mas[int(entry_slow)])
    xf = (prices < mas[int(exit_ma_p)]) | (prices.pct_change(20) < exit_ret_thresh)

    score    = momentum_score(prices, mom_weights)
    filtered = score.where(ef)
    ranks    = filtered.rank(axis=1, ascending=False, method="first")

    return {"score": score, "filtered": filtered, "ranks": ranks,
            "exit": xf, "entry_filter": ef, "mas": mas}


# ── BACKTEST ─────────────────────────────────────────────────────────────────

_FREQ = {"Weekly": "W-FRI", "Monthly": "BME", "Quarterly": "QE", "Yearly": "YE"}

def _rdates(index, freq):
    dates = pd.date_range(index[0], index[-1], freq=_FREQ.get(freq, "W-FRI"))
    out = set()
    for d in dates:
        fwd = index[index >= d]
        if len(fwd):
            out.add(fwd[0])
    return out

def _ivw(symbols, vol_df, dt):
    row = vol_df.loc[dt, [s for s in symbols if s in vol_df.columns]].dropna().replace(0, np.nan).dropna()
    if row.empty:
        return {s: 1.0 / len(symbols) for s in symbols}
    inv = 1.0 / row
    return {s: float(inv.get(s, 0) / inv.sum()) for s in symbols}

def run_backtest(prices, signals, benchmark, regime_df=None,
                 capital=1_000_000, rebal="Weekly", top_n=10, exit_rank=15,
                 txn=0.001, slip=0.001, sizing="Inverse Volatility", rf=0.065):

    prices  = prices.sort_index().dropna(how="all", axis=1).dropna(how="all", axis=0)
    ranks   = signals["ranks"].reindex(prices.index)
    xf      = signals["exit"].reindex(prices.index)
    fs      = signals["filtered"].reindex(prices.index)
    vols    = vol(prices, 20).reindex(prices.index)
    rdates  = _rdates(prices.index, rebal)

    holdings, epx, edt = {}, {}, {}
    cash = float(capital)
    eq_rows, tr_rows = [], []

    for dt in prices.index:
        px = prices.loc[dt]

        # mark-to-market
        pv = cash + sum(holdings[s] * float(px[s]) for s in holdings
                        if s in px.index and not pd.isna(px[s]))
        eq_rows.append({"date": dt, "value": pv})
        exp = get_exposure(regime_df, dt)

        # exit signals
        for sym in list(holdings):
            rb = (float(ranks.loc[dt, sym]) > exit_rank) if sym in ranks.columns else False
            xt = bool(xf.loc[dt, sym]) if sym in xf.columns else False
            if (rb or xt) and sym in px.index and not pd.isna(px[sym]):
                spx = float(px[sym]) * (1 - slip)
                proc = holdings[sym] * spx * (1 - txn)
                pnl  = proc - holdings[sym] * epx.get(sym, spx)
                cash += proc
                tr_rows.append({"date": dt, "action": "SELL", "symbol": sym,
                                 "price": round(spx, 2), "shares": round(holdings[sym], 4),
                                 "value": round(proc, 2), "pnl": round(pnl, 2),
                                 "held_days": (dt - edt[sym]).days if sym in edt else 0})
                holdings.pop(sym, None); epx.pop(sym, None); edt.pop(sym, None)

        # rebalance
        if dt in rdates:
            today_r = ranks.loc[dt].dropna()
            eligible = fs.loc[dt].dropna().index.tolist()
            top = today_r[today_r.index.isin(eligible)].nsmallest(top_n).index.tolist()

            for sym in [s for s in list(holdings) if s not in top]:
                if sym not in px.index or pd.isna(px[sym]): continue
                spx  = float(px[sym]) * (1 - slip)
                proc = holdings[sym] * spx * (1 - txn)
                pnl  = proc - holdings[sym] * epx.get(sym, spx)
                cash += proc
                tr_rows.append({"date": dt, "action": "SELL", "symbol": sym,
                                 "price": round(spx, 2), "shares": round(holdings[sym], 4),
                                 "value": round(proc, 2), "pnl": round(pnl, 2),
                                 "held_days": (dt - edt[sym]).days if sym in edt else 0})
                holdings.pop(sym, None); epx.pop(sym, None); edt.pop(sym, None)

            pv_now = cash + sum(holdings[s] * float(px[s]) for s in holdings
                                if s in px.index and not pd.isna(px[s]))
            invest = pv_now * exp

            if top and invest > 0:
                wmap = _ivw(top, vols, dt) if sizing == "Inverse Volatility" \
                       else {s: 1.0 / len(top) for s in top}
                for sym in top:
                    if sym not in px.index or pd.isna(px[sym]): continue
                    bpx   = float(px[sym]) * (1 + slip)
                    cper  = bpx * (1 + txn)
                    alloc = invest * wmap.get(sym, 0)
                    sh    = min(alloc, cash) / cper if cper > 0 else 0
                    tc    = sh * cper
                    if sh <= 0 or tc <= 0: continue
                    cash -= tc
                    if sym in holdings:
                        old = holdings[sym]; op = epx.get(sym, bpx)
                        holdings[sym] = old + sh
                        epx[sym] = (old * op + sh * bpx) / holdings[sym]
                    else:
                        holdings[sym] = sh; epx[sym] = bpx; edt[sym] = dt
                    tr_rows.append({"date": dt, "action": "BUY", "symbol": sym,
                                     "price": round(bpx, 2), "shares": round(sh, 4),
                                     "value": round(tc, 2), "pnl": 0, "held_days": 0})

    equity = pd.DataFrame(eq_rows).set_index("date")["value"]
    trades = pd.DataFrame(tr_rows) if tr_rows else pd.DataFrame()

    bm = benchmark.reindex(equity.index).ffill()
    bm_curve = (bm / bm.iloc[0] * capital) if (not bm.empty and bm.iloc[0] != 0) else pd.Series(dtype=float)

    return {"equity": equity, "benchmark": bm_curve, "trades": trades,
            "metrics": compute_metrics(equity, bm, trades, rf, capital),
            "final_holdings": holdings,
            "final_prices": prices.iloc[-1] if not prices.empty else pd.Series()}


# ── METRICS ──────────────────────────────────────────────────────────────────

def compute_metrics(equity, benchmark, trades, rf, capital):
    if equity.empty or len(equity) < 5:
        return {}
    eq   = equity.dropna()
    dr   = eq.pct_change().dropna()
    nyrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.01)
    tot  = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (1 + tot) ** (1 / nyrs) - 1

    bm = benchmark.reindex(eq.index).ffill().dropna()
    bm_cagr = ((bm.iloc[-1] / bm.iloc[0]) ** (1 / nyrs) - 1) if len(bm) > 2 and bm.iloc[0] != 0 else 0.0

    rm  = eq.cummax()
    dds = (eq - rm) / rm
    mdd = dds.min()

    sharpe  = (dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0
    down    = dr[dr < 0]
    sortino = ((dr.mean() * 252 - rf) / (down.std() * np.sqrt(252))) if len(down) > 1 and down.std() > 0 else 0.0

    mo = eq.resample("ME").last().pct_change().dropna()
    an = eq.resample("YE").last().pct_change().dropna()

    if not trades.empty and "pnl" in trades.columns and "action" in trades.columns:
        sells    = trades[trades["action"] == "SELL"]
        nt       = len(sells)
        pos      = int((sells["pnl"] > 0).sum())
        real     = float(sells["pnl"].sum())
        try:
            churn = trades.groupby(pd.to_datetime(trades["date"]).dt.to_period("M")).size().mean()
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
        "CAGR (%)":               round(cagr * 100, 2),
        "Index CAGR (%)":         round(bm_cagr * 100, 2),
        "Total Return (%)":       round(tot * 100, 2),
        "Max Drawdown (%)":       round(mdd * 100, 2),
        "Sharpe Ratio":           round(sharpe, 3),
        "Sortino Ratio":          round(sortino, 3),
        "Annual Volatility (%)":  round(float(dr.std()) * np.sqrt(252) * 100, 2),
        "Realized P&L (₹)":       round(real, 0),
        "Total P&L (₹)":          round(float(eq.iloc[-1] - capital), 0),
        "Total Trades":           nt,
        "Positive Trades (%)":    round(pos / nt * 100, 1) if nt else 0.0,
        "Avg Monthly Churn":      round(float(churn), 1),
        "Avg Monthly Return (%)": pct(mo),
        "Positive Months (%)":    round(float((mo > 0).mean()) * 100, 1) if not mo.empty else 0.0,
        "Avg +ve Month (%)":      pct(mo, mo > 0),
        "Avg -ve Month (%)":      pct(mo, mo < 0),
        "Avg Annual Return (%)":  pct(an),
        "Median Annual Return (%)": round(float(an.median()) * 100, 2) if not an.empty else 0.0,
        "Positive Years (%)":     round(float((an > 0).mean()) * 100, 1) if not an.empty else 0.0,
        "Avg +ve Year (%)":       pct(an, an > 0),
        "Avg -ve Year (%)":       pct(an, an < 0),
        "_monthly": mo, "_annual": an, "_drawdown": dds,
    }


# ── MONTE CARLO ───────────────────────────────────────────────────────────────

def run_monte_carlo(equity, trades=None, n_sim=1000, method="Bootstrap",
                    capital=1_000_000, rf=0.065, seed=42):
    np.random.seed(seed)
    dr = equity.pct_change().dropna().values
    n  = len(dr)
    if n < 10:
        return {"paths": np.array([]), "pcts": {}, "final": np.array([]),
                "ret_dist": np.array([]), "cagr_dist": np.array([]),
                "dd_dist": np.array([]), "n_sim": 0, "n": n,
                "initial": capital, "stats": {}}

    if method == "Parametric":
        sim = np.random.normal(dr.mean(), dr.std(), (n_sim, n))
    elif method == "Trade Shuffle" and trades is not None and not trades.empty:
        col = trades["pnl"] if "pnl" in trades.columns else pd.Series([0.0])
        if "action" in trades.columns:
            col = trades[trades["action"] == "SELL"]["pnl"]
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
    pcts  = {p: np.percentile(paths, p, axis=0) for p in [5, 25, 50, 75, 95]}
    v95   = float(np.percentile(ret, 5))
    cv95  = float(ret[ret <= v95].mean()) if (ret <= v95).any() else v95

    return {
        "paths": paths, "pcts": pcts, "final": final,
        "ret_dist": ret, "cagr_dist": cagr, "dd_dist": dd,
        "n_sim": n_sim, "n": n, "initial": capital,
        "stats": {
            "Median Final Value (₹)":    round(float(np.median(final)), 0),
            "5th Pct Final Value (₹)":   round(float(np.percentile(final, 5)), 0),
            "95th Pct Final Value (₹)":  round(float(np.percentile(final, 95)), 0),
            "Probability of Profit (%)": round(float((final > capital).mean() * 100), 1),
            "Prob of >20% Loss (%)":     round(float((final < capital * 0.8).mean() * 100), 1),
            "Median CAGR (%)":           round(float(np.median(cagr)), 2),
            "VaR 95% (%)":               round(v95, 2),
            "CVaR 95% (%)":              round(cv95, 2),
            "Median Max Drawdown (%)":   round(float(np.median(dd)), 2),
            "Worst-case Drawdown (%)":   round(float(dd.min()), 2),
        },
    }
