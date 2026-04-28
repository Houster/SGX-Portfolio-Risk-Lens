"""
Pure quantitative analysis functions for SGX Portfolio Risk Lens.
All functions are stateless — they take data in, return dicts out.
Dependencies: pandas, numpy, yfinance, requests (no scipy/statsmodels required).
"""
import logging
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import yfinance as yf
import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

def _build_yf_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    proxy_user = os.getenv("PROXY_USER")
    proxy_pass = os.getenv("PROXY_PASS")
    proxy_host = os.getenv("PROXY_HOST")
    proxy_port = os.getenv("PROXY_PORT")
    if proxy_user and proxy_host:
        proxy_url = f"http://{proxy_user}:{proxy_pass}@{proxy_host}:{proxy_port}"
        session.proxies.update({"http": proxy_url, "https": proxy_url})
    return session

_YF_SESSION = _build_yf_session()


PERIOD_DAYS: Dict[str, int] = {"1y": 365, "3y": 1095, "5y": 1825}

# Extra tickers always fetched alongside portfolio tickers
AUX_TICKERS = ["ES3.SI", "SGDUSD=X", "GC=F", "AAXJ", "AXJR"]

# Waterfall of SORA proxy candidates, tried in order.
# Third element: invert=True means the series moves inversely to rates
# (bond ETF price rises when rates fall), so multiply by -1 after differentiating.
SORA_CANDIDATES: List[Tuple[str, str, bool]] = [
    ("2YY=R",     "SGS 2Y Bond Yield",                False),
    ("SG10YT=RR", "SGS 10Y Bond Yield",               False),
    ("A35.SI",    "iShares Asia IG Bond ETF (A35.SI)", True),
    ("^IRX",      "US 3M T-Bill (fallback)",           False),
]


# ─── Numpy OLS helpers ────────────────────────────────────────────────────────

def _ols(X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Ordinary least squares via numpy.  X must already include the constant column.
    Returns params, standard errors, 95% CI bounds, R², adjusted R².
    Uses 1.96 as the critical value (valid for n ≥ 30).
    """
    n, k = X.shape
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)

    y_hat = X @ beta
    resid = y - y_hat
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    adj_r2 = 1.0 - (1 - r2) * (n - 1) / max(n - k, 1)

    mse = ss_res / max(n - k, 1)
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.maximum(mse * np.diag(XtX_inv), 0.0))
    except np.linalg.LinAlgError:
        se = np.zeros(k)

    ci_low  = beta - 1.96 * se
    ci_high = beta + 1.96 * se

    return {
        "params": beta, "se": se,
        "ci_low": ci_low, "ci_high": ci_high,
        "r2": r2, "adj_r2": adj_r2,
    }


def _rolling_ols_params(
    X_df: pd.DataFrame, y: pd.Series, window: int, step: int = 10
) -> pd.DataFrame:
    """
    Rolling OLS over aligned (X_df, y) with stride `step`.
    Returns DataFrame of coefficient time series indexed by date.
    """
    X = X_df.values
    yv = y.values
    idx = X_df.index
    cols = X_df.columns.tolist()
    n = len(y)

    rows = []
    dates = []
    for end in range(window, n + 1, step):
        start = end - window
        Xi = X[start:end]
        yi = yv[start:end]
        try:
            res = _ols(Xi, yi)
            rows.append(res["params"])
            dates.append(idx[end - 1])
        except Exception:
            pass

    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, index=dates, columns=cols)


# ─── Data Fetching ────────────────────────────────────────────────────────────

def _download_one(ticker: str, start: datetime, end: datetime) -> Optional[pd.Series]:
    try:
        df = yf.download(
            ticker, start=start, end=end,
            auto_adjust=True, progress=False, threads=False,
            session=_YF_SESSION,
        )
        if df is None or len(df) < 20:
            return None
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.squeeze()
        close.name = ticker
        return close
    except Exception:
        return None


def fetch_sora_proxy(start: datetime, end: datetime) -> Tuple[Optional[pd.Series], str]:
    """
    Try each entry in SORA_CANDIDATES in order; return the first series with
    sufficient data, already differentiated and sign-adjusted so that
    positive values = rising rates.  Returns (None, "") if all fail.

    Sign convention: yield tickers already move in the rate direction;
    bond ETF prices move inversely, so those are multiplied by -1 after
    pct_change so the sign is consistent across all fallbacks.
    """
    for ticker, label, invert in SORA_CANDIDATES:
        series = _download_one(ticker, start, end)
        if series is None:
            continue
        clean = series.ffill(limit=5).bfill(limit=5)
        if clean.dropna().empty:
            continue
        if ticker == "^IRX":
            logging.warning(
                "SORA proxy: all SGS/SGX sources failed — "
                "using US 3M T-Bill (^IRX) as last-resort fallback"
            )
        chg = clean.pct_change().dropna()
        if invert:
            chg = chg * -1
        return chg, label
    return None, ""


def fetch_price_data(tickers: List[str], period: str = "5y") -> Dict[str, Any]:
    """
    Fetch adjusted closing prices for portfolio tickers plus aux tickers.
    Returns an aligned price DataFrame with coverage metadata.
    """
    days = PERIOD_DAYS.get(period, 1825)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 90)   # 90-day buffer

    all_tickers = list(dict.fromkeys(tickers + AUX_TICKERS))
    raw: Dict[str, pd.Series] = {}
    failed: List[str] = []

    for ticker in all_tickers:
        series = _download_one(ticker, start_date, end_date)
        if series is not None:
            raw[ticker] = series
        elif ticker not in AUX_TICKERS:
            failed.append(ticker)

    if not raw:
        return {"error": "No price data could be fetched for any ticker."}

    df = pd.DataFrame(raw)
    df = df.ffill(limit=3)

    # Trim to the requested period
    cutoff = end_date - timedelta(days=days)
    df = df[df.index >= pd.Timestamp(cutoff)]

    # Drop rows where any portfolio ticker is NaN
    portfolio_cols = [t for t in tickers if t in df.columns]
    df_aligned = df.dropna(subset=portfolio_cols)

    # Coverage warnings
    raw_df = pd.DataFrame({k: v for k, v in raw.items()})
    raw_df = raw_df[raw_df.index >= pd.Timestamp(cutoff)]
    coverage_warnings: List[str] = []
    for col in portfolio_cols:
        if col in raw_df.columns:
            pct = raw_df[col].isna().mean()
            if pct > 0.05:
                coverage_warnings.append(f"{col}: {pct:.1%} missing after alignment")

    return {
        "prices": df_aligned,
        "failed_tickers": failed,
        "warnings": coverage_warnings,
        "coverage_start": df_aligned.index[0].strftime("%Y-%m-%d") if len(df_aligned) > 0 else None,
        "coverage_end":   df_aligned.index[-1].strftime("%Y-%m-%d") if len(df_aligned) > 0 else None,
        "trading_days":   int(len(df_aligned)),
        "available_tickers": list(df_aligned.columns),
    }


def compute_portfolio_returns(
    prices_df: pd.DataFrame, weights: Dict[str, float]
) -> pd.Series:
    returns = prices_df.pct_change().dropna()
    available = [t for t in weights if t in returns.columns]
    if not available:
        return pd.Series(dtype=float)
    total_w = sum(weights[t] for t in available)
    port = sum(returns[t] * (weights[t] / total_w) for t in available)
    port.name = "portfolio"
    return port


def _max_drawdown_stats(returns: pd.Series) -> Dict[str, Any]:
    cum = (1 + returns).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max

    max_dd = float(dd.min())
    if np.isnan(max_dd) or len(dd) == 0:
        return {"depth": 0.0}

    trough = dd.idxmin()
    peak   = cum[:trough].idxmax()

    after  = cum[trough:]
    peak_v = float(cum[peak])
    rec_mask = after >= peak_v
    recovery = rec_mask.index[rec_mask].min() if rec_mask.any() else None

    return {
        "depth": round(max_dd, 6),
        "peak_date":    peak.strftime("%Y-%m-%d"),
        "trough_date":  trough.strftime("%Y-%m-%d"),
        "duration_days": int((trough - peak).days),
        "recovery_date": recovery.strftime("%Y-%m-%d") if recovery else "Not recovered",
        "recovery_days": int((recovery - trough).days) if recovery else None,
    }


# ─── Module 1: Rolling Correlations ──────────────────────────────────────────

def compute_correlations(
    prices_df: pd.DataFrame, weights: Dict[str, float], window: int = 90
) -> Dict[str, Any]:
    returns = prices_df.pct_change().dropna()
    portfolio_tickers = [t for t in weights if t in returns.columns]
    if len(portfolio_tickers) < 2:
        return {"error": "Need at least 2 tickers for correlation analysis."}

    ret = returns[portfolio_tickers]
    full_corr = ret.corr()

    pairs: List[Dict] = []
    for i in range(len(portfolio_tickers)):
        for j in range(i + 1, len(portfolio_tickers)):
            ti, tj = portfolio_tickers[i], portfolio_tickers[j]
            val = float(full_corr.loc[ti, tj])
            if not np.isnan(val):
                pairs.append({"pair": [ti, tj], "correlation": round(val, 4)})

    pairs.sort(key=lambda x: x["correlation"], reverse=True)
    concentration_risk = [p for p in pairs if p["correlation"] > 0.75]

    rolling_sample: Dict = {}
    if len(portfolio_tickers) >= 2 and len(ret) >= window:
        t1 = pairs[0]["pair"][0] if pairs else portfolio_tickers[0]
        t2 = pairs[0]["pair"][1] if pairs else portfolio_tickers[1]
        roll = ret[t1].rolling(window).corr(ret[t2]).dropna()
        rolling_sample = {
            "dates":  [d.strftime("%Y-%m-%d") for d in roll.index],
            "values": [round(float(v), 4) for v in roll.values],
            "pair":   [t1, t2],
        }

    upper_vals = full_corr.values[np.triu_indices_from(full_corr.values, k=1)]
    upper_vals = upper_vals[~np.isnan(upper_vals)]

    return {
        "correlation_matrix": {
            col: {row: round(float(full_corr.loc[row, col]), 4) for row in full_corr.index}
            for col in full_corr.columns
        },
        "tickers": portfolio_tickers,
        "top_correlated":   pairs[:5],
        "least_correlated": list(reversed(pairs[-5:])),
        "concentration_risk_pairs": concentration_risk,
        "rolling_sample":   rolling_sample,
        "window": window,
        "stats": {
            "avg_correlation":    round(float(upper_vals.mean()), 4) if len(upper_vals) > 0 else 0,
            "max_correlation":    round(float(pairs[0]["correlation"]), 4) if pairs else 0,
            "n_concentration_risk": len(concentration_risk),
            "n_pairs": len(pairs),
        },
    }


# ─── Module 2: Regime Detection ───────────────────────────────────────────────

def _build_regimes(bench: pd.Series, window: int = 60) -> Tuple[pd.Series, float, float]:
    roll_ret = bench.rolling(window).mean() * window
    roll_vol = bench.rolling(window).std() * np.sqrt(252)
    vol_75th = float(roll_vol.quantile(0.75))
    vol_med  = float(roll_vol.median())

    labels = []
    for r, v in zip(roll_ret, roll_vol):
        if pd.isna(r) or pd.isna(v):
            labels.append(None)
        elif r < 0 and v > vol_75th:
            labels.append("Crisis")
        elif r < 0:
            labels.append("Bear")
        else:
            labels.append("Bull")

    regimes = pd.Series(labels, index=bench.index, name="regime").dropna()
    return regimes, vol_75th, vol_med


def detect_regimes(
    prices_df: pd.DataFrame, weights: Dict[str, float], benchmark: str = "ES3.SI"
) -> Dict[str, Any]:
    returns = prices_df.pct_change().dropna()
    if benchmark not in returns.columns:
        return {"error": f"Benchmark {benchmark} not found in price data."}

    bench = returns[benchmark]
    port  = compute_portfolio_returns(prices_df, weights)
    common = bench.index.intersection(port.index)
    bench, port = bench.loc[common], port.loc[common]

    regimes, vol_75th, vol_med = _build_regimes(bench)

    # Compress into contiguous periods
    regime_periods: List[Dict] = []
    if len(regimes) > 0:
        curr  = regimes.iloc[0]
        start = regimes.index[0]
        for date, reg in regimes.items():
            if reg != curr:
                regime_periods.append({"start": start.strftime("%Y-%m-%d"),
                                       "end":   date.strftime("%Y-%m-%d"),
                                       "regime": curr})
                curr, start = reg, date
        regime_periods.append({"start": start.strftime("%Y-%m-%d"),
                                "end":   regimes.index[-1].strftime("%Y-%m-%d"),
                                "regime": curr})

    cum = (1 + port).cumprod()

    def _rstats(name: str) -> Dict:
        mask = regimes == name
        if mask.sum() < 5:
            return {"days": 0}
        r = port.loc[mask.index[mask]]
        dd   = _max_drawdown_stats(r)
        vol  = float(r.std() * np.sqrt(252))
        sharpe = float(r.mean() * 252 / (vol + 1e-9))
        return {
            "avg_daily_return":  round(float(r.mean()), 6),
            "annualized_return": round(float(r.mean() * 252), 4),
            "volatility":        round(vol, 4),
            "max_drawdown":      round(dd.get("depth", 0.0), 4),
            "sharpe":            round(sharpe, 3),
            "days":              int(mask.sum()),
        }

    stats = {n: _rstats(n) for n in ["Bull", "Bear", "Crisis"]}

    portfolio_tickers = [t for t in weights if t in returns.columns]
    regime_corrs: Dict[str, float] = {}
    for name in ["Bull", "Bear", "Crisis"]:
        mask = regimes == name
        idx = mask.index[mask]
        if len(idx) >= 60:
            sub = returns.loc[returns.index.intersection(idx), portfolio_tickers]
            vals = sub.corr().values
            upper = vals[np.triu_indices_from(vals, k=1)]
            upper = upper[~np.isnan(upper)]
            regime_corrs[name] = round(float(upper.mean()), 4) if len(upper) > 0 else float("nan")

    total = max(len(regimes), 1)
    return {
        "regime_periods": regime_periods,
        "portfolio_cumulative_returns": {
            "dates":  [d.strftime("%Y-%m-%d") for d in cum.index],
            "values": [round(float(v), 6) for v in cum.values],
        },
        "regime_stats":        stats,
        "regime_correlations": regime_corrs,
        "regime_distribution": {
            n: round(int((regimes == n).sum()) / total, 3)
            for n in ["Bull", "Bear", "Crisis"]
        },
        "current_regime": str(regimes.iloc[-1]) if len(regimes) > 0 else "Unknown",
        "vol_thresholds": {"median": round(vol_med, 4), "p75": round(vol_75th, 4)},
    }


# ─── Module 3: Factor Exposure ────────────────────────────────────────────────

def compute_factor_exposure(
    prices_df: pd.DataFrame,
    weights: Dict[str, float],
    benchmark: str = "ES3.SI",
    window: int = 252,
) -> Dict[str, Any]:
    returns = prices_df.pct_change().dropna()
    port    = compute_portfolio_returns(prices_df, weights)

    factors: Dict[str, pd.Series] = {}

    if benchmark in returns.columns:
        factors["market_beta"] = returns[benchmark]

    if "SGDUSD=X" in returns.columns:
        factors["fx_sensitivity"] = returns["SGDUSD=X"]

    # SGD interest rate factor via SORA proxy waterfall
    start_dt = (prices_df.index.min() - timedelta(days=10)).to_pydatetime()
    end_dt   = prices_df.index.max().to_pydatetime()
    sora_chg, sora_source = fetch_sora_proxy(start_dt, end_dt)
    if sora_chg is None:
        return {"error": "SORA proxy could not be loaded: all fallback sources (SGS 2Y, SGS 10Y, A35.SI, ^IRX) failed. Please try again later."}
    sora_chg.name = "sgd_rate_sensitivity"
    factors["sgd_rate_sensitivity"] = sora_chg

    if not factors:
        return {"error": "No factor data available."}

    common_idx = port.index
    for s in factors.values():
        common_idx = common_idx.intersection(s.dropna().index)

    fdf = pd.DataFrame({k: v.loc[common_idx] for k, v in factors.items()}).dropna()
    port_a = port.loc[fdf.index]

    if len(fdf) < 60:
        return {"error": "Insufficient overlapping data for factor regression."}

    effective_window = min(window, max(60, len(fdf) - 10))

    # Add constant column
    const_col = np.ones((len(fdf), 1))
    X_full = np.hstack([const_col, fdf.values])
    col_names = ["const"] + list(fdf.columns)
    y = port_a.values.astype(float)

    res = _ols(X_full, y)

    factor_labels = {
        "market_beta":          "Market Beta (STI ETF)",
        "fx_sensitivity":       "FX Sensitivity (USD/SGD)",
        "sgd_rate_sensitivity": "SGD Rate Sensitivity (SORA Proxy)",
    }
    factor_desc_tpl = {
        "market_beta":          "Market Beta: {v:.2f} — portfolio moves ~{pct:.0f}% as much as the STI",
        "fx_sensitivity":       "FX Sensitivity: {v:.2f} — a 1% move in USD/SGD shifts portfolio by ~{pct:.2f}%",
        "sgd_rate_sensitivity": "SGD Rate Sensitivity: {v:.2f} — portfolio return shift per 1% move in rate proxy",
    }

    current_exposures: Dict[str, Any] = {}
    for i, name in enumerate(col_names):
        if name == "const":
            continue
        v = float(res["params"][i])
        desc = factor_desc_tpl.get(name, "{v:.2f}").format(v=v, pct=abs(v) * 100)
        current_exposures[name] = {
            "coef":    round(v, 4),
            "ci_low":  round(float(res["ci_low"][i]), 4),
            "ci_high": round(float(res["ci_high"][i]), 4),
            "label":   factor_labels.get(name, name),
            "description": desc,
        }

    # Rolling OLS
    X_fdf = pd.DataFrame(
        np.hstack([const_col, fdf.values]),
        index=fdf.index, columns=col_names,
    )
    roll_params = _rolling_ols_params(X_fdf, port_a, effective_window, step=10)

    rolling_exposures: Dict[str, Any] = {
        "dates": [d.strftime("%Y-%m-%d") for d in roll_params.index]
    }
    for name in factors:
        if name in roll_params.columns:
            rolling_exposures[name] = [round(float(v), 4) for v in roll_params[name].values]

    reit_tickers = {"C31.SI", "A17U.SI", "ME8U.SI"}
    total_w = max(sum(weights.values()), 1)
    reit_weight = sum(weights.get(t, 0) for t in reit_tickers) / total_w

    return {
        "current_exposures": current_exposures,
        "rolling_exposures": rolling_exposures,
        "reit_weight":      round(reit_weight, 4),
        "r_squared":        round(res["r2"], 4),
        "adj_r_squared":    round(res["adj_r2"], 4),
        "n_obs":            int(len(port_a)),
        "effective_window": effective_window,
        "sora_source":      sora_source,
    }


# ─── Module 4: Tail Risk ──────────────────────────────────────────────────────

def _var_cvar(returns: np.ndarray, confidence: float, horizon_days: int = 1) -> Tuple[float, float]:
    r = returns * np.sqrt(horizon_days) if horizon_days > 1 else returns
    cut = float(np.percentile(r, (1 - confidence) * 100))
    tail = r[r <= cut]
    cvar = float(tail.mean()) if len(tail) > 0 else cut
    return round(cut, 6), round(cvar, 6)


def compute_tail_risk(
    prices_df: pd.DataFrame, weights: Dict[str, float]
) -> Dict[str, Any]:
    port = compute_portfolio_returns(prices_df, weights).dropna()
    if len(port) < 60:
        return {"error": "Insufficient data for tail risk analysis."}

    rv = port.values
    var95_1d,  cvar95_1d  = _var_cvar(rv, 0.95, 1)
    var99_1d,  cvar99_1d  = _var_cvar(rv, 0.99, 1)
    var95_1m,  cvar95_1m  = _var_cvar(rv, 0.95, 21)
    var99_1m,  cvar99_1m  = _var_cvar(rv, 0.99, 21)

    max_dd = _max_drawdown_stats(port)

    # Rolling 252-day max drawdown
    roll_window = min(252, len(port) // 2)
    roll_dd_dates: List[str] = []
    roll_dd_vals:  List[float] = []
    for i in range(roll_window, len(port)):
        w_r = port.iloc[i - roll_window: i]
        cum = (1 + w_r).cumprod()
        dd  = (cum - cum.cummax()) / cum.cummax()
        roll_dd_dates.append(port.index[i].strftime("%Y-%m-%d"))
        roll_dd_vals.append(round(float(dd.min()), 6))

    hist_counts, hist_bins = np.histogram(port, bins=60)

    ind_returns = prices_df.pct_change().dropna()
    available   = [t for t in weights if t in ind_returns.columns]
    total_w     = max(sum(weights[t] for t in available), 1)

    worst_10_idx = port.nsmallest(10).index
    worst_10: List[Dict] = []
    for date in worst_10_idx:
        if date in ind_returns.index:
            contribs = {
                t: round(float(ind_returns.loc[date, t] * weights[t] / total_w), 6)
                for t in available if t in ind_returns.columns
            }
            worst_10.append({
                "date": date.strftime("%Y-%m-%d"),
                "portfolio_return": round(float(port[date]), 6),
                "contributors": contribs,
            })

    ann_vol = float(port.std() * np.sqrt(252))
    return {
        "var_95_1d":  var95_1d,  "var_99_1d":  var99_1d,
        "var_95_1m":  var95_1m,  "var_99_1m":  var99_1m,
        "cvar_95_1d": cvar95_1d, "cvar_99_1d": cvar99_1d,
        "cvar_95_1m": cvar95_1m, "cvar_99_1m": cvar99_1m,
        "max_drawdown": max_dd,
        "rolling_max_drawdown": {"dates": roll_dd_dates, "values": roll_dd_vals},
        "return_histogram": {
            "bins":   [round(float(b), 6) for b in hist_bins],
            "counts": [int(c) for c in hist_counts],
        },
        "worst_10_days": worst_10,
        "summary": {
            "worst_single_day":  round(float(port.min()), 6),
            "best_single_day":   round(float(port.max()), 6),
            "annualized_vol":    round(ann_vol, 4),
            "skewness":          round(float(port.skew()), 4),
            "kurtosis":          round(float(port.kurtosis()), 4),
            "n_observations":    int(len(port)),
        },
    }


# ─── Module 5.5: Benchmark Comparison ────────────────────────────────────────

def compute_benchmark_comparison(
    prices_df: pd.DataFrame,
    weights: Dict[str, float],
    benchmark: str = "ES3.SI",
) -> Dict[str, Any]:
    returns = prices_df.pct_change().dropna()
    if benchmark not in returns.columns:
        return {"error": f"Benchmark {benchmark} not found in price data."}

    port = compute_portfolio_returns(prices_df, weights).dropna()
    bench_ret = returns[benchmark]

    common = port.index.intersection(bench_ret.index)
    port  = port.loc[common]
    bench = bench_ret.loc[common]

    if len(port) < 60:
        return {"error": "Insufficient data for benchmark comparison."}

    port_cum  = (1 + port).cumprod()
    bench_cum = (1 + bench).cumprod()

    def _stats(r: pd.Series) -> Dict:
        ann_ret = float(r.mean() * 252)
        ann_vol = float(r.std() * np.sqrt(252))
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0
        dd      = _max_drawdown_stats(r)
        return {
            "annualized_return": round(ann_ret, 4),
            "annualized_vol":    round(ann_vol, 4),
            "sharpe":            round(sharpe,  3),
            "max_drawdown":      round(dd.get("depth", 0.0), 4),
        }

    port_stats  = _stats(port)
    bench_stats = _stats(bench)

    X   = np.column_stack([np.ones(len(bench)), bench.values])
    res = _ols(X, port.values)
    alpha_ann = float(res["params"][0]) * 252
    beta      = float(res["params"][1])

    active = port - bench
    te     = float(active.std() * np.sqrt(252))
    ir     = float(active.mean() * 252 / te) if te > 0 else 0.0

    # Rolling 252-day returns (stride 5)
    roll_window = min(252, len(port) - 1)
    roll_dates: List[str]  = []
    port_roll:  List[float] = []
    bench_roll: List[float] = []
    for i in range(roll_window, len(port) + 1, 5):
        sub_p = port.iloc[i - roll_window: i]
        sub_b = bench.iloc[i - roll_window: i]
        roll_dates.append(port.index[i - 1].strftime("%Y-%m-%d"))
        port_roll.append(round(float((1 + sub_p).prod() - 1), 4))
        bench_roll.append(round(float((1 + sub_b).prod() - 1), 4))

    # Calendar-year returns
    yearly: Dict[str, Dict] = {}
    for year in sorted(port.index.year.unique()):
        py = port[port.index.year == year]
        by = bench[bench.index.year == year]
        if len(py) > 30:
            yearly[str(year)] = {
                "portfolio": round(float((1 + py).prod() - 1), 4),
                "benchmark": round(float((1 + by).prod() - 1), 4),
            }

    return {
        "dates":                  [d.strftime("%Y-%m-%d") for d in port_cum.index],
        "portfolio_cumulative":   [round(float(v), 6) for v in port_cum.values],
        "benchmark_cumulative":   [round(float(v), 6) for v in bench_cum.values],
        "portfolio_stats":        port_stats,
        "benchmark_stats":        bench_stats,
        "alpha":                  round(alpha_ann, 4),
        "beta":                   round(beta, 4),
        "tracking_error":         round(te, 4),
        "information_ratio":      round(ir, 3),
        "r_squared":              round(res["r2"], 4),
        "rolling_12m": {
            "dates":     roll_dates,
            "portfolio": port_roll,
            "benchmark": bench_roll,
        },
        "calendar_year_returns":  yearly,
        "benchmark_ticker":       benchmark,
    }


# ─── Module 5: Historical Hedging Offsets ─────────────────────────────────────

def compute_hedging_offsets(
    prices_df: pd.DataFrame,
    weights: Dict[str, float],
    benchmark: str = "ES3.SI",
) -> Dict[str, Any]:
    returns = prices_df.pct_change().dropna()
    port    = compute_portfolio_returns(prices_df, weights).dropna()

    regimes = pd.Series(dtype=str)
    if benchmark in returns.columns:
        bench = returns[benchmark].loc[returns.index.intersection(port.index)]
        raw_reg, vol_75th, _ = _build_regimes(bench)
        regimes = raw_reg

    worst_20_idx = port.nsmallest(20).index

    hedge_specs = [
        ("Gold (SGD)",           "GC=F",              True),
        ("Short STI ETF",        f"SHORT_{benchmark}", False),
        ("Asia ex-Japan (AAXJ)", "AAXJ",               False),
        ("SGD Cash (0%)",        "CASH",               False),
        ("Asia REIT (AXJR)",     "AXJR",               False),
    ]

    results: Dict[str, Any] = {}

    for hedge_name, ticker, needs_fx in hedge_specs:
        try:
            if ticker == "CASH":
                hedge_ret = pd.Series(0.0, index=port.index, dtype=float)

            elif ticker.startswith("SHORT_"):
                base = ticker.replace("SHORT_", "")
                if base not in returns.columns:
                    continue
                hedge_ret = -returns[base]

            elif ticker in returns.columns:
                hedge_ret = returns[ticker].copy()
                if needs_fx and "SGDUSD=X" in returns.columns:
                    fx_ret = returns["SGDUSD=X"]
                    ci = hedge_ret.index.intersection(fx_ret.index)
                    hedge_ret = hedge_ret.loc[ci] - fx_ret.loc[ci]
            else:
                continue

            common_idx = port.index.intersection(hedge_ret.index)
            if len(common_idx) < 60:
                continue

            p = port.loc[common_idx]
            h = hedge_ret.loc[common_idx]

            full_corr = round(float(p.corr(h)), 4)

            bear_corr: Optional[float] = None
            crisis_corr: Optional[float] = None
            if len(regimes) > 0:
                creg = common_idx.intersection(regimes.index)
                bear_idx  = creg[regimes.loc[creg] == "Bear"]
                cris_idx  = creg[regimes.loc[creg] == "Crisis"]
                if len(bear_idx) >= 15:
                    bear_corr = round(float(p.loc[bear_idx].corr(h.loc[bear_idx])), 4)
                if len(cris_idx) >= 10:
                    crisis_corr = round(float(p.loc[cris_idx].corr(h.loc[cris_idx])), 4)

            w20 = worst_20_idx.intersection(h.index)
            avg_worst20 = round(float(h.loc[w20].mean()), 6) if len(w20) > 0 else None

            p_vol = float(p.std() * np.sqrt(252))
            h_vol = float(h.std() * np.sqrt(252))

            def blended_vol(alpha: float) -> float:
                return float(np.sqrt(
                    (1 - alpha) ** 2 * p_vol ** 2
                    + alpha ** 2 * h_vol ** 2
                    + 2 * (1 - alpha) * alpha * full_corr * p_vol * h_vol
                ))

            results[hedge_name] = {
                "full_period_correlation":      full_corr,
                "bear_correlation":             bear_corr,
                "crisis_correlation":           crisis_corr,
                "avg_return_worst20_days":      avg_worst20,
                "annualized_vol":               round(h_vol, 4),
                "portfolio_vol_reduction_10pct": round(p_vol - blended_vol(0.10), 4),
                "portfolio_vol_reduction_20pct": round(p_vol - blended_vol(0.20), 4),
                "available": True,
            }

        except Exception as exc:
            results[hedge_name] = {"available": False, "error": str(exc)}

    return {
        "instruments":    results,
        "portfolio_vol":  round(float(port.std() * np.sqrt(252)), 4),
        "n_bear_days":    int((regimes == "Bear").sum())   if len(regimes) > 0 else 0,
        "n_crisis_days":  int((regimes == "Crisis").sum()) if len(regimes) > 0 else 0,
        "worst_20_dates": [d.strftime("%Y-%m-%d") for d in worst_20_idx],
    }


# ─── Module 7: Portfolio Construction Scenarios ───────────────────────────────

RF_ANNUAL = 0.035   # SGD risk-free rate proxy


def _portfolio_metrics(returns: pd.Series) -> Dict[str, float]:
    """Compute the five baseline metrics for a returns series."""
    if len(returns) < 10:
        return {"return": 0.0, "vol": 0.0, "sharpe": 0.0, "cvar_99": 0.0, "max_drawdown": 0.0}
    ann_ret = float(returns.mean() * 252)
    ann_vol = float(returns.std() * np.sqrt(252))
    rf_daily = RF_ANNUAL / 252
    sharpe = (returns.mean() - rf_daily) / (returns.std() + 1e-9) * np.sqrt(252)
    cut = float(np.percentile(returns.values, 1))
    tail = returns.values[returns.values <= cut]
    cvar_99 = float(tail.mean()) if len(tail) > 0 else cut
    dd = _max_drawdown_stats(returns)
    return {
        "return":       round(ann_ret, 6),
        "vol":          round(ann_vol, 6),
        "sharpe":       round(float(sharpe), 4),
        "cvar_99":      round(cvar_99, 6),
        "max_drawdown": round(dd.get("depth", 0.0), 6),
    }


def _stress_corr_with_portfolio(
    candidate_returns: pd.Series,
    port_returns: pd.Series,
    regimes: pd.Series,
) -> float:
    """Average correlation of candidate with portfolio during Bear + Crisis regimes."""
    if len(regimes) == 0:
        common = candidate_returns.index.intersection(port_returns.index)
        if len(common) < 10:
            return 0.0
        return float(candidate_returns.loc[common].corr(port_returns.loc[common]))
    stress_idx = regimes.index[regimes.isin(["Bear", "Crisis"])]
    ci = stress_idx.intersection(candidate_returns.index).intersection(port_returns.index)
    if len(ci) < 10:
        return float(candidate_returns.corr(port_returns))
    return float(candidate_returns.loc[ci].corr(port_returns.loc[ci]))


def _optimise_weights(
    tickers: List[str],
    returns_df: pd.DataFrame,
    objective: str,
) -> Dict[str, float]:
    """
    Run PyPortfolioOpt with Ledoit-Wolf shrinkage covariance.
    Constraints: max 25% per holding, max 40% per sector (not enforced here — done at caller),
    min 2% per holding.
    Returns dict of {ticker: weight} or equal-weight fallback on failure.
    """
    try:
        from pypfopt import EfficientFrontier, risk_models, expected_returns
        from pypfopt import objective_functions

        sub = returns_df[tickers].dropna()
        if len(sub) < 60 or len(tickers) < 2:
            raise ValueError("Insufficient data for optimisation")

        mu = expected_returns.mean_historical_return(sub, returns_data=True, frequency=252)
        S  = risk_models.CovarianceShrinkage(sub, returns_data=True, frequency=252).ledoit_wolf()

        ef = EfficientFrontier(mu, S, weight_bounds=(0.02, 0.25))

        if objective == "min_vol":
            ef.min_volatility()
        elif objective == "cvar":
            # CVaR optimisation needs scipy — fall back to max Sharpe if unavailable
            try:
                ef.add_objective(objective_functions.L2_reg, gamma=0.1)
                ef.min_volatility()
            except Exception:
                ef.max_sharpe(risk_free_rate=RF_ANNUAL)
        else:  # sharpe (default)
            ef.max_sharpe(risk_free_rate=RF_ANNUAL)

        weights = ef.clean_weights(cutoff=0.02, rounding=4)
        return {k: v for k, v in weights.items() if v > 0.001}

    except Exception as exc:
        # Equal-weight fallback
        w = round(1.0 / len(tickers), 4)
        return {t: w for t in tickers}


def run_construction_analysis(
    current_portfolio: Dict[str, float],
    returns_data: pd.DataFrame,
    universe_returns: pd.DataFrame,
    universe_metadata: Dict[str, Any],
    objective: str,
    pass1_shortlist: Optional[Dict[str, Any]],
    regime_series: Optional[pd.Series] = None,
    estimated_portfolio_size_sgd: float = 10_000_000,
) -> Dict[str, Any]:
    """
    Two-pass portfolio construction analysis.

    pass1_shortlist: output of Claude Pass 1 with keys:
        addition_candidates: list of {"ticker": ..., "rationale": ...}
        removal_candidates:  list of {"ticker": ..., "rationale": ...}
    If None, runs on full universe (fallback mode).
    """
    port_tickers = [t for t in current_portfolio if t in returns_data.columns]
    if not port_tickers:
        return {"error": "No portfolio tickers found in returns data."}

    total_w = sum(current_portfolio[t] for t in port_tickers)
    norm_weights = {t: current_portfolio[t] / total_w for t in port_tickers}

    port_returns = compute_portfolio_returns(returns_data, norm_weights)
    if len(port_returns) < 60:
        return {"error": "Insufficient portfolio return history."}

    # ── Step 1: Baseline metrics ──
    baseline = _portfolio_metrics(port_returns)

    # ── Regimes for stress correlation ──
    regimes = regime_series if regime_series is not None else pd.Series(dtype=str)

    # ── Determine candidates ──
    if pass1_shortlist is not None:
        add_candidates = [c["ticker"] for c in pass1_shortlist.get("addition_candidates", [])]
        rem_candidates = [c["ticker"] for c in pass1_shortlist.get("removal_candidates", [])]
        pass1_rationales = {
            c["ticker"]: c.get("rationale", "") for c in
            pass1_shortlist.get("addition_candidates", []) + pass1_shortlist.get("removal_candidates", [])
        }
    else:
        # Fallback: use all available universe tickers not in portfolio
        add_candidates = [
            t for t in universe_metadata
            if t not in port_tickers and t in universe_returns.columns
        ][:12]
        rem_candidates = list(port_tickers)[:6]
        pass1_rationales = {}

    # Filter candidates to those with available returns
    add_candidates = [t for t in add_candidates if t in universe_returns.columns]
    rem_candidates = [t for t in rem_candidates if t in port_tickers]

    # Align returns index
    common_idx = port_returns.index
    if len(universe_returns) > 0:
        common_idx = common_idx.intersection(universe_returns.index)

    all_returns = pd.concat([
        returns_data[port_tickers],
        universe_returns[[t for t in add_candidates if t in universe_returns.columns]],
    ], axis=1).loc[common_idx].dropna(how="all")

    port_returns_aligned = all_returns[port_tickers].apply(
        lambda col: col * norm_weights.get(col.name, 0), axis=0
    ).sum(axis=1)

    # ── Step 2: Marginal removal analysis ──
    removal_scores: List[Dict] = []
    for ticker in rem_candidates:
        try:
            remaining = [t for t in port_tickers if t != ticker]
            if not remaining:
                continue
            rem_total = sum(norm_weights[t] for t in remaining)
            new_weights = {t: norm_weights[t] / rem_total for t in remaining}
            sub_ret = all_returns[remaining].dropna()
            if len(sub_ret) < 30:
                continue
            new_port = sum(sub_ret[t] * new_weights[t] for t in remaining)
            new_met = _portfolio_metrics(new_port)

            stress_c = _stress_corr_with_portfolio(
                all_returns[ticker].dropna() if ticker in all_returns.columns else port_returns_aligned,
                port_returns_aligned,
                regimes,
            )

            # Improvement score: higher is better for removal
            if objective == "min_vol":
                obj_improvement = baseline["vol"] - new_met["vol"]
            elif objective == "cvar":
                obj_improvement = baseline["cvar_99"] - new_met["cvar_99"]
            else:
                obj_improvement = new_met["sharpe"] - baseline["sharpe"]

            score = obj_improvement * 0.6 + stress_c * 0.4

            removal_scores.append({
                "ticker":           ticker,
                "metrics_delta":    {k: round(new_met[k] - baseline[k], 6) for k in baseline},
                "stress_corr":      round(stress_c, 4),
                "score":            round(score, 6),
            })
        except Exception:
            continue

    removal_scores.sort(key=lambda x: x["score"], reverse=True)

    # ── Step 3: Marginal addition analysis ──
    addition_scores: List[Dict] = []
    small_portfolio = estimated_portfolio_size_sgd < 10_000_000

    for ticker in add_candidates:
        try:
            meta = universe_metadata.get(ticker, {})
            if small_portfolio and meta.get("liquidity_warning", False):
                continue

            if ticker not in all_returns.columns:
                continue

            # Add at 5%, funded by trimming current holdings proportionally
            add_weight = 0.05
            scale = 1.0 - add_weight
            new_weights = {t: norm_weights[t] * scale for t in port_tickers}
            new_weights[ticker] = add_weight

            sub_cols = [t for t in list(new_weights.keys()) if t in all_returns.columns]
            sub_ret = all_returns[sub_cols].dropna()
            if len(sub_ret) < 30:
                continue

            new_port = sum(sub_ret[t] * new_weights[t] for t in sub_cols if t in sub_ret.columns)
            new_met = _portfolio_metrics(new_port)

            stress_c = _stress_corr_with_portfolio(
                all_returns[ticker].dropna(),
                port_returns_aligned,
                regimes,
            )

            if objective == "min_vol":
                obj_improvement = baseline["vol"] - new_met["vol"]
            elif objective == "cvar":
                obj_improvement = baseline["cvar_99"] - new_met["cvar_99"]
            else:
                obj_improvement = new_met["sharpe"] - baseline["sharpe"]

            # Negative stress correlation is desirable for diversification
            score = obj_improvement * 0.6 + (-stress_c) * 0.4

            addition_scores.append({
                "ticker":           ticker,
                "metrics_delta":    {k: round(new_met[k] - baseline[k], 6) for k in baseline},
                "stress_corr":      round(stress_c, 4),
                "score":            round(score, 6),
                "liquidity_warning": meta.get("liquidity_warning", False),
            })
        except Exception:
            continue

    addition_scores.sort(key=lambda x: x["score"], reverse=True)

    # ── Step 4: Construct three scenarios ──
    def _build_scenario(
        n_rem: int, n_add: int, label: str, tag: str
    ) -> Dict[str, Any]:
        top_rem = removal_scores[:n_rem]
        top_add = addition_scores[:n_add]

        rem_tickers = [r["ticker"] for r in top_rem]
        add_tickers = [a["ticker"] for a in top_add]

        holding_set = [t for t in port_tickers if t not in rem_tickers] + add_tickers
        holding_set = [t for t in holding_set if t in all_returns.columns]

        if not holding_set:
            holding_set = port_tickers[:]

        opt_weights = _optimise_weights(holding_set, all_returns, objective)

        # Enforce sector constraint: no sector > 40%
        sector_totals: Dict[str, float] = {}
        for t, w in opt_weights.items():
            sec = universe_metadata.get(t, {}).get("sector", "Unknown")
            sector_totals[sec] = sector_totals.get(sec, 0) + w

        for sec, total in sector_totals.items():
            if total > 0.40:
                scale = 0.40 / total
                for t in list(opt_weights.keys()):
                    if universe_metadata.get(t, {}).get("sector", "") == sec:
                        opt_weights[t] = round(opt_weights[t] * scale, 4)

        # Re-normalise
        wsum = sum(opt_weights.values())
        if wsum > 0:
            opt_weights = {t: round(w / wsum, 4) for t, w in opt_weights.items()}

        # After metrics
        sub_cols_s = [t for t in opt_weights if t in all_returns.columns]
        sub_ret_s = all_returns[sub_cols_s].dropna()
        if len(sub_ret_s) >= 30:
            new_port_s = sum(sub_ret_s[t] * opt_weights.get(t, 0) for t in sub_cols_s)
            metrics_after = _portfolio_metrics(new_port_s)
        else:
            metrics_after = baseline.copy()

        metrics_delta = {k: round(metrics_after[k] - baseline[k], 6) for k in baseline}

        additions_out = []
        for a in top_add:
            meta = universe_metadata.get(a["ticker"], {})
            additions_out.append({
                "ticker":           a["ticker"],
                "name":             meta.get("name", a["ticker"]),
                "rationale":        pass1_rationales.get(a["ticker"], ""),
                "liquidity_warning": meta.get("liquidity_warning", False),
            })

        removals_out = []
        for r in top_rem:
            meta = universe_metadata.get(r["ticker"], {})
            removals_out.append({
                "ticker":   r["ticker"],
                "name":     meta.get("name", r["ticker"]),
                "rationale": pass1_rationales.get(r["ticker"], ""),
            })

        liq_warnings = [a["ticker"] for a in additions_out if a["liquidity_warning"]]

        return {
            "label":            label,
            "additions":        additions_out,
            "removals":         removals_out,
            "proposed_weights": opt_weights,
            "metrics_before":   baseline,
            "metrics_after":    metrics_after,
            "metrics_delta":    metrics_delta,
            "liquidity_warnings": liq_warnings,
            "ai_interpretation": "",  # filled by Pass 3
        }

    scenarios = {
        "A": _build_scenario(1, 1, "Conservative", "A"),
        "B": _build_scenario(min(2, len(removal_scores)), min(2, len(addition_scores)), "Moderate", "B"),
        "C": _build_scenario(min(3, len(removal_scores)), min(3, len(addition_scores)), "Aggressive", "C"),
    }

    return {
        "baseline_metrics": baseline,
        "scenarios":        scenarios,
        "removal_ranked":   removal_scores,
        "addition_ranked":  addition_scores,
    }
