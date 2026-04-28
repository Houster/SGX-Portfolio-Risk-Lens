"""
SGX universe definition and startup pre-fetch.
50 tickers: STI 30 + Next 20 by approximate market cap.
Pre-fetches 3 years of daily adjusted close prices once at app startup.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import pandas as pd
from analysis import _download_one

logger = logging.getLogger(__name__)

# ─── Universe Definition ──────────────────────────────────────────────────────

UNIVERSE: Dict[str, Dict[str, Any]] = {
    # ── STI 30 ──
    "D05.SI":  {"name": "DBS Group Holdings",           "sector": "Banking",     "market_cap_tier": "STI30",  "min_adv_sgd": 120_000_000, "liquidity_warning": False},
    "O39.SI":  {"name": "OCBC Bank",                    "sector": "Banking",     "market_cap_tier": "STI30",  "min_adv_sgd":  80_000_000, "liquidity_warning": False},
    "U11.SI":  {"name": "UOB",                          "sector": "Banking",     "market_cap_tier": "STI30",  "min_adv_sgd":  60_000_000, "liquidity_warning": False},
    "Z74.SI":  {"name": "Singtel",                      "sector": "Telco",       "market_cap_tier": "STI30",  "min_adv_sgd":  40_000_000, "liquidity_warning": False},
    "S68.SI":  {"name": "Singapore Exchange",           "sector": "Financials",  "market_cap_tier": "STI30",  "min_adv_sgd":  25_000_000, "liquidity_warning": False},
    "C31.SI":  {"name": "CapitaLand Integrated Commercial Trust", "sector": "REIT", "market_cap_tier": "STI30", "min_adv_sgd": 30_000_000, "liquidity_warning": False},
    "A17U.SI": {"name": "Ascendas REIT",                "sector": "REIT",        "market_cap_tier": "STI30",  "min_adv_sgd":  35_000_000, "liquidity_warning": False},
    "ME8U.SI": {"name": "Mapletree Industrial Trust",   "sector": "REIT",        "market_cap_tier": "STI30",  "min_adv_sgd":  18_000_000, "liquidity_warning": False},
    "BN4.SI":  {"name": "Keppel Corporation",           "sector": "Industrials", "market_cap_tier": "STI30",  "min_adv_sgd":  20_000_000, "liquidity_warning": False},
    "G13.SI":  {"name": "Genting Singapore",            "sector": "Consumer",    "market_cap_tier": "STI30",  "min_adv_sgd":  22_000_000, "liquidity_warning": False},
    "C09.SI":  {"name": "City Developments",            "sector": "Property",    "market_cap_tier": "STI30",  "min_adv_sgd":  10_000_000, "liquidity_warning": False},
    "BS6.SI":  {"name": "YZJ Shipbuilding",             "sector": "Industrials", "market_cap_tier": "STI30",  "min_adv_sgd":  15_000_000, "liquidity_warning": False},
    "F34.SI":  {"name": "Wilmar International",         "sector": "Consumer",    "market_cap_tier": "STI30",  "min_adv_sgd":  12_000_000, "liquidity_warning": False},
    "H78.SI":  {"name": "Hongkong Land Holdings",       "sector": "Property",    "market_cap_tier": "STI30",  "min_adv_sgd":   5_000_000, "liquidity_warning": False},
    "C6L.SI":  {"name": "Singapore Airlines",           "sector": "Transport",   "market_cap_tier": "STI30",  "min_adv_sgd":  30_000_000, "liquidity_warning": False},
    "J36.SI":  {"name": "Jardine Matheson",             "sector": "Industrials", "market_cap_tier": "STI30",  "min_adv_sgd":   4_000_000, "liquidity_warning": False},
    "N2IU.SI": {"name": "Mapletree Pan Asia Commercial Trust", "sector": "REIT", "market_cap_tier": "STI30",  "min_adv_sgd":  20_000_000, "liquidity_warning": False},
    "BUOU.SI": {"name": "Frasers Logistics & Commercial Trust", "sector": "REIT","market_cap_tier": "STI30",  "min_adv_sgd":   8_000_000, "liquidity_warning": False},
    "K71U.SI": {"name": "Keppel REIT",                  "sector": "REIT",        "market_cap_tier": "STI30",  "min_adv_sgd":   7_000_000, "liquidity_warning": False},
    "T82U.SI": {"name": "Suntec REIT",                  "sector": "REIT",        "market_cap_tier": "STI30",  "min_adv_sgd":   8_000_000, "liquidity_warning": False},
    "U96.SI":  {"name": "Sembcorp Industries",          "sector": "Industrials", "market_cap_tier": "STI30",  "min_adv_sgd":  12_000_000, "liquidity_warning": False},
    "V03.SI":  {"name": "Venture Corporation",          "sector": "Technology",  "market_cap_tier": "STI30",  "min_adv_sgd":   6_000_000, "liquidity_warning": False},
    "Y92.SI":  {"name": "Thai Beverage",                "sector": "Consumer",    "market_cap_tier": "STI30",  "min_adv_sgd":  10_000_000, "liquidity_warning": False},
    "9CI.SI":  {"name": "CapitaLand Investment",        "sector": "Financials",  "market_cap_tier": "STI30",  "min_adv_sgd":  15_000_000, "liquidity_warning": False},
    "S58.SI":  {"name": "SATS",                         "sector": "Transport",   "market_cap_tier": "STI30",  "min_adv_sgd":   8_000_000, "liquidity_warning": False},
    "M44U.SI": {"name": "Mapletree Logistics Trust",    "sector": "REIT",        "market_cap_tier": "STI30",  "min_adv_sgd":  12_000_000, "liquidity_warning": False},
    "RW0U.SI": {"name": "Frasers Centrepoint Trust",   "sector": "REIT",        "market_cap_tier": "STI30",  "min_adv_sgd":   5_000_000, "liquidity_warning": False},
    "AJBU.SI": {"name": "Keppel Infrastructure Trust", "sector": "Industrials", "market_cap_tier": "STI30",  "min_adv_sgd":   3_000_000, "liquidity_warning": False},
    "U14.SI":  {"name": "UOL Group",                    "sector": "Property",    "market_cap_tier": "STI30",  "min_adv_sgd":   4_000_000, "liquidity_warning": False},
    "C52.SI":  {"name": "ComfortDelGro",                "sector": "Transport",   "market_cap_tier": "STI30",  "min_adv_sgd":   8_000_000, "liquidity_warning": False},

    # ── Next 20 by approximate market cap ──
    "S63.SI":  {"name": "Singapore Technologies Engineering", "sector": "Industrials", "market_cap_tier": "Next20", "min_adv_sgd": 8_000_000, "liquidity_warning": False},
    "RE4.SI":  {"name": "GLP J-REIT",                  "sector": "REIT",        "market_cap_tier": "Next20", "min_adv_sgd":   1_500_000, "liquidity_warning": False},
    "5E2.SI":  {"name": "BRC Asia",                     "sector": "Industrials", "market_cap_tier": "Next20", "min_adv_sgd":     800_000, "liquidity_warning": True},
    "AWX.SI":  {"name": "AEM Holdings",                 "sector": "Technology",  "market_cap_tier": "Next20", "min_adv_sgd":   2_000_000, "liquidity_warning": False},
    "OV8.SI":  {"name": "Sheng Siong Group",            "sector": "Consumer",    "market_cap_tier": "Next20", "min_adv_sgd":   3_000_000, "liquidity_warning": False},
    "1D0.SI":  {"name": "Nanofilm Technologies",        "sector": "Technology",  "market_cap_tier": "Next20", "min_adv_sgd":   1_200_000, "liquidity_warning": False},
    "BVA.SI":  {"name": "Banyan Tree Holdings",         "sector": "Consumer",    "market_cap_tier": "Next20", "min_adv_sgd":     600_000, "liquidity_warning": True},
    "TS0.SI":  {"name": "Tuan Sing Holdings",           "sector": "Property",    "market_cap_tier": "Next20", "min_adv_sgd":     400_000, "liquidity_warning": True},
    "CLN.SI":  {"name": "Capitaland Ascendas India Trust","sector": "REIT",      "market_cap_tier": "Next20", "min_adv_sgd":   3_000_000, "liquidity_warning": False},
    "5DM.SI":  {"name": "Dyna-Mac Holdings",            "sector": "Industrials", "market_cap_tier": "Next20", "min_adv_sgd":     500_000, "liquidity_warning": True},
    "558.SI":  {"name": "HRnetGroup",                   "sector": "Financials",  "market_cap_tier": "Next20", "min_adv_sgd":     800_000, "liquidity_warning": True},
    "5IF.SI":  {"name": "Samudera Shipping Line",       "sector": "Transport",   "market_cap_tier": "Next20", "min_adv_sgd":     700_000, "liquidity_warning": True},
    "F9D.SI":  {"name": "Centurion Corporation",        "sector": "Property",    "market_cap_tier": "Next20", "min_adv_sgd":     500_000, "liquidity_warning": True},
    "43B.SI":  {"name": "Marco Polo Marine",            "sector": "Industrials", "market_cap_tier": "Next20", "min_adv_sgd":     400_000, "liquidity_warning": True},
    "42S.SI":  {"name": "Food Empire Holdings",         "sector": "Consumer",    "market_cap_tier": "Next20", "min_adv_sgd":     900_000, "liquidity_warning": True},
    "CY6U.SI": {"name": "Cromwell European REIT",       "sector": "REIT",        "market_cap_tier": "Next20", "min_adv_sgd":   1_000_000, "liquidity_warning": False},
    "8K7.SI":  {"name": "Intraco",                      "sector": "Industrials", "market_cap_tier": "Next20", "min_adv_sgd":     200_000, "liquidity_warning": True},
    "Q5T.SI":  {"name": "Acromec",                      "sector": "Industrials", "market_cap_tier": "Next20", "min_adv_sgd":     150_000, "liquidity_warning": True},
    "5TP.SI":  {"name": "CSE Global",                   "sector": "Technology",  "market_cap_tier": "Next20", "min_adv_sgd":     800_000, "liquidity_warning": True},
    "5WA.SI":  {"name": "Challenger Technologies",      "sector": "Consumer",    "market_cap_tier": "Next20", "min_adv_sgd":     300_000, "liquidity_warning": True},
}

ALL_TICKERS = list(UNIVERSE.keys())

# ─── In-memory price cache ─────────────────────────────────────────────────────

# Populated once at startup by prefetch_universe()
_universe_prices: Optional[pd.DataFrame] = None
_universe_returns: Optional[pd.DataFrame] = None
_unavailable_tickers: list = []


def get_universe_prices() -> Optional[pd.DataFrame]:
    return _universe_prices


def get_universe_returns() -> Optional[pd.DataFrame]:
    return _universe_returns


def get_unavailable_tickers() -> list:
    return _unavailable_tickers


def get_available_universe_metadata(exclude_tickers: list = None) -> Dict[str, Dict[str, Any]]:
    """Return universe metadata for tickers that were successfully pre-fetched."""
    exclude = set(exclude_tickers or [])
    unavail = set(_unavailable_tickers)
    return {
        k: v for k, v in UNIVERSE.items()
        if k not in unavail and k not in exclude
    }


# ─── Startup Pre-fetch ────────────────────────────────────────────────────────

def prefetch_universe(timeout_seconds: int = 60) -> None:
    """
    Pre-fetch 3 years of daily adjusted close prices for all universe tickers.
    Called once at app startup. Never raises — individual failures are logged and excluded.
    """
    global _universe_prices, _universe_returns, _unavailable_tickers

    import signal
    import threading

    end_date = datetime.now()
    start_date = end_date - timedelta(days=3 * 365 + 90)

    raw: Dict[str, pd.Series] = {}
    excluded: list = []

    completed_event = threading.Event()
    timed_out = [False]

    def _do_fetch():
        for i, ticker in enumerate(ALL_TICKERS):
            if timed_out[0]:
                logger.warning(f"Universe pre-fetch timed out — skipping remaining tickers from {ticker}")
                for remaining in ALL_TICKERS[i:]:
                    excluded.append(remaining)
                break
            try:
                cutoff = end_date - timedelta(days=3 * 365)
                series = _download_one(ticker, start_date, end_date)

                if series is None or len(series) < 20:
                    logger.info(f"Universe pre-fetch: {ticker} — insufficient data, excluding")
                    excluded.append(ticker)
                    logger.info(f"Fetching universe data: {i+1}/{len(ALL_TICKERS)} complete.")
                    continue

                trimmed = series[series.index >= pd.Timestamp(cutoff)]
                if len(trimmed) > 0:
                    missing_pct = trimmed.isna().mean()
                    if missing_pct > 0.10:
                        logger.info(f"Universe pre-fetch: {ticker} — {missing_pct:.1%} missing data, excluding")
                        excluded.append(ticker)
                        logger.info(f"Fetching universe data: {i+1}/{len(ALL_TICKERS)} complete.")
                        continue

                raw[ticker] = trimmed
                logger.info(f"Fetching universe data: {i+1}/{len(ALL_TICKERS)} complete.")

            except Exception as exc:
                logger.info(f"Universe pre-fetch: {ticker} failed ({exc}), excluding")
                excluded.append(ticker)
                logger.info(f"Fetching universe data: {i+1}/{len(ALL_TICKERS)} complete.")

        completed_event.set()

    fetch_thread = threading.Thread(target=_do_fetch, daemon=True)
    fetch_thread.start()

    finished = completed_event.wait(timeout=timeout_seconds)
    if not finished:
        timed_out[0] = True
        logger.warning(
            f"Universe pre-fetch exceeded {timeout_seconds}s — "
            f"some tickers will be marked unavailable."
        )
        completed_event.wait(timeout=30)  # give thread time to register remaining as excluded

    if raw:
        df_all = pd.DataFrame(raw)
        df_all = df_all.ffill(limit=3)
        _universe_prices = df_all
        _universe_returns = df_all.pct_change().dropna()

    _unavailable_tickers = excluded
    avail = len(ALL_TICKERS) - len(excluded)
    logger.info(
        f"Universe pre-fetch complete: {avail}/{len(ALL_TICKERS)} tickers available. "
        f"Excluded: {excluded if excluded else 'none'}"
    )
