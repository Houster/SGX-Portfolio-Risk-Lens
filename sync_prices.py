"""
Local script: fetch prices via yfinance and rsync CSVs to the server.
Run this from your local machine (not the server) to refresh price data.

Usage:
    python sync_prices.py [--server ubuntu@13.229.204.6]

The CSVs are written to ~/price_cache/ locally, then rsynced to
~/price_cache/ on the server. The server's analysis.py reads them from there.
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LOCAL_CACHE = Path.home() / "price_cache"

# All tickers the server needs: universe + aux + SORA candidates
TICKERS = [
    # SGX Universe (STI 30 + Next 20)
    "D05.SI", "O39.SI", "U11.SI", "Z74.SI", "S68.SI",
    "C31.SI", "A17U.SI", "ME8U.SI", "BN4.SI", "G13.SI",
    "C09.SI", "BS6.SI", "F34.SI", "H78.SI", "C6L.SI",
    "J36.SI", "N2IU.SI", "BUOU.SI", "K71U.SI", "T82U.SI",
    "U96.SI", "V03.SI", "Y92.SI", "9CI.SI", "S58.SI",
    "M44U.SI", "RW0U.SI", "AJBU.SI", "U14.SI", "C52.SI",
    "S63.SI", "RE4.SI", "5E2.SI", "AWX.SI", "OV8.SI",
    "1D0.SI", "BVA.SI", "TS0.SI", "CLN.SI", "5DM.SI",
    "558.SI", "5IF.SI", "F9D.SI", "43B.SI", "42S.SI",
    "CY6U.SI", "8K7.SI", "Q5T.SI", "5TP.SI", "5WA.SI",
    # Aux tickers (benchmark, FX, gold, regional ETFs)
    "ES3.SI", "SGDUSD=X", "GC=F", "AAXJ",
    # SORA proxy candidates
    "2YY=R", "SG10YT=RR", "A35.SI", "^IRX",
]


def _safe_name(ticker: str) -> str:
    return ticker.replace("=", "_").replace("^", "_").replace("/", "_")


def fetch_one(ticker: str, days: int = 1915) -> bool:
    """Download up to `days` of daily close and save as CSV. Returns True on success."""
    path = LOCAL_CACHE / f"{_safe_name(ticker)}.csv"
    try:
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if df.empty or "Close" not in df.columns:
            log.warning(f"{ticker}: no data returned")
            return False
        close = df["Close"].dropna()
        if len(close) < 20:
            log.warning(f"{ticker}: only {len(close)} rows — skipping")
            return False
        # Remove timezone so CSV reads cleanly on server
        close.index = close.index.tz_localize(None) if close.index.tz else close.index
        close.to_csv(path, header=["Close"])
        log.info(f"{ticker}: {len(close)} rows → {path.name}")
        return True
    except Exception as exc:
        log.warning(f"{ticker}: fetch failed — {exc}")
        return False


def rsync_to_server(server: str) -> None:
    key = Path.home() / ".ssh" / "orikai-key.pem"
    cmd = [
        "rsync", "-avz", "--delete",
        "-e", f"ssh -i {key}",
        str(LOCAL_CACHE) + "/",
        f"{server}:~/price_cache/",
    ]
    log.info(f"rsyncing to {server} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"rsync failed:\n{result.stderr}")
        sys.exit(1)
    log.info("rsync complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync price CSVs to SGX Risk Lens server")
    parser.add_argument(
        "--server", default="ubuntu@13.229.204.6",
        help="SSH destination for rsync (default: ubuntu@13.229.204.6)",
    )
    parser.add_argument(
        "--no-rsync", action="store_true",
        help="Fetch locally only, skip rsync",
    )
    parser.add_argument(
        "--days", type=int, default=1915,
        help="Days of history to fetch (default 1915 ≈ 5y + 3m buffer)",
    )
    args = parser.parse_args()

    LOCAL_CACHE.mkdir(parents=True, exist_ok=True)

    ok, fail = 0, 0
    for ticker in TICKERS:
        if fetch_one(ticker, args.days):
            ok += 1
        else:
            fail += 1

    log.info(f"Fetch complete: {ok} succeeded, {fail} failed.")

    if not args.no_rsync:
        rsync_to_server(args.server)


if __name__ == "__main__":
    main()
