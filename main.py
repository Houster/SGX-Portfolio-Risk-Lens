"""
SGX Portfolio Risk Lens — FastAPI backend.
Stateless: all data is held in an in-memory cache keyed by portfolio hash + period.
"""
import asyncio
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, field_validator
from sse_starlette.sse import EventSourceResponse

load_dotenv()

logging.basicConfig(level=logging.INFO)

import analysis as an
import ai_interpreter as ai
import universe as univ

app = FastAPI(title="SGX Portfolio Risk Lens", version="1.0.0")


@app.on_event("startup")
async def startup_prefetch():
    """Pre-fetch universe price data in a background thread at startup."""
    import threading
    t = threading.Thread(target=univ.prefetch_universe, kwargs={"timeout_seconds": 60}, daemon=True)
    t.start()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory price cache — key: sha256 of (sorted_tickers, period)
_price_cache: Dict[str, Dict[str, Any]] = {}

FRONTEND_PATH = Path(__file__).parent / "index.html"


# ─── Request / Response Models ────────────────────────────────────────────────

class Holding(BaseModel):
    ticker: str
    name: str
    weight: float   # percentage, e.g. 15.0 for 15%
    sector: str


class PortfolioRequest(BaseModel):
    holdings: List[Holding]
    period: str = "5y"
    benchmark: str = "ES3.SI"

    @field_validator("period")
    @classmethod
    def validate_period(cls, v: str) -> str:
        if v not in ("1y", "3y", "5y"):
            raise ValueError("period must be one of: 1y, 3y, 5y")
        return v

    def tickers(self) -> List[str]:
        return [h.ticker for h in self.holdings]

    def weights(self) -> Dict[str, float]:
        return {h.ticker: h.weight for h in self.holdings}

    def cache_key(self) -> str:
        payload = json.dumps(
            {"tickers": sorted(self.tickers()), "period": self.period},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class CorrelationsRequest(PortfolioRequest):
    window: int = 90

    @field_validator("window")
    @classmethod
    def validate_window(cls, v: int) -> int:
        if v not in (30, 90, 180):
            raise ValueError("window must be 30, 90, or 180")
        return v


class AIInterpretRequest(BaseModel):
    module: str
    data: Dict[str, Any]


class ConstructionRequest(BaseModel):
    portfolio: Dict[str, float]   # {ticker: weight_fraction}  weights need not sum to 1 — normalised internally
    objective: str = "sharpe"

    @field_validator("objective")
    @classmethod
    def validate_objective(cls, v: str) -> str:
        if v not in ("sharpe", "min_vol", "cvar"):
            raise ValueError("objective must be one of: sharpe, min_vol, cvar")
        return v


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_prices(req: PortfolioRequest) -> Dict[str, Any]:
    key = req.cache_key()
    if key not in _price_cache:
        result = an.fetch_price_data(req.tickers(), req.period)
        if "error" in result:
            raise HTTPException(status_code=502, detail=result["error"])
        _price_cache[key] = result
    return _price_cache[key]


def _prices_df(req: PortfolioRequest):
    cached = _get_prices(req)
    return cached["prices"]


def _safe_json(obj: Any) -> Any:
    """Convert numpy/pandas types to Python native types for JSON serialisation."""
    import pandas as pd

    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    return obj


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_frontend():
    if not FRONTEND_PATH.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(FRONTEND_PATH), media_type="text/html")


@app.post("/api/load")
async def load_portfolio(req: PortfolioRequest):
    """
    Fetch (and cache) price data for the requested portfolio.
    Returns coverage metadata and basic portfolio stats.
    """
    cached = _get_prices(req)
    prices = cached["prices"]

    port = an.compute_portfolio_returns(prices, req.weights())
    ann_return = float(port.mean() * 252) if len(port) > 0 else 0.0
    ann_vol = float(port.std() * np.sqrt(252)) if len(port) > 0 else 0.0
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    return JSONResponse(_safe_json({
        "status": "ok",
        "cache_key": req.cache_key(),
        "coverage_start": cached["coverage_start"],
        "coverage_end": cached["coverage_end"],
        "trading_days": cached["trading_days"],
        "failed_tickers": cached["failed_tickers"],
        "warnings": cached["warnings"],
        "available_tickers": cached["available_tickers"],
        "portfolio_stats": {
            "annualized_return": round(ann_return, 4),
            "annualized_vol": round(ann_vol, 4),
            "sharpe": round(sharpe, 3),
        },
    }))


@app.post("/api/analysis/correlations")
async def correlations(req: CorrelationsRequest):
    prices = _prices_df(req)
    result = an.compute_correlations(prices, req.weights(), window=req.window)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return JSONResponse(_safe_json(result))


@app.post("/api/analysis/regimes")
async def regimes(req: PortfolioRequest):
    prices = _prices_df(req)
    result = an.detect_regimes(prices, req.weights(), benchmark=req.benchmark)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return JSONResponse(_safe_json(result))


@app.post("/api/analysis/factors")
async def factors(req: PortfolioRequest):
    prices = _prices_df(req)
    result = an.compute_factor_exposure(prices, req.weights(), benchmark=req.benchmark)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return JSONResponse(_safe_json(result))


@app.post("/api/analysis/tail-risk")
async def tail_risk(req: PortfolioRequest):
    prices = _prices_df(req)
    result = an.compute_tail_risk(prices, req.weights())
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return JSONResponse(_safe_json(result))


@app.post("/api/analysis/benchmark")
async def benchmark(req: PortfolioRequest):
    prices = _prices_df(req)
    result = an.compute_benchmark_comparison(prices, req.weights(), benchmark=req.benchmark)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return JSONResponse(_safe_json(result))


@app.post("/api/analysis/hedging")
async def hedging(req: PortfolioRequest):
    prices = _prices_df(req)
    result = an.compute_hedging_offsets(prices, req.weights(), benchmark=req.benchmark)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return JSONResponse(_safe_json(result))


@app.post("/api/ai/interpret")
async def ai_interpret(req: AIInterpretRequest):
    """Call Claude to interpret a module's quantitative output."""
    text = ai.interpret_module(req.module, req.data)
    return JSONResponse({"interpretation": text})


@app.post("/api/construction-scenarios")
async def construction_scenarios(req: ConstructionRequest):
    """
    Two-pass portfolio construction scenario engine streamed via SSE.
    """
    import pandas as pd

    async def event_stream():
        def emit(msg: str):
            return {"data": json.dumps({"event": msg})}

        portfolio = req.portfolio
        objective = req.objective

        # Normalise weights
        total_w = sum(portfolio.values())
        if total_w <= 0:
            yield {"data": json.dumps({"error": "Portfolio weights must be positive."})}
            return
        norm_portfolio = {t: w / total_w for t, w in portfolio.items()}

        # ── Fetch portfolio price data ──
        yield emit("Preparing portfolio risk profile...")
        await asyncio.sleep(0)

        tickers = list(norm_portfolio.keys())
        price_result = an.fetch_price_data(tickers, period="3y")
        if "error" in price_result:
            yield {"data": json.dumps({"error": price_result["error"]})}
            return

        prices = price_result["prices"]

        # ── Build portfolio risk profile from available base modules ──
        holdings_info = []
        for t, w in norm_portfolio.items():
            meta = univ.UNIVERSE.get(t, {})
            holdings_info.append({
                "ticker": t,
                "name": meta.get("name", t),
                "sector": meta.get("sector", "Unknown"),
                "weight": round(w, 4),
            })

        sector_totals: Dict[str, float] = {}
        for h in holdings_info:
            sector_totals[h["sector"]] = sector_totals.get(h["sector"], 0) + h["weight"]

        tail_result = an.compute_tail_risk(prices, norm_portfolio)
        cvar_99 = tail_result.get("cvar_99_1d", None)
        max_dd = tail_result.get("max_drawdown", {}).get("depth", None)

        corr_result = an.compute_correlations(prices, norm_portfolio)
        top5_corr = corr_result.get("top_correlated", [])[:5]

        portfolio_risk_profile = {
            "holdings": holdings_info,
            "sector_concentration": {k: round(v, 4) for k, v in sector_totals.items()},
            "top_5_correlated_pairs": top5_corr,
            "cvar_99_1d": cvar_99,
            "max_drawdown": max_dd,
        }

        # ── Universe metadata (exclude current holdings) ──
        universe_metadata = univ.get_available_universe_metadata(exclude_tickers=tickers)
        universe_ret = univ.get_universe_returns()

        if universe_ret is None:
            universe_ret = pd.DataFrame()

        # ── Pass 1: Claude pre-screening ──
        yield emit("Pass 1: Claude pre-screening 50 candidates...")
        await asyncio.sleep(0)

        pass1_result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: ai.prescreening_pass(norm_portfolio, portfolio_risk_profile, universe_metadata, objective),
        )

        prescreening_fallback = pass1_result is None

        if prescreening_fallback:
            yield emit("Pass 1 pre-screening unavailable. Running full universe analysis...")
            n_add = len([t for t in universe_metadata if t not in norm_portfolio])
            n_rem = len(norm_portfolio)
        else:
            n_add = len(pass1_result.get("addition_candidates", []))
            n_rem = len(pass1_result.get("removal_candidates", []))
            yield emit(f"Pass 1 complete. Shortlisted {n_add} addition candidates, {n_rem} removal candidates.")

        await asyncio.sleep(0)

        # ── Pass 2: Python marginal contribution analysis ──
        yield emit("Pass 2: Running marginal contribution analysis on shortlisted names...")
        await asyncio.sleep(0)

        analysis_result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: an.run_construction_analysis(
                norm_portfolio,
                prices,
                universe_ret,
                universe_metadata,
                objective,
                pass1_result,
            ),
        )

        if "error" in analysis_result:
            yield {"data": json.dumps({"error": analysis_result["error"]})}
            return

        yield emit("Pass 2 complete. Constructing scenarios A, B, C...")
        await asyncio.sleep(0)

        baseline_metrics = analysis_result["baseline_metrics"]
        scenarios = analysis_result["scenarios"]

        # ── Pass 3: Claude synthesis ──
        yield emit("Pass 3: Generating AI interpretation...")
        await asyncio.sleep(0)

        pass1_rationales: Dict[str, str] = {}
        if pass1_result:
            for c in pass1_result.get("addition_candidates", []) + pass1_result.get("removal_candidates", []):
                pass1_rationales[c["ticker"]] = c.get("rationale", "")

        interpretation_unavailable = False
        try:
            interp = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ai.interpret_construction_scenarios(
                    {k: _safe_json(v) for k, v in scenarios.items()},
                    _safe_json(baseline_metrics),
                    objective,
                    pass1_rationales,
                ),
            )
            for key in ("A", "B", "C"):
                if key in scenarios and interp.get(key):
                    scenarios[key]["ai_interpretation"] = interp[key]
            strategic_signal = interp.get("strategic_signal", "")
        except Exception:
            interpretation_unavailable = True
            strategic_signal = ""

        yield emit("Complete.")
        await asyncio.sleep(0)

        # ── Build final response ──
        response_payload = _safe_json({
            "baseline_metrics":        baseline_metrics,
            "prescreening_fallback":   prescreening_fallback,
            "interpretation_unavailable": interpretation_unavailable,
            "scenarios":               scenarios,
            "strategic_signal":        strategic_signal,
        })

        yield {"data": json.dumps({"result": response_payload})}

    return EventSourceResponse(event_stream())


@app.get("/api/health")
async def health():
    return {"status": "ok", "cache_entries": len(_price_cache)}


# Allow running directly: python main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
