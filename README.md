# SGX Portfolio Risk Lens

An AI-powered quantitative risk analysis dashboard for SGX-listed equity portfolios. Built for Singapore-based asset managers and family offices.

---

## Features

- **Rolling Correlations** — pairwise correlation heatmap, concentration risk flagging
- **Regime Detection** — Bull / Bear / Crisis classification via STI ETF, per-regime statistics
- **Factor Exposure** — OLS regression against market beta, FX, volatility regime; rolling 252-day window
- **Tail Risk** — VaR, CVaR, maximum drawdown, worst-day attribution
- **Historical Hedging Offsets** — stress-period correlations for Gold, inverse STI, AAXJ, cash
- **AI Interpretation** — Claude-powered plain-English analysis for each module

---

## Requirements

- Python 3.11+
- `ANTHROPIC_API_KEY` environment variable

---

## Setup

```bash
# 1. Clone / enter project directory
cd "SGX Portfolio Risk Lens"

# 2. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
export ANTHROPIC_API_KEY=sk-ant-...   # or add to a .env file

# 5. Run
uvicorn main:app --reload
```

Then open **http://localhost:8000** in your browser.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key for AI interpretation |

Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

---

## Data Sources

- **Price data**: Yahoo Finance via [yfinance](https://github.com/ranaroussi/yfinance). Tickers follow SGX format (e.g. `D05.SI`).
- **Benchmark**: STI ETF (`ES3.SI`)
- **FX**: USD/SGD via `SGDUSD=X` (Yahoo Finance)
- **Gold**: COMEX Gold futures `GC=F` (converted to SGD terms)
- **Hedge proxies**: `AAXJ` (iShares Asia ex-Japan ETF), `AXJR` (if available)

**Disclosure**: Corporate action adjustments provided by Yahoo Finance may be incomplete. This tool is not suitable for institutional production use without a licensed data provider. All outputs are historical observations, not investment advice.

---

## Architecture

```
sgx-risk-lens/
├── main.py           # FastAPI app, routes, in-memory cache
├── analysis.py       # Quantitative computation (pure functions)
├── ai_interpreter.py # Claude API calls, prompt construction
├── index.html        # Single-file frontend (Vanilla JS + Plotly.js)
├── requirements.txt
└── README.md
```

- **Stateless**: no database. Price data is cached in memory per session.
- **Backend**: Python 3.11 / FastAPI / uvicorn
- **Frontend**: Single HTML file, Vanilla JS, Plotly.js v2.35
- **AI**: Anthropic Claude (`claude-sonnet-4-5`) for module interpretations

---

## Default Portfolio

| Ticker | Name | Weight | Sector |
|---|---|---|---|
| D05.SI | DBS Group | 15% | Banking |
| O39.SI | OCBC | 10% | Banking |
| U11.SI | UOB | 10% | Banking |
| C31.SI | CapitaLand Integrated Commercial Trust | 10% | REIT |
| A17U.SI | Ascendas REIT | 8% | REIT |
| ME8U.SI | Mapletree Industrial Trust | 7% | REIT |
| BN4.SI | Keppel Corporation | 8% | Industrials |
| S68.SI | Singapore Exchange | 7% | Financials |
| Z74.SI | Singtel | 8% | Telco |
| G13.SI | Genting Singapore | 7% | Consumer |
| C09.SI | City Developments | 5% | Property |
| BS6.SI | YZJ Shipbuilding | 5% | Industrials |

---

## Disclaimer

This tool uses publicly available price data for analytical purposes only. All outputs are historical observations, not investment advice. Price data sourced from Yahoo Finance via yfinance. Corporate action adjustments may be incomplete. Not suitable for institutional production use without a licensed data provider.
