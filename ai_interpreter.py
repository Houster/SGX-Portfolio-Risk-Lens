"""
Claude API integration for AI interpretation of quantitative module outputs.
Each module has a tailored prompt that extracts the most relevant metrics.
"""
import json
from typing import Any, Dict

import anthropic
import config

SYSTEM_PROMPT = (
    "You are a senior quantitative analyst at a Singapore-focused asset management firm. "
    "You write for portfolio managers who are investment professionals but not quants. "
    "Be direct, specific, and use the actual numbers from the data provided. "
    "Do not hedge excessively. "
    "Do not use the words 'recommend' or 'strategy' — frame everything as historical observation. "
    "Never use bullet points — write in tight paragraphs. "
    "Keep the response between 150 and 200 words."
)

MODULE_PROMPTS: Dict[str, str] = {
    "correlations": (
        "Here is the quantitative output for Rolling Correlations on a SGX equity portfolio: {data}. "
        "Write a 150-200 word interpretation highlighting the most important non-obvious insight. "
        "Reference the specific correlation values of the most and least correlated pairs. "
        "Discuss what the concentration risk pairs mean for portfolio construction. "
        "Close with one sentence on what a PM should watch going forward."
    ),
    "regimes": (
        "Here is the quantitative output for Regime Detection on a SGX equity portfolio: {data}. "
        "Write a 150-200 word interpretation. "
        "Contrast the portfolio's behaviour in Bull vs Bear vs Crisis regimes using the specific Sharpe, "
        "drawdown, and return figures. "
        "Highlight how the average pairwise correlation shifts across regimes — this is the key non-obvious insight. "
        "Close with one sentence on what a PM should watch going forward."
    ),
    "factors": (
        "Here is the quantitative output for Factor Exposure on a SGX equity portfolio: {data}. "
        "Write a 150-200 word interpretation. "
        "Lead with the market beta figure and what it implies about the portfolio's directional risk. "
        "Discuss the FX sensitivity — how much does a 1% move in USD/SGD affect the portfolio? "
        "Reference the R-squared to contextualise how well these factors explain returns. "
        "Close with one sentence on what a PM should watch going forward."
    ),
    "tail_risk": (
        "Here is the quantitative output for Tail Risk analysis on a SGX equity portfolio: {data}. "
        "Write a 150-200 word interpretation. "
        "Lead with the 99% 1-day VaR in dollar terms context (i.e. what it means for a S$10M portfolio). "
        "Compare the CVaR to the VaR — the gap reveals the severity of tail events beyond the threshold. "
        "Reference the maximum drawdown depth and recovery duration. "
        "Highlight the skewness and kurtosis — what do they imply about the distribution shape? "
        "Close with one sentence on what a PM should watch going forward."
    ),
    "hedging": (
        "Here is the quantitative output for Historical Hedging Offsets analysis on a SGX equity portfolio: {data}. "
        "Write a 150-200 word interpretation. "
        "Focus on which instruments showed the lowest (most negative) correlation during Crisis and Bear regimes specifically. "
        "Reference the average return during the portfolio's worst 20 days for the two most effective historical offsets. "
        "Note the vol reduction achievable at 10% and 20% allocations. "
        "Do not use the word 'recommend'. Frame strictly as historical observations. "
        "Close with one sentence on what a PM should watch going forward."
    ),
    "benchmark": (
        "Here is the quantitative output for Benchmark Comparison of a SGX equity portfolio against the STI ETF (ES3.SI): {data}. "
        "Write a 150-200 word interpretation. "
        "Lead with the annualised alpha and whether the portfolio has historically outperformed or underperformed the index. "
        "Reference the beta — does the portfolio amplify or dampen market moves? "
        "Compare the Sharpe ratios and max drawdown of the portfolio vs the STI ETF directly. "
        "Use the information ratio to assess whether active risk has been rewarded. "
        "Close with one sentence on what a PM should watch going forward."
    ),
}


def _slim_data(module: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strip large time-series arrays before sending to Claude — we only
    need the summary statistics for interpretation, not thousands of data points.
    """
    slim = json.loads(json.dumps(data, default=str))  # deep copy via JSON

    if module == "correlations":
        slim.pop("correlation_matrix", None)
        slim.pop("rolling_sample", None)

    elif module == "regimes":
        slim.pop("regime_periods", None)
        slim.pop("portfolio_cumulative_returns", None)

    elif module == "factors":
        slim.pop("rolling_exposures", None)

    elif module == "tail_risk":
        slim.pop("rolling_max_drawdown", None)
        slim.pop("return_histogram", None)
        # Keep worst_10_days but trim contributors detail
        if "worst_10_days" in slim:
            slim["worst_10_days"] = slim["worst_10_days"][:5]

    elif module == "hedging":
        slim.pop("worst_20_dates", None)

    elif module == "benchmark":
        slim.pop("dates", None)
        slim.pop("portfolio_cumulative", None)
        slim.pop("benchmark_cumulative", None)
        slim.pop("rolling_12m", None)

    return slim


_PRESCREENING_SYSTEM = (
    "You are a quantitative portfolio analyst at a Singapore-focused asset management firm. "
    "You have deep knowledge of SGX-listed companies, their business models, sector dynamics, "
    "and how they have historically behaved in different market environments. "
    "Your task is to pre-screen a candidate universe to identify the most promising names for "
    "further quantitative analysis — you are narrowing the field, not making final decisions. "
    "Be efficient and direct. Respond only in the JSON format specified. "
    "Do not include any preamble, explanation, or markdown formatting outside the JSON object."
)

_CONSTRUCTION_SYSTEM = (
    "You are a senior portfolio strategist at a Singapore-focused asset management firm writing "
    "for a portfolio manager audience. You have deep familiarity with SGX-listed companies, "
    "Singapore market structure, family-controlled conglomerates, REIT governance, and SEA "
    "macroeconomic context. Write in tight paragraphs, no bullet points. Be specific — reference "
    "actual company names and actual numbers. Do not use the words 'recommend', 'advice', or "
    "'strategy'. Frame all outputs as historical quantitative scenarios. Always acknowledge one "
    "key limitation of each scenario at the end."
)


def prescreening_pass(
    current_portfolio: Dict[str, Any],
    portfolio_risk_profile: Dict[str, Any],
    universe_metadata: Dict[str, Any],
    objective: str,
) -> Any:
    """
    Pass 1: Claude pre-screens the universe before any Python computation.
    Returns parsed dict with addition_candidates and removal_candidates, or None on failure.
    """
    import logging
    logger = logging.getLogger(__name__)

    universe_list = [
        {
            "ticker": ticker,
            "name": meta["name"],
            "sector": meta["sector"],
            "market_cap_tier": meta["market_cap_tier"],
            "liquidity_warning": meta.get("liquidity_warning", False),
        }
        for ticker, meta in universe_metadata.items()
        if ticker not in current_portfolio
    ]

    user_content = (
        f"Current portfolio risk profile:\n{json.dumps(portfolio_risk_profile, indent=2)}\n\n"
        f"Candidate universe (names not already in portfolio):\n{json.dumps(universe_list, indent=2)}\n\n"
        f"Optimisation objective: {objective}\n\n"
        "Based on the portfolio's current risk profile and the selected optimisation objective, identify:\n\n"
        "1. Top 12 addition candidates from the universe most likely to improve the portfolio — "
        "prioritise names that fill identified gaps: sector underweights, low stress-correlation "
        "potential, factor exposure offsets. For each provide a one-sentence rationale referencing "
        "the specific portfolio gap it addresses.\n\n"
        "2. Top 6 removal candidates from the current holdings most likely to be dragging on the "
        "objective — consider sector concentration, high stress correlations, factor exposure overlap. "
        "For each provide a one-sentence rationale.\n\n"
        'Respond ONLY with valid JSON in this exact structure:\n'
        '{\n'
        '  "addition_candidates": [\n'
        '    {"ticker": "...", "rationale": "..."},\n'
        '    ...\n'
        '  ],\n'
        '  "removal_candidates": [\n'
        '    {"ticker": "...", "rationale": "..."},\n'
        '    ...\n'
        '  ]\n'
        '}'
    )

    try:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=2048,
            system=_PRESCREENING_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
        )
        raw = message.content[0].text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)
        # Validate structure
        if "addition_candidates" not in result or "removal_candidates" not in result:
            raise ValueError("Missing required keys in Pass 1 response")
        return result

    except Exception as exc:
        logger.warning(f"Pass 1 JSON parse failed, falling back to full universe analysis: {exc}")
        return None


def interpret_construction_scenarios(
    scenarios_json: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
    objective: str,
    pass1_rationales: Dict[str, str],
) -> Dict[str, str]:
    """
    Pass 3: Claude synthesises the final construction scenario interpretation.
    Returns dict with keys 'A', 'B', 'C' (scenario interpretations) and 'strategic_signal'.
    On failure returns empty strings.
    """
    import logging
    logger = logging.getLogger(__name__)

    rationale_text = ""
    if pass1_rationales:
        rationale_text = "\nAddition rationales from pre-screening:\n"
        for t, r in pass1_rationales.items():
            if r:
                rationale_text += f"  {t}: {r}\n"

    user_content = (
        f"The current portfolio has the following baseline metrics:\n"
        f"{json.dumps(baseline_metrics, indent=2)}\n\n"
        f"Optimisation objective: {objective}\n\n"
        f"Three portfolio construction scenarios based on marginal contribution analysis:\n"
        f"{json.dumps(scenarios_json, indent=2)}\n\n"
        f"The following qualitative rationales informed the initial candidate selection in Pass 1 "
        f"— incorporate these where relevant:{rationale_text}\n\n"
        "For each scenario (A, B, C), write 200-250 words covering:\n"
        "1. Which assets are being added and removed, and the primary quantitative rationale for each change\n"
        "2. The most important non-quantitative consideration a PM should weigh — draw on your knowledge "
        "of the specific companies involved: governance, business model, SGX market dynamics, sector context, "
        "family-controlled structures where relevant\n"
        "3. The before/after improvement in the selected objective metric as a specific number\n"
        "4. One key limitation of this scenario that the historical data cannot capture\n\n"
        "Conclude with a 100-word synthesis across all three scenarios under the heading 'Strategic Signal': "
        "what is the consistent finding across all three, and what does it reveal about the fundamental "
        "risk characteristic of the current book?\n\n"
        "Respond ONLY with valid JSON in this exact structure:\n"
        '{\n'
        '  "A": "...",\n'
        '  "B": "...",\n'
        '  "C": "...",\n'
        '  "strategic_signal": "..."\n'
        '}'
    )

    try:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=4096,
            system=_CONSTRUCTION_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
        )
        raw = message.content[0].text.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)
        return {
            "A": result.get("A", ""),
            "B": result.get("B", ""),
            "C": result.get("C", ""),
            "strategic_signal": result.get("strategic_signal", ""),
        }

    except Exception as exc:
        logger.warning(f"Pass 3 interpretation failed: {exc}")
        return {"A": "", "B": "", "C": "", "strategic_signal": ""}


def interpret_module(module: str, data: Dict[str, Any]) -> str:
    """
    Call Claude to generate a plain-English interpretation of a module's output.
    Returns the interpretation string, or an error message if the API call fails.
    """
    if module not in MODULE_PROMPTS:
        return f"No prompt configured for module: {module}"

    slim = _slim_data(module, data)
    data_str = json.dumps(slim, indent=2, default=str)

    user_content = MODULE_PROMPTS[module].format(data=data_str)

    try:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=config.CLAUDE_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        return message.content[0].text.strip()
    except anthropic.AuthenticationError:
        return (
            "AI interpretation unavailable: invalid or missing ANTHROPIC_API_KEY. "
            "Set the environment variable and restart the server."
        )
    except anthropic.RateLimitError:
        return "AI interpretation temporarily unavailable: API rate limit reached. Please try again shortly."
    except Exception as exc:
        return f"AI interpretation unavailable: {exc}"
