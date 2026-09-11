from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
import json

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG


def save_decision_report(ticker, date, decision, results_dir):
    report_dir = Path(results_dir) / ticker
    report_dir.mkdir(parents=True, exist_ok=True)

    path = report_dir / f"{ticker.lower()}_decision_report.md"

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {ticker} Decision Report\n\n")
        f.write(f"**Analysis Date:** {date}\n\n")
        f.write("---\n\n")
        f.write(str(decision))

    return path
def get_nested(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
        if cur is None:
            return None
    return cur


def write_subsection(f, title, content):
    if not content:
        return
    f.write(f"## {title}\n\n")
    f.write(str(content).strip())
    f.write("\n\n")


def save_complete_report(ticker, date, state, decision, results_dir):
    from pathlib import Path
    from datetime import datetime

    report_dir = Path(results_dir) / ticker
    report_dir.mkdir(parents=True, exist_ok=True)

    path = report_dir / f"{ticker.lower()}_complete_report.md"

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Trading Analysis Report: {ticker}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # I. Analyst Team Reports
        f.write("# I. Analyst Team Reports\n\n")
        write_subsection(f, "Market Analyst", state.get("market_report"))
        write_subsection(f, "Social Analyst", state.get("sentiment_report"))
        write_subsection(f, "News Analyst", state.get("news_report"))
        write_subsection(f, "Fundamentals Analyst", state.get("fundamentals_report"))

        # II. Research Team Decision
        f.write("# II. Research Team Decision\n\n")
        write_subsection(
            f,
            "Bull Researcher",
            get_nested(state, "investment_debate_state", "bull_history"),
        )
        write_subsection(
            f,
            "Bear Researcher",
            get_nested(state, "investment_debate_state", "bear_history"),
        )
        write_subsection(
            f,
            "Research Manager",
            get_nested(state, "investment_debate_state", "judge_decision"),
        )

        # III. Trading Team Plan
        f.write("# III. Trading Team Plan\n\n")
        write_subsection(
            f,
            "Trader",
            state.get("trader_investment_decision"),
        )

        # IV. Risk Management Team Decision
        f.write("# IV. Risk Management Team Decision\n\n")
        write_subsection(
            f,
            "Aggressive Analyst",
            get_nested(state, "risk_debate_state", "aggressive_history"),
        )
        write_subsection(
            f,
            "Conservative Analyst",
            get_nested(state, "risk_debate_state", "conservative_history"),
        )
        write_subsection(
            f,
            "Neutral Analyst",
            get_nested(state, "risk_debate_state", "neutral_history"),
        )

        # V. Portfolio Manager Decision
        f.write("# V. Portfolio Manager Decision\n\n")
        write_subsection(
            f,
            "Portfolio Manager",
            get_nested(state, "risk_debate_state", "judge_decision") or decision,
        )

    return path
config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = "gpt-5.4-mini"
config["quick_think_llm"] = "gpt-5.4-mini"
config["max_debate_rounds"] = 1

config["data_vendors"] = {
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "yfinance",
    "news_data": "yfinance",
}

ticker = "CGGR"
date = "2026-05-08"

ta = TradingAgentsGraph(
    debug=True, config=config)

state, decision = ta.propagate(ticker, date)

print(decision)

decision_path = save_decision_report(
    ticker=ticker,
    date=date,
    decision=decision,
    results_dir=config["results_dir"],
)

complete_path = save_complete_report(
    ticker=ticker,
    date=date,
    state=state,
    decision=decision,
    results_dir=config["results_dir"],
)

print(f"\nSaved decision report to: {decision_path}")
print(f"Saved complete report to: {complete_path}")