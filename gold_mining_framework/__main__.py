"""CLI entry point for the Market Idea Generator.

Usage:
    python -m gold_mining_framework "Health"
    python -m gold_mining_framework alternative medicine
"""

import argparse
import logging
import warnings

warnings.filterwarnings("ignore", message="LangSmith now uses UUID v7")

# Package import applies the allowed_objects filter after langchain_core loads.
from gold_mining_framework.graph import graph  # noqa: E402


def main() -> None:
    """Invoke the pipeline with a market query from the command line."""
    parser = argparse.ArgumentParser(
        description="Run the Market Idea Generator on a market or focus area."
    )
    parser.add_argument(
        "query",
        nargs="+",
        help="Market segment or focus area, e.g. Health or 'alternative medicine'",
    )
    args = parser.parse_args()
    query = " ".join(args.query)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logging.info("Running Market Idea Generator for: %s", query)
    logging.info("This can take several minutes (web search + Google Trends checks).")
    result = graph.invoke({"query": query})
    print(result["market_hierarchy"])  # noqa: T201


if __name__ == "__main__":
    main()
