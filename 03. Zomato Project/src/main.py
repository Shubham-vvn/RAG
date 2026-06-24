"""
Entry point for the Zomato AI Restaurant Recommendation System.

Usage:
    CLI mode:       python3 -m src.main --mode cli
    Web mode:       .venv/bin/streamlit run src/ui/streamlit_app.py
    API mode:       uvicorn src.api.routes:app --reload
"""

import argparse
import logging
import sys

from src.data import initialize_data
from src.ui.cli import run_cli

logger = logging.getLogger(__name__)


def main():
    """Main entry point — initialize data and launch the selected interface."""
    parser = argparse.ArgumentParser(
        description="🍽️ Zomato AI Restaurant Recommender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 -m src.main --mode cli       Launch interactive CLI
  python3 -m src.main --mode web       Launch Streamlit web UI
  python3 -m src.main --mode api       Launch FastAPI server
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "web", "api"],
        default="cli",
        help="Interface mode: cli (terminal), web (Streamlit), or api (FastAPI)",
    )

    args = parser.parse_args()

    logger.info("🍽️  Zomato AI Restaurant Recommender starting...")
    logger.info(f"Mode: {args.mode}")

    if args.mode == "cli":
        logger.info("Initializing dataset repository...")
        try:
            repository = initialize_data()
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            sys.exit(1)

        logger.info("Launching CLI interface...")
        run_cli(repository)

    elif args.mode == "web":
        logger.info("Launching Streamlit Web App...")
        import subprocess
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", "src/ui/streamlit_app.py"])
        except KeyboardInterrupt:
            logger.info("Streamlit Web App stopped.")

    elif args.mode == "api":
        logger.info("Launching FastAPI server...")
        import uvicorn
        try:
            uvicorn.run("src.api.routes:app", host="127.0.0.1", port=8000, reload=True)
        except KeyboardInterrupt:
            logger.info("FastAPI server stopped.")


if __name__ == "__main__":
    main()

