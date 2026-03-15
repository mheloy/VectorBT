"""Streamlit dashboard entry point."""

import sys
from pathlib import Path

# Add project root to sys.path so page files can do `from src.* import ...`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
PAGES_DIR = PROJECT_ROOT / "src" / "dashboard" / "pages"

import streamlit as st

st.set_page_config(
    page_title="XAUUSD Backtest Engine",
    page_icon="📈",
    layout="wide",
)

# Reduce default padding so content fills the browser width
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100%;
    }
</style>
""", unsafe_allow_html=True)


def main():
    pages = {
        "Backtest": [
            st.Page(str(PAGES_DIR / "backtest.py"), title="Run Backtest", icon="🧪"),
            st.Page(str(PAGES_DIR / "optimize.py"), title="Optimize", icon="📊"),
        ],
        "Advanced Analysis": [
            st.Page(str(PAGES_DIR / "monte_carlo.py"), title="Monte Carlo", icon="🎲"),
            st.Page(str(PAGES_DIR / "walk_forward.py"), title="Walk-Forward", icon="🔁"),
            st.Page(str(PAGES_DIR / "trade_deep.py"), title="Trade Analysis", icon="🔍"),
            st.Page(str(PAGES_DIR / "regime.py"), title="Regime Analysis", icon="🌐"),
        ],
        "Results": [
            st.Page(str(PAGES_DIR / "results.py"), title="Saved Results", icon="💾"),
            st.Page(str(PAGES_DIR / "compare.py"), title="Compare Runs", icon="⚖️"),
        ],
    }
    pg = st.navigation(pages)
    pg.run()


if __name__ == "__main__":
    main()
