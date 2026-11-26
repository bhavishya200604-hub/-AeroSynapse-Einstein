"""AeroSynapse–Einstein research workbench (Streamlit). Run: ``streamlit run src/aerosynapse/ui/app.py``."""

from __future__ import annotations

import streamlit as st

from aerosynapse.safety import DISCLAIMER
from aerosynapse.ui import workspaces as W

st.set_page_config(page_title="AeroSynapse–Einstein workbench", layout="wide")
st.warning(DISCLAIMER)
PAGES = {
    "Physics": W.physics,
    "Simulation": W.simulation,
    "ML": W.ml,
    "Optimization": W.optimization,
    "Research": W.research,
}
choice = st.sidebar.radio("Workspace", list(PAGES), key="workspace")
PAGES[choice]()
