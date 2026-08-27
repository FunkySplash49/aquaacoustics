"""
============================================================================
 app.py - AquaAcoustics: map-driven acoustic leak detection
============================================================================
Run with:
    streamlit run app.py

Two integrated screens:
    Survey Map              - pick a monitoring site, trigger real GCC-PHAT
                              detection, see the honest result on a map.
    Leak Detection Detail   - the full "how it figured that out" story for
                              whichever site was last triggered.

A role picker ("Admin" / "Field Staff") is a plain, passwordless selector -
cosmetic for this prototype, not a security boundary. Admin can trigger
detection; Field Staff can only view results.

All leak-detection physics live untouched in src/aeld/. All map + role
integration logic lives in src/aquaacoustics/. This file is routing only.
============================================================================
"""

import sys
from pathlib import Path

import streamlit as st

SRC_DIR = Path(__file__).parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aquaacoustics.map_view import render_map_page              # noqa: E402
from aquaacoustics.detection_view import render_detection_page  # noqa: E402


st.set_page_config(
    page_title="AquaAcoustics",
    page_icon="~",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("AquaAcoustics")
st.sidebar.caption("Leak detection (AELD) across monitoring sites")

role = st.sidebar.radio("View as", ["Admin", "Field Staff"])
st.session_state["role"] = role

st.sidebar.divider()

page = st.sidebar.radio(
    "Go to",
    ["Survey Map", "Leak Detection Detail"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.markdown(
    """
**The idea in three lines**

A leak hisses continuously.

A sensor at each end of the pipe hears that hiss - but the nearer one hears
it a fraction of a second sooner.

That fraction tells you where the leak is.
    """
)

if page == "Survey Map":
    render_map_page(role)
else:
    render_detection_page(role)

st.sidebar.divider()
st.sidebar.caption(
    "Prototype. The physics simulation and the leak-locating maths are real "
    "and unit-tested; sites and their pipe parameters are illustrative "
    "presets, and the role picker is not a security boundary."
)
