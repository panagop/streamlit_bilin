"""Streamlit front end: `uv run streamlit run app.py`."""

from __future__ import annotations

import sys
from pathlib import Path

# Hosts that install only the dependencies, not the project itself (e.g. Streamlit Cloud).
sys.path.insert(0, str(Path(__file__).parent / "src"))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from bilin import BilinearizationError, BilinOptions, Curve, bilinearize

EXAMPLES = Path(__file__).parent / "examples"

st.title("Bilin App")
st.write("Bilinear approximation of *force* - *deformation* curves.")

st.sidebar.header("Input")
uploaded = st.sidebar.file_uploader("Delimited text file (x y), space, tab, ';' or ','")
example_names = sorted(p.name for p in EXAMPLES.glob("Example*"))
example = st.sidebar.selectbox(
    "…or an example", example_names, index=example_names.index("Example2"), disabled=uploaded is not None
)
chart_title = st.sidebar.text_input("Chart title", value="Διγραμμικοποίηση καμπύλης αντίστασης")
x_label = st.sidebar.text_input("x label", value="δ (m)")
y_label = st.sidebar.text_input("y label", value="V (kN)")

st.sidebar.header("Options")
options = BilinOptions(
    xtarget=st.sidebar.number_input("xtarget (0 = whole curve)", value=0.35, min_value=0.0, format="%.4f"),
    dropstrength=st.sidebar.number_input("dropstrength", value=0.75, min_value=0.0, max_value=1.0),
    elastoplastic=st.sidebar.checkbox("Elastic-perfectly plastic", value=False),
    allowa010=st.sidebar.checkbox("Allow α > 0.10", value=True),
)

try:
    curve = (
        Curve.from_text(uploaded.getvalue().decode("utf-8"))
        if uploaded is not None
        else Curve.from_file(EXAMPLES / example)
    )
except (ValueError, UnicodeDecodeError) as exc:
    st.error(f"Could not read the curve: {exc}")
    st.stop()

try:
    result = bilinearize(curve, options)
except BilinearizationError as exc:
    st.error(str(exc))
    st.stop()

bl = result.curve
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(curve.x, curve.y, label="Initial curve", lw=2)
ax.plot(bl.d_array, bl.a_array, label="Bilinear curve", lw=2)
ax.set_title(chart_title)
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
ax.grid(True, alpha=0.4)
ax.legend()
st.pyplot(fig)

st.subheader("Bilinear curve")
st.dataframe(
    pd.DataFrame({"quantity": list(bl.quantities()), "value": list(bl.quantities().values())}),
    hide_index=True,
)

with st.expander(f"Iterations ({len(result.iterations)})"):
    st.dataframe(pd.DataFrame([vars(it) for it in result.iterations]), hide_index=True)
