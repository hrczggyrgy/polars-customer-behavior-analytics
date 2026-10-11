"""Launch with streamlit run loyalty_app.py; app.py stays unchanged."""

import app
import streamlit as st
from patronage_dashboard import render

if not hasattr(app, "_patronage_original_display"):
    app._patronage_original_display = app.display_results
original_display = app._patronage_original_display


def display_results(root, coverage):
    workspace = st.radio(
        "Dashboard workspace",
        ["Visual analytics", "Observed patronage"],
        horizontal=True,
        key="dashboard_workspace",
    )
    if workspace == "Observed patronage":
        render(root, coverage)
    else:
        original_display(root, coverage)


if __name__ == "__main__":
    app.display_results = display_results
    app.main()
