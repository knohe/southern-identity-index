
import io
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import streamlit as st

st.set_page_config(page_title="Southern Identity Index — Quadrant & Blobs", layout="wide")

st.title("Southern Identity Index — Quadrant & Blobs")

st.markdown(
    "Use this app to collect **Southern Identity Index (SII)** category scores for places, "
    "position them on a quadrant (Economy vs. Hierarchy), and generate both a **scatter map** "
    "and **sharper-edged blobs** figure."
)

# -----------------------
# Sidebar — Weights
# -----------------------
st.sidebar.header("Category Weights (0–1)")
default_weights = {
    "Economic System": 1.0,
    "Class & Inequality": 1.0,
    "Racial Disparities": 1.0,
    "Political Culture": 1.0,
    "Cultural Identity": 1.0,
    "Historical Continuity": 1.0,
}

weights = {}
for k, v in default_weights.items():
    weights[k] = st.sidebar.slider(k, 0.0, 2.0, float(v), 0.1)

st.sidebar.markdown("---")
kde_bw = st.sidebar.slider("KDE bandwidth (blob spread)", 0.10, 1.00, 0.35, 0.05)
sharpen_power = st.sidebar.slider("Edge sharpness (power transform)", 1.0, 6.0, 4.0, 0.5)
threshold = st.sidebar.slider("Primary contour threshold", 0.05, 0.95, 0.50, 0.05)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: upload a CSV or enter places manually below.")

# -----------------------
# Data Entry
# -----------------------
st.header("1) Enter or Upload Places & Scores")

uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])

schema_note = st.expander("Expected CSV Schema (click to expand)")
with schema_note:
    st.code(
        "name,econ_x,hier_y,economic,class_inequality,racial_disparities,political_culture,cultural_identity,historical_continuity\n"
        "Mississippi Delta,-4,5,5,5,5,4,5,5\n"
        "Atlanta,2,3,2,3,4,2,3,4\n"
        "Austin,4,-2,1,2,2,1,2,2\n"
        "Northern Virginia,5,-3,1,1,1,1,2,2\n"
        "Appalachia (progressive pockets),-3,-2,2,2,3,2,2,3\n"
        "Charlotte,3,2,2,3,3,3,3,3\n"
    )

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    st.subheader("Manual Entry Form")
    with st.form("place_form", clear_on_submit=True):
        name = st.text_input("Place name", "Mississippi Delta")
        econ_x = st.slider("Economic Structure (X-axis): -5=agrarian/extractive, +5=diversified/modern", -5.0, 5.0, -4.0, 0.1)
        hier_y = st.slider("Social Hierarchy (Y-axis): -5=egalitarian, +5=hierarchical", -5.0, 5.0, 5.0, 0.1)

        st.markdown("### SII Categories (0–5 each)")
        economic = st.slider("Economic System", 0, 5, 5, 1)
        class_inq = st.slider("Class & Inequality", 0, 5, 5, 1)
        racial = st.slider("Racial Disparities", 0, 5, 5, 1)
        political = st.slider("Political Culture", 0, 5, 4, 1)
        cultural = st.slider("Cultural Identity", 0, 5, 5, 1)
        historical = st.slider("Historical Continuity", 0, 5, 5, 1)

        submitted = st.form_submit_button("Add place")
        if "rows" not in st.session_state:
            st.session_state["rows"] = []

        if submitted:
            st.session_state["rows"].append({
                "name": name,
                "econ_x": econ_x,
                "hier_y": hier_y,
                "economic": economic,
                "class_inequality": class_inq,
                "racial_disparities": racial,
                "political_culture": political,
                "cultural_identity": cultural,
                "historical_continuity": historical,
            })

    if "rows" in st.session_state and st.session_state["rows"]:
        df = pd.DataFrame(st.session_state["rows"])
    else:
        # Start with seed examples if nothing uploaded/added yet
        df = pd.DataFrame([
            ["Mississippi Delta",-4,5,5,5,5,4,5,5],
            ["Atlanta",2,3,2,3,4,2,3,4],
            ["Austin",4,-2,1,2,2,1,2,2],
            ["Northern Virginia",5,-3,1,1,1,1,2,2],
            ["Appalachia (progressive pockets)",-3,-2,2,2,3,2,2,3],
            ["Charlotte",3,2,2,3,3,3,3,3],
        ], columns=[
            "name","econ_x","hier_y","economic","class_inequality","racial_disparities",
            "political_culture","cultural_identity","historical_continuity"
        ])

st.dataframe(df, use_container_width=True)

# -----------------------
# Compute SII
# -----------------------
def compute_sii(row):
    s = (
        row["economic"] * weights["Economic System"] +
        row["class_inequality"] * weights["Class & Inequality"] +
        row["racial_disparities"] * weights["Racial Disparities"] +
        row["political_culture"] * weights["Political Culture"] +
        row["cultural_identity"] * weights["Cultural Identity"] +
        row["historical_continuity"] * weights["Historical Continuity"]
    )
    return float(s)

df["SII"] = df.apply(compute_sii, axis=1)

st.header("2) Quadrant Scatter (size = SII)")
fig1, ax1 = plt.subplots(figsize=(7,7))

ax1.axhline(0, linewidth=1)
ax1.axvline(0, linewidth=1)

sizes = (df["SII"] * 10.0).clip(lower=20.0, upper=1000.0)
ax1.scatter(df["econ_x"], df["hier_y"], s=sizes)

for _, r in df.iterrows():
    ax1.text(r["econ_x"] + 0.15, r["hier_y"] + 0.15, f'{r["name"]}\n(SII {r["SII"]:.1f})', fontsize=8)

ax1.set_title("Quadrant Map of Southernness (SII as Size)")
ax1.set_xlabel("Economic Structure (← Agrarian / Extractive | Diversified / Modern →)")
ax1.set_ylabel("Social Hierarchy (↓ Egalitarian | Hierarchical ↑)")
ax1.set_xlim(-6, 6)
ax1.set_ylim(-6, 6)
ax1.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig1)

# -----------------------
# Blobs (KDE + Sharpened)
# -----------------------
st.header("3) Sharper-Edged Blobs (KDE + Threshold Contours)")

# Prepare data
X = df["econ_x"].to_numpy()
Y = df["hier_y"].to_numpy()
W = df["SII"].to_numpy()

# Build grid
x_grid = np.linspace(-6, 6, 400)
y_grid = np.linspace(-6, 6, 400)
XX, YY = np.meshgrid(x_grid, y_grid)
grid_pts = np.vstack([XX.ravel(), YY.ravel()])

if len(df) >= 2:
    kde = gaussian_kde(np.vstack([X, Y]), weights=W, bw_method=float(kde_bw))
    Z = kde(grid_pts).reshape(XX.shape)
    Z = (Z - Z.min()) / (Z.max() - Z.min() + 1e-9)
    Z = Z ** float(sharpen_power)

    # Plot
    fig2, ax2 = plt.subplots(figsize=(7,7))
    ax2.axhline(0, linewidth=1)
    ax2.axvline(0, linewidth=1)

    # Discrete filled contours for crisp edges
    levels = [0.15, 0.30, 0.50, 0.70, 0.85]
    cf = ax2.contourf(XX, YY, Z, levels=levels, alpha=0.85)
    ax2.contour(XX, YY, Z, levels=[float(threshold)], linewidths=2)

    # Labels
    for _, r in df.iterrows():
        ax2.text(r["econ_x"] + 0.15, r["hier_y"] + 0.15, r["name"], fontsize=8)

    ax2.set_title("Southernness Influence Blobs (Sharper Edges)")
    ax2.set_xlabel("Economic Structure (← Agrarian / Extractive | Diversified / Modern →)")
    ax2.set_ylabel("Social Hierarchy (↓ Egalitarian | Hierarchical ↑)")
    ax2.set_xlim(-6, 6)
    ax2.set_ylim(-6, 6)
    ax2.grid(True, linestyle="--", alpha=0.4)

    st.pyplot(fig2)

    # Download PNGs
    buf1, buf2 = io.BytesIO(), io.BytesIO()
    fig1.savefig(buf1, format="png", dpi=200, bbox_inches="tight")
    fig2.savefig(buf2, format="png", dpi=200, bbox_inches="tight")
    st.download_button("Download Scatter PNG", data=buf1.getvalue(), file_name="southern_quadrant_scatter.png", mime="image/png")
    st.download_button("Download Blobs PNG", data=buf2.getvalue(), file_name="southern_blobs.png", mime="image/png")
else:
    st.info("Add at least two places to compute blobs.")
