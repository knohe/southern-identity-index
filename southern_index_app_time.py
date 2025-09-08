
import io
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import streamlit as st

st.set_page_config(page_title="Southern Identity Index — Time Slider", layout="wide")
st.title("Southern Identity Index — Time Slider")

st.markdown(
    "Upload a time-series CSV or enter rows manually, then use the **time slider** to see how places move "
    "across the Southernness quadrant and how their influence blobs change. "
    "Optionally **interpolate** between years and draw **trails** of movement."
)

# -----------------------
# Sidebar — Weights & Viz Controls
# -----------------------
st.sidebar.header("Category Weights (0–2)")
default_weights = {
    "Economic System": 1.0,
    "Class & Inequality": 1.0,
    "Racial Disparities": 1.0,
    "Political Culture": 1.0,
    "Cultural Identity": 1.0,
    "Historical Continuity": 1.0,
}
weights = {k: st.sidebar.slider(k, 0.0, 2.0, float(v), 0.1) for k, v in default_weights.items()}

st.sidebar.markdown("---")
kde_bw = st.sidebar.slider("KDE bandwidth (blob spread)", 0.10, 1.00, 0.35, 0.05)
sharpen_power = st.sidebar.slider("Edge sharpness (power)", 1.0, 6.0, 4.0, 0.5)
threshold = st.sidebar.slider("Primary contour threshold", 0.05, 0.95, 0.50, 0.05)
show_trails = st.sidebar.checkbox("Show trails (paths across years)", value=True)
interpolate = st.sidebar.checkbox("Interpolate between years", value=True)

# -----------------------
# Data Entry / Upload
# -----------------------
st.header("1) Upload or Enter Time-Series Data")

template = (
    "name,year,econ_x,hier_y,economic,class_inequality,racial_disparities,political_culture,cultural_identity,historical_continuity\\n"
    "Mississippi Delta,1920,-4.5,5.5,5,5,5,5,5,5\\n"
    "Mississippi Delta,1970,-4.2,5.2,5,5,5,4,5,5\\n"
    "Mississippi Delta,2020,-4.0,5.0,5,5,5,4,5,5\\n"
    "Atlanta,1920,0.0,3.0,3,4,5,3,3,4\\n"
    "Atlanta,1970,1.5,3.0,2,3,4,2,3,4\\n"
    "Atlanta,2020,2.0,3.0,2,3,4,2,3,4\\n"
    "Austin,1920,1.0,-1.0,2,3,3,2,2,3\\n"
    "Austin,1970,2.5,-1.5,2,2,2,2,2,2\\n"
    "Austin,2020,4.0,-2.0,1,2,2,1,2,2\\n"
)

with st.expander("Expected CSV Schema & Example (click to expand)"):
    st.code(template)

uploaded = st.file_uploader("Upload CSV with a 'year' column (recommended)", type=["csv"])

def compute_sii(row):
    return float(
        row["economic"] * weights["Economic System"] +
        row["class_inequality"] * weights["Class & Inequality"] +
        row["racial_disparities"] * weights["Racial Disparities"] +
        row["political_culture"] * weights["Political Culture"] +
        row["cultural_identity"] * weights["Cultural Identity"] +
        row["historical_continuity"] * weights["Historical Continuity"]
    )

num_cols = [
    "econ_x","hier_y","economic","class_inequality","racial_disparities",
    "political_culture","cultural_identity","historical_continuity"
]

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    # Seed example dataframe
    from io import StringIO
    df = pd.read_csv(StringIO(template))

# Ensure schema
missing = set(["name","year"] + num_cols) - set(df.columns)
if missing:
    st.error(f"Missing required columns: {sorted(list(missing))}")
    st.stop()

df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["year"] + num_cols).copy()
df["SII"] = df.apply(compute_sii, axis=1)

st.dataframe(df.sort_values(["name","year"]), use_container_width=True)

# -----------------------
# Time Slider
# -----------------------
years_sorted = np.sort(df["year"].unique())
min_y, max_y = int(years_sorted.min()), int(years_sorted.max())
st.header("2) Choose a Year")
year = st.slider("Year", min_value=min_y, max_value=max_y, value=max_y, step=1)

# Snapshot builder with optional interpolation
def snapshot_at_year(df, year, interpolate):
    out_rows = []
    for name, grp in df.groupby("name"):
        grp = grp.sort_values("year")
        if not interpolate:
            # nearest year
            idx = (grp["year"] - year).abs().idxmin()
            out_rows.append(grp.loc[idx])
        else:
            # interpolate between bounding years
            earlier = grp[grp["year"] <= year].tail(1)
            later = grp[grp["year"] >= year].head(1)
            if earlier.empty and later.empty:
                continue
            if earlier.empty:
                out_rows.append(later.iloc[0])
            elif later.empty:
                out_rows.append(earlier.iloc[0])
            else:
                r0, r1 = earlier.iloc[0], later.iloc[0]
                if r0["year"] == r1["year"]:
                    out_rows.append(r0)
                else:
                    t = (year - r0["year"]) / (r1["year"] - r0["year"])
                    interp = {}
                    for c in num_cols + ["SII"]:
                        interp[c] = (1 - t) * r0[c] + t * r1[c]
                    row = pd.Series({"name": name, "year": year, **interp})
                    out_rows.append(row)
    return pd.DataFrame(out_rows)

snap = snapshot_at_year(df, year, interpolate=interpolate)

# -----------------------
# Scatter with optional trails
# -----------------------
st.header("3) Quadrant Scatter (Time Snapshot)")
fig1, ax1 = plt.subplots(figsize=(7,7))
ax1.axhline(0, linewidth=1)
ax1.axvline(0, linewidth=1)

if show_trails:
    for name, grp in df.groupby("name"):
        g = grp.sort_values("year")
        ax1.plot(g["econ_x"], g["hier_y"], linewidth=1)

sizes = (snap["SII"] * 10.0).clip(lower=20.0, upper=1000.0) if len(snap) else []
if len(sizes):
    ax1.scatter(snap["econ_x"], snap["hier_y"], s=sizes)
    for _, r in snap.iterrows():
        ax1.text(r["econ_x"] + 0.12, r["hier_y"] + 0.12, f'{r["name"]}\\n{int(r["year"])} (SII {r["SII"]:.1f})', fontsize=8)

ax1.set_title("Quadrant Map — Snapshot")
ax1.set_xlabel("Economic Structure (← Agrarian / Extractive | Diversified / Modern →)")
ax1.set_ylabel("Social Hierarchy (↓ Egalitarian | Hierarchical ↑)")
ax1.set_xlim(-6, 6)
ax1.set_ylim(-6, 6)
ax1.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig1)

# -----------------------
# Sharper-Edged Blobs at snapshot year
# -----------------------
st.header("4) Influence Blobs — Snapshot Year")
if len(snap) >= 2:
    X = snap["econ_x"].to_numpy()
    Y = snap["hier_y"].to_numpy()
    W = snap["SII"].to_numpy()

    x_grid = np.linspace(-6, 6, 400)
    y_grid = np.linspace(-6, 6, 400)
    XX, YY = np.meshgrid(x_grid, y_grid)
    grid_pts = np.vstack([XX.ravel(), YY.ravel()])

    kde = gaussian_kde(np.vstack([X, Y]), weights=W, bw_method=float(kde_bw))
    Z = kde(grid_pts).reshape(XX.shape)
    Z = (Z - Z.min()) / (Z.max() - Z.min() + 1e-9)
    Z = Z ** float(sharpen_power)

    fig2, ax2 = plt.subplots(figsize=(7,7))
    ax2.axhline(0, linewidth=1)
    ax2.axvline(0, linewidth=1)
    levels = [0.15, 0.30, 0.50, 0.70, 0.85]
    ax2.contourf(XX, YY, Z, levels=levels, alpha=0.85)
    ax2.contour(XX, YY, Z, levels=[float(threshold)], linewidths=2)

    for _, r in snap.iterrows():
        ax2.text(r["econ_x"] + 0.12, r["hier_y"] + 0.12, r["name"], fontsize=8)

    ax2.set_title(f"Southernness Influence Blobs — {int(year)}")
    ax2.set_xlabel("Economic Structure (← Agrarian / Extractive | Diversified / Modern →)")
    ax2.set_ylabel("Social Hierarchy (↓ Egalitarian | Hierarchical ↑)")
    ax2.set_xlim(-6, 6)
    ax2.set_ylim(-6, 6)
    ax2.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig2)

    # Downloads
    buf1, buf2 = io.BytesIO(), io.BytesIO()
    fig1.savefig(buf1, format="png", dpi=200, bbox_inches="tight")
    fig2.savefig(buf2, format="png", dpi=200, bbox_inches="tight")
    st.download_button("Download Scatter (PNG)", data=buf1.getvalue(), file_name=f"southern_quadrant_scatter_{int(year)}.png", mime="image/png")
    st.download_button("Download Blobs (PNG)", data=buf2.getvalue(), file_name=f"southern_blobs_{int(year)}.png", mime="image/png")
else:
    st.info("Need at least two places in the selected year (or via interpolation) to draw blobs.")
