
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def compute_sii(row, weights):
    return float(
        row["economic"] * weights["economic"] +
        row["class_inequality"] * weights["class_inequality"] +
        row["racial_disparities"] * weights["racial_disparities"] +
        row["political_culture"] * weights["political_culture"] +
        row["cultural_identity"] * weights["cultural_identity"] +
        row["historical_continuity"] * weights["historical_continuity"]
    )

def main(args):
    df = pd.read_csv(args.csv)

    weights = {
        "economic": args.w_economic,
        "class_inequality": args.w_class,
        "racial_disparities": args.w_racial,
        "political_culture": args.w_politics,
        "cultural_identity": args.w_culture,
        "historical_continuity": args.w_history,
    }
    df["SII"] = df.apply(lambda r: compute_sii(r, weights), axis=1)

    # Scatter
    fig1, ax1 = plt.subplots(figsize=(7,7))
    ax1.axhline(0, linewidth=1)
    ax1.axvline(0, linewidth=1)
    sizes = (df["SII"] * 10.0).clip(lower=20.0, upper=1000.0)
    ax1.scatter(df["econ_x"], df["hier_y"], s=sizes)
    for _, r in df.iterrows():
        ax1.text(r["econ_x"] + 0.15, r["hier_y"] + 0.15, f'{r["name"]}\\n(SII {r["SII"]:.1f})', fontsize=8)
    ax1.set_title("Quadrant Map of Southernness (SII as Size)")
    ax1.set_xlabel("Economic Structure (← Agrarian / Extractive | Diversified / Modern →)")
    ax1.set_ylabel("Social Hierarchy (↓ Egalitarian | Hierarchical ↑)")
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(-6, 6)
    ax1.grid(True, linestyle="--", alpha=0.5)
    fig1.savefig(args.out_scatter, dpi=200, bbox_inches="tight")

    # Blobs
    if len(df) >= 2:
        X = df["econ_x"].to_numpy()
        Y = df["hier_y"].to_numpy()
        W = df["SII"].to_numpy()

        x_grid = np.linspace(-6, 6, 400)
        y_grid = np.linspace(-6, 6, 400)
        XX, YY = np.meshgrid(x_grid, y_grid)
        grid_pts = np.vstack([XX.ravel(), YY.ravel()])

        kde = gaussian_kde(np.vstack([X, Y]), weights=W, bw_method=args.kde_bw)
        Z = kde(grid_pts).reshape(XX.shape)
        Z = (Z - Z.min()) / (Z.max() - Z.min() + 1e-9)
        Z = Z ** args.sharpen_power

        fig2, ax2 = plt.subplots(figsize=(7,7))
        ax2.axhline(0, linewidth=1)
        ax2.axvline(0, linewidth=1)

        levels = [0.15, 0.30, 0.50, 0.70, 0.85]
        ax2.contourf(XX, YY, Z, levels=levels, alpha=0.85)
        ax2.contour(XX, YY, Z, levels=[args.threshold], linewidths=2)

        for _, r in df.iterrows():
            ax2.text(r["econ_x"] + 0.15, r["hier_y"] + 0.15, r["name"], fontsize=8)

        ax2.set_title("Southernness Influence Blobs (Sharper Edges)")
        ax2.set_xlabel("Economic Structure (← Agrarian / Extractive | Diversified / Modern →)")
        ax2.set_ylabel("Social Hierarchy (↓ Egalitarian | Hierarchical ↑)")
        ax2.set_xlim(-6, 6)
        ax2.set_ylim(-6, 6)
        ax2.grid(True, linestyle="--", alpha=0.4)
        fig2.savefig(args.out_blobs, dpi=200, bbox_inches="tight")
    else:
        print("Need at least two places to compute blobs.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate Southernness quadrant and blob graphics from CSV.")
    p.add_argument("csv", help="Input CSV with schema described in places_template.csv")
    p.add_argument("--out_scatter", default="southern_quadrant_scatter.png")
    p.add_argument("--out_blobs", default="southern_blobs.png")
    p.add_argument("--kde_bw", type=float, default=0.35, help="KDE bandwidth (0.1–1.0)")
    p.add_argument("--sharpen_power", type=float, default=4.0, help="Power transform to sharpen edges (>=1)")
    p.add_argument("--threshold", type=float, default=0.5, help="Primary contour threshold (0–1)")
    # Weights
    p.add_argument("--w_economic", type=float, default=1.0)
    p.add_argument("--w_class", type=float, default=1.0)
    p.add_argument("--w_racial", type=float, default=1.0)
    p.add_argument("--w_politics", type=float, default=1.0)
    p.add_argument("--w_culture", type=float, default=1.0)
    p.add_argument("--w_history", type=float, default=1.0)

    args = p.parse_args()
    main(args)
