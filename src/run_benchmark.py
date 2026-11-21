"""Run the real-OSM georeferenced microdata anonymization benchmark and generate plots.

Executes a deterministic 50-replicate spatial privacy and utility benchmark on real
OpenStreetMap building centroids projected into metric UTM coordinates:
- Primary showcase: Dortmund, Germany (EPSG:25832, 125,612 OSM buildings)
  -> results/dortmund_benchmark_summary.csv, figures/*.png
- Cross-region validation: Erie County, NY (EPSG:26918, 53,394 OSM buildings)
  -> results/erie_county/benchmark_summary.csv, figures/erie_county/*.png
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.ticker import FuncFormatter
from scipy.spatial import cKDTree

from .landscape import (
    RegionKey,
    StudyLandscape,
    eval_corridor_boundary,
    load_osm_landscape,
)
from .geomasking import (
    MaskingResult,
    mask_donut_geomasking,
    mask_random_perturbation,
    mask_verified_neighbor,
)
from .metrics import (
    ClusterScanResult,
    compute_net_displacements,
    evaluate_method,
    kulldorff_poisson_scan,
    ripley_l_centered,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "figures"
RES_DIR = REPO_ROOT / "results"

# Clean scientific editorial palette
COL_DARK = "#1e293b"
COL_MUTED = "#94a3b8"
COL_RP = "#2563eb"
COL_RP500 = "#0284c7"
COL_DG = "#d97706"
COL_VN50 = "#059669"
COL_VN100 = "#0d9488"
COL_VN200 = "#7c3aed"
COL_VN_ENV = "#dc2626"

OSM_CREDIT = "Building centroids: OpenStreetMap contributors (ODbL 1.0)"


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 0.9,
            "axes.titlesize": 11.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.8,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
            "legend.fontsize": 8.0,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#e2e8f0",
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.7,
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        }
    )


def _km_formatter(decimals: int = 1) -> FuncFormatter:
    return FuncFormatter(lambda val, _pos: f"{val / 1000.0:.{decimals}f}")


def _add_credit(fig: plt.Figure, crs_label: str) -> None:
    fig.text(
        0.99,
        0.008,
        f"{OSM_CREDIT} | {crs_label}",
        ha="right",
        va="bottom",
        fontsize=7.6,
        color="#64748b",
    )


def plot_verified_neighbor_pipeline(landscape: StudyLandscape, out_dir: Path) -> None:
    """Figure 1: Step-by-step Verified Neighbor winnowing + rural-fringe contrast on real OSM."""
    parcels = landscape.parcels
    res_df = parcels.loc[parcels["is_residential"]].reset_index(drop=True)

    # Panel A subject: residential building near the corridor boundary in a mixed-use area
    b_y_all = eval_corridor_boundary(
        res_df["x"].to_numpy(),
        landscape.x_min,
        landscape.x_max,
        landscape.boundary_y_fn_params,
    )
    b_dist = np.abs(res_df["y"].to_numpy() - b_y_all)
    x_span = landscape.x_max - landscape.x_min
    cand = res_df.loc[
        (res_df["water_zone"] == "North_Service_Zone")
        & (b_dist > 55.0)
        & (b_dist < 190.0)
        & (res_df["x"] > landscape.x_min + 0.18 * x_span)
        & (res_df["x"] < landscape.x_max - 0.18 * x_span)
    ].copy()
    # Pick a candidate with moderate density (nice readable local plot with visible non-res buildings)
    cand["density_diff"] = np.abs(cand["local_res_500m"] - 280)
    subj = cand.sort_values("density_diff").iloc[0]
    sx, sy = float(subj["x"]), float(subj["y"])

    # Panel B subject: real sparse outskirt residential building with < 75 residences inside 905 m
    res_xy = res_df[["x", "y"]].to_numpy()
    res_tree_all = cKDTree(res_xy)
    pad = 1_400.0
    interior_mask = (
        (res_df["x"] > landscape.x_min + pad)
        & (res_df["x"] < landscape.x_max - pad)
        & (res_df["y"] > landscape.y_min + pad)
        & (res_df["y"] < landscape.y_max - pad)
    )
    sparse_cand = res_df.loc[interior_mask & (res_df["local_res_500m"] < 90)].copy()
    if len(sparse_cand) == 0:
        sparse_cand = res_df.loc[interior_mask].nsmallest(120, "local_res_500m").copy()

    sc_xy = sparse_cand[["x", "y"]].to_numpy()
    n_out = np.asarray(res_tree_all.query_ball_point(sc_xy, r=905.0, return_length=True), dtype=int)
    n_in = np.asarray(res_tree_all.query_ball_point(sc_xy, r=50.0, return_length=True), dtype=int)
    sparse_cand["rp_k"] = np.maximum(0, n_out - n_in)

    ideal = sparse_cand.loc[sparse_cand["rp_k"].between(22, 65)]
    if len(ideal):
        r_subj = ideal.sort_values("rp_k").iloc[len(ideal) // 2]
    else:
        r_subj = sparse_cand.sort_values("rp_k").iloc[0]
    rx, ry = float(r_subj["x"]), float(r_subj["y"])

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.9), dpi=220)

    # --- Panel A: Step-by-step Verified Neighbor winnowing near corridor boundary ---
    ax = axes[0]
    win = 420.0
    local = parcels.loc[
        (parcels["x"].between(sx - win, sx + win))
        & (parcels["y"].between(sy - win, sy + win))
    ].copy()
    local["dist"] = np.hypot(local["x"] - sx, local["y"] - sy)

    non_res = local.loc[~local["is_residential"]]
    other_zone = local.loc[
        local["is_residential"] & (local["water_zone"] != "North_Service_Zone")
    ]
    same_zone = local.loc[
        local["is_residential"] & (local["water_zone"] == "North_Service_Zone")
    ].sort_values("dist")
    qual = same_zone.loc[same_zone["dist"] > 50.0]
    pool_50 = qual.iloc[:50]
    chosen = pool_50.iloc[min(18, len(pool_50) - 1)]
    r_pool = float(pool_50["dist"].max())
    win_a = max(360.0, r_pool * 1.48)

    bx = np.linspace(sx - win_a, sx + win_a, 200)
    by = eval_corridor_boundary(
        bx, landscape.x_min, landscape.x_max, landscape.boundary_y_fn_params
    )
    ax.plot(bx, by, "--", color="#475569", lw=1.4, label="Service-zone boundary")

    ax.scatter(non_res["x"], non_res["y"], s=14, marker="s", color="#cbd5e1", alpha=0.85, label="Non-residential OSM building")
    ax.scatter(other_zone["x"], other_zone["y"], s=18, color="#93c5fd", alpha=0.85, label="Residential (other zone)")
    ax.scatter(same_zone["x"], same_zone["y"], s=20, color="#64748b", alpha=0.72, label="Residential (same zone)")
    ax.scatter(
        pool_50["x"],
        pool_50["y"],
        s=42,
        facecolors="none",
        edgecolors=COL_VN50,
        linewidths=1.4,
        label=f"Verified k=50 pool (>50 m)",
    )
    ax.scatter(
        [chosen["x"]],
        [chosen["y"]],
        s=125,
        color=COL_VN_ENV,
        edgecolors="white",
        linewidths=1.2,
        zorder=6,
        label="Selected surrogate",
    )
    ax.scatter(
        [sx],
        [sy],
        s=195,
        marker="*",
        color=COL_DARK,
        edgecolors="white",
        linewidths=0.8,
        zorder=7,
        label="True subject residence",
    )

    ax.add_patch(Circle((sx, sy), 50.0, fill=False, ls=":", ec=COL_VN_ENV, lw=1.4))
    ax.add_patch(Circle((sx, sy), r_pool, fill=False, ls="-", ec=COL_VN50, lw=1.25, alpha=0.9))
    ax.annotate(
        "50 m exclusion",
        xy=(sx + 36, sy + 36),
        xytext=(sx + win_a * 0.28, sy + win_a * 0.36),
        arrowprops=dict(arrowstyle="->", color=COL_VN_ENV, lw=1.0),
        fontsize=8.1,
        color=COL_VN_ENV,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.88),
        zorder=9,
    )
    ax.annotate(
        f"Exact k=50 radius ({r_pool:.0f} m)",
        xy=(sx - r_pool * 0.72, sy + r_pool * 0.69),
        xytext=(sx - win_a * 0.94, sy + win_a * 0.78),
        arrowprops=dict(arrowstyle="->", color=COL_VN50, lw=1.0),
        fontsize=8.1,
        color=COL_VN50,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.88),
        zorder=9,
    )

    ax.set_xlim(sx - win_a, sx + win_a)
    ax.set_ylim(sy - win_a, sy + win_a)
    ax.xaxis.set_major_formatter(_km_formatter(2))
    ax.yaxis.set_major_formatter(_km_formatter(2))
    ax.set_aspect("equal")
    ax.set_title(f"A. Verified Neighbor Winnowing ({landscape.region_label}, Env k = 50)")
    ax.set_xlabel("UTM Easting (km)")
    ax.set_ylabel("UTM Northing (km)")
    ax.legend(loc="lower left", fontsize=7.3, framealpha=0.95)
    ax.grid(True, alpha=0.4)

    # --- Panel B: Real sparse outskirt comparison (Fixed 905 m RP vs. Adaptive VN k = 100) ---
    ax2 = axes[1]
    r_win = 2_400.0
    local_r = parcels.loc[
        (parcels["x"].between(rx - r_win, rx + r_win))
        & (parcels["y"].between(ry - r_win, ry + r_win))
    ].copy()
    local_r["dist"] = np.hypot(local_r["x"] - rx, local_r["y"] - ry)
    r_non_res = local_r.loc[~local_r["is_residential"]]
    r_res = local_r.loc[local_r["is_residential"]].sort_values("dist")

    rp_in = r_res.loc[(r_res["dist"] > 50.0) & (r_res["dist"] <= 905.0)]
    vn_pool = r_res.loc[r_res["dist"] > 50.0].iloc[:100]
    vn_rad = float(vn_pool["dist"].max())
    r_lim = max(1_300.0, vn_rad * 1.18)

    ax2.scatter(r_non_res["x"], r_non_res["y"], s=11, marker="s", color="#cbd5e1", alpha=0.75, label="Non-residential OSM building")
    ax2.scatter(r_res["x"], r_res["y"], s=14, color="#94a3b8", alpha=0.55, label="Other residential buildings")
    ax2.scatter(
        vn_pool["x"],
        vn_pool["y"],
        s=32,
        facecolors="none",
        edgecolors=COL_VN50,
        lw=1.2,
        label=f"Verified Neighbor pool (k={len(vn_pool)}, r={vn_rad:.0f} m)",
    )
    ax2.scatter(
        rp_in["x"],
        rp_in["y"],
        s=26,
        color=COL_RP,
        alpha=0.9,
        label=f"Inside fixed 905 m RP (only {len(rp_in)} res.)",
    )
    ax2.scatter(
        [rx],
        [ry],
        s=195,
        marker="*",
        color=COL_DARK,
        edgecolors="white",
        lw=0.8,
        zorder=7,
        label="Low-density fringe subject",
    )

    ax2.add_patch(Circle((rx, ry), 905.0, fill=False, ls="--", ec=COL_RP, lw=1.8))
    ax2.add_patch(Circle((rx, ry), vn_rad, fill=False, ls="-", ec=COL_VN50, lw=1.5))

    ax2.set_xlim(rx - r_lim, rx + r_lim)
    ax2.set_ylim(ry - r_lim, ry + r_lim)
    ax2.xaxis.set_major_formatter(_km_formatter(1))
    ax2.yaxis.set_major_formatter(_km_formatter(1))
    ax2.set_aspect("equal")
    ax2.set_title(f"B. Sparse Outskirts: Fixed 905 m RP (k={len(rp_in)}) vs. Adaptive VN (k=100)")
    ax2.set_xlabel("UTM Easting (km)")
    ax2.set_ylabel("UTM Northing (km)")
    ax2.legend(loc="lower left", fontsize=7.4, framealpha=0.95)
    ax2.grid(True, alpha=0.4)

    fig.tight_layout(rect=[0, 0.025, 1, 1], pad=1.5)
    _add_credit(fig, landscape.crs_label)
    fig.savefig(out_dir / "verified_neighbor_pipeline.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_methods_comparison(
    landscape: StudyLandscape,
    results: dict[str, MaskingResult],
    out_dir: Path,
) -> None:
    """Figure 2: 2x2 city-scale real OSM map comparing masked surrogate footprints."""
    parcels = landscape.parcels
    res_df = parcels.loc[parcels["is_residential"]]
    non_res_df = parcels.loc[~parcels["is_residential"]]
    subj_xy = landscape.subjects[["x", "y"]].to_numpy()

    x_min, x_max = landscape.x_min, landscape.x_max
    y_min, y_max = landscape.y_min, landscape.y_max

    panels = [
        ("rp_905", "Random Perturbation (r = 905 m)", COL_RP),
        ("dg_100", "Donut Geomasking (k_min = 100, Env)", COL_DG),
        ("vn_all_50", "Verified Neighbor (All Residences, k = 50)", COL_VN50),
        ("vn_env_100", "Verified Neighbor (Env-Matched, k = 100)", COL_VN_ENV),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 11.0), dpi=220, sharex=True, sharey=True)
    bx = np.linspace(x_min, x_max, 250)
    by = eval_corridor_boundary(bx, x_min, x_max, landscape.boundary_y_fn_params)

    for ax, (key, title, color) in zip(axes.ravel(), panels):
        res = results[key]
        # Full real OSM building fabric (non-residential light grey + residential cool slate)
        if len(non_res_df):
            ax.scatter(
                non_res_df["x"],
                non_res_df["y"],
                s=0.6,
                color="#e2e8f0",
                alpha=0.55,
                rasterized=True,
                edgecolors="none",
            )
        ax.scatter(
            res_df["x"],
            res_df["y"],
            s=0.8,
            color="#94a3b8",
            alpha=0.45,
            rasterized=True,
            edgecolors="none",
        )
        # Service corridor boundary
        ax.plot(bx, by, "--", color="#475569", lw=1.1, alpha=0.85, label="Service-zone boundary")
        # 15 masked replicates per subject
        pts = res.surrogates[:, :15, :].reshape(-1, 2)
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=6.5,
            color=color,
            alpha=0.34,
            edgecolors="none",
            label="Masked surrogates",
        )
        # True sampled residential subjects
        ax.scatter(
            subj_xy[:, 0],
            subj_xy[:, 1],
            s=20,
            color=COL_DARK,
            edgecolors="white",
            lw=0.45,
            zorder=5,
            label="True subjects (n=160)",
        )

        valid_pct = 100.0 * float(res.on_residential_parcel.mean())
        env_pct = 100.0 * float(res.env_preserved.mean())
        ax.text(
            0.02,
            0.975,
            f"Valid res. (<=25 m): {valid_pct:.1f}%  |  Same zone: {env_pct:.1f}%",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.0,
            zorder=10,
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="#cbd5e1", alpha=0.96),
        )
        ax.set_title(title)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max + 0.068 * (y_max - y_min))
        ax.xaxis.set_major_formatter(_km_formatter(1))
        ax.yaxis.set_major_formatter(_km_formatter(1))
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)

    axes[0, 0].legend(loc="lower left", fontsize=7.5, framealpha=0.95)
    for ax in axes[1, :]:
        ax.set_xlabel("UTM Easting (km)")
    for ax in axes[:, 0]:
        ax.set_ylabel("UTM Northing (km)")

    n_total = len(parcels)
    n_res = len(res_df)
    fig.suptitle(
        f"{landscape.region_label}: Masked Surrogate Footprints on {n_total:,} Real OSM Buildings ({n_res:,} Residential)",
        fontsize=12.4,
        fontweight="bold",
        y=0.992,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.978])
    _add_credit(fig, landscape.crs_label)
    fig.savefig(out_dir / "spatial_methods_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_privacy_vs_displacement(
    landscape: StudyLandscape,
    results: dict[str, MaskingResult],
    out_dir: Path,
) -> None:
    """Figure 3: Rank-ordered realized spatial k-anonymity + net displacement CDF."""
    subj_xy = landscape.subjects[["x", "y"]].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4), dpi=220)

    # --- Panel A: Rank-ordered realized k across 160 subjects ---
    ax = axes[0]
    ranks = np.arange(1, len(subj_xy) + 1)

    rp905_k = np.sort(results["rp_905"].realized_k)[::-1]
    rp500_k = np.sort(results["rp_500"].realized_k)[::-1]
    dg_k = np.sort(results["dg_100"].realized_k)[::-1]
    vn100_k = np.sort(results["vn_all_100"].realized_k)[::-1]
    vn200_k = np.sort(results["vn_all_200"].realized_k)[::-1]

    ax.plot(ranks, rp905_k, color=COL_RP, lw=2.1, label="Random Perturbation (r = 905 m)")
    ax.plot(ranks, rp500_k, color=COL_RP500, lw=1.9, ls="-.", label="Random Perturbation (r = 500 m)")
    ax.plot(ranks, dg_k, color=COL_DG, lw=2.0, ls="--", label="Donut Geomasking (target k_min = 100)")
    ax.plot(ranks, vn100_k, color=COL_VN100, lw=2.2, label="Verified Neighbor (k = 100, exact)")
    ax.plot(ranks, vn200_k, color=COL_VN200, lw=1.9, ls=":", label="Verified Neighbor (k = 200, exact)")

    # Highlight shortfall below k = 100 for fixed-radius RP
    below_500 = int(np.sum(rp500_k < 100))
    below_905 = int(np.sum(rp905_k < 100))
    ax.axhline(100, color="#64748b", lw=1.0, ls=":")
    ax.fill_between(
        ranks,
        np.maximum(rp500_k, 1),
        100,
        where=(rp500_k < 100),
        color="#ef4444",
        alpha=0.20,
        label=f"Below k=100: {below_500}/160 (r=500m), {below_905}/160 (r=905m)",
    )

    ax.set_yscale("log")
    y_hi = max(float(dg_k.max()), float(rp905_k.max()), 500.0) * 1.45
    ax.set_ylim(5, y_hi)
    ax.set_xlim(1, len(subj_xy))
    ax.set_xlabel("Subject Rank Order (Highest to Lowest Realized Pool Size)")
    ax.set_ylabel("Realized Residential Surrogates (k, log scale)")
    ax.set_title(f"A. Realized Spatial k-Anonymity ({landscape.region_label}, n = 160)")
    ax.legend(loc="lower left", fontsize=7.5)
    ax.grid(True, which="both", alpha=0.35)

    # --- Panel B: Cumulative percent of subjects vs. net mean-center displacement ---
    ax2 = axes[1]
    cdf_specs = [
        ("vn_all_50", "Verified Neighbor (All, k=50)", COL_VN50, "-", 2.2),
        ("vn_all_100", "Verified Neighbor (All, k=100)", COL_VN100, "--", 2.0),
        ("vn_env_100", "Verified Neighbor (Env, k=100)", COL_VN_ENV, "-.", 2.0),
        ("vn_all_200", "Verified Neighbor (All, k=200)", COL_VN200, ":", 2.1),
        ("rp_500", "Random Perturbation (r=500 m)", COL_RP500, "-.", 1.9),
        ("rp_905", "Random Perturbation (r=905 m)", COL_RP, "-", 2.1),
        ("dg_100", "Donut Geomasking (k_min=100)", COL_DG, "-", 2.2),
    ]
    max_x = 1_500.0
    for key, label, col, ls, lw in cdf_specs:
        net_d = np.sort(compute_net_displacements(subj_xy, results[key].surrogates))
        max_x = max(max_x, float(np.percentile(net_d, 98)) * 1.35)
        pct = 100.0 * np.arange(1, len(net_d) + 1) / len(net_d)
        ax2.plot(np.maximum(net_d, 1.0), pct, color=col, ls=ls, lw=lw, label=label)

    ax2.set_xscale("log")
    ax2.set_xlim(4, max(2_500.0, min(max_x, 8_000.0)))
    ax2.set_ylim(0, 102)
    ax2.set_xlabel("Net Mean-Center Displacement Across 50 Replicates (m, log scale)")
    ax2.set_ylabel("Cumulative Percent of Subjects (%)")
    ax2.set_title("B. Net Displacement Distribution (Smaller = Less Spatial Bias)")
    ax2.legend(loc="lower right", fontsize=7.4)
    ax2.grid(True, which="both", alpha=0.35)

    fig.tight_layout(rect=[0, 0.025, 1, 1], pad=1.5)
    _add_credit(fig, landscape.crs_label)
    fig.savefig(out_dir / "privacy_vs_displacement.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ripley_and_cluster_utility(
    landscape: StudyLandscape,
    radii: np.ndarray,
    true_l: np.ndarray,
    l_reps_map: dict[str, np.ndarray],
    bg_freq_map: dict[str, np.ndarray],
    true_cluster: ClusterScanResult,
    out_dir: Path,
) -> None:
    """Figure 4: Besag L(r)-r preservation + Kulldorff Poisson spatial scan heatmaps."""
    fig = plt.figure(figsize=(13.6, 5.7), dpi=220)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.55, 1.55, 1.0, 1.0], wspace=0.28, hspace=0.38)

    # Left half: Besag L(r) - r across scales
    ax_l = fig.add_subplot(gs[:, :2])
    ax_l.plot(radii, true_l, color=COL_DARK, lw=2.6, zorder=6, label="Unmasked Ground Truth")

    l_specs = [
        ("vn_all_50", "Verified Neighbor (All, k=50)", COL_VN50, "-"),
        ("vn_env_100", "Verified Neighbor (Env, k=100)", COL_VN_ENV, "-."),
        ("rp_500", "Random Perturbation (r=500 m)", COL_RP500, "--"),
        ("rp_905", "Random Perturbation (r=905 m)", COL_RP, "--"),
        ("dg_100", "Donut Geomasking (k_min=100)", COL_DG, ":"),
    ]
    for key, label, col, ls in l_specs:
        arr = l_reps_map[key]
        mean_curve = arr.mean(axis=0)
        lo = np.percentile(arr, 5, axis=0)
        hi = np.percentile(arr, 95, axis=0)
        ax_l.plot(radii, mean_curve, color=col, ls=ls, lw=1.95, label=label)
        ax_l.fill_between(radii, lo, hi, color=col, alpha=0.09)

    ax_l.axhline(0.0, color="#94a3b8", lw=0.9, ls=":")
    ax_l.set_xlabel("Distance Scale r (m)")
    ax_l.set_ylabel("Besag's Centered L(r) - r (m)")
    ax_l.set_title(f"A. Spatial Pattern Fidelity on {landscape.region_label}: L(r) - r")
    ax_l.legend(loc="lower right", fontsize=7.8, framealpha=0.95)
    ax_l.grid(True, alpha=0.35)

    # Right half: 2x2 Kulldorff Poisson Primary Cluster inclusion heatmaps (out of 50 runs)
    hm_specs = [
        ("vn_all_50", "VN All (k=50)", gs[0, 2]),
        ("vn_env_100", "VN Env (k=100)", gs[0, 3]),
        ("rp_905", "Random Perturb. (905 m)", gs[1, 2]),
        ("dg_100", "Donut Geomask. (k=100)", gs[1, 3]),
    ]
    gnx, gny = landscape.grid_nx, landscape.grid_ny
    for key, title, slot in hm_specs:
        ax_h = fig.add_subplot(slot)
        grid = bg_freq_map[key].reshape(gny, gnx)
        ax_h.imshow(grid, origin="lower", cmap="Greys", vmin=0, vmax=50, aspect="equal")
        for bg_id in true_cluster.primary_block_groups:
            gy, gx = divmod(bg_id, gnx)
            ax_h.add_patch(
                Rectangle(
                    (gx - 0.5, gy - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="#dc2626",
                    lw=1.35,
                )
            )
        ax_h.set_title(title, fontsize=9.6)
        ax_h.set_xticks([])
        ax_h.set_yticks([])
        ax_h.set_xlim(-0.5, gnx - 0.5)
        ax_h.set_ylim(-0.5, gny - 0.5)

    fig.text(
        0.72,
        0.96,
        f"B. Kulldorff Poisson Scan Primary Cluster (50 Runs; Red = Ground Truth, {len(true_cluster.primary_block_groups)} tiles)",
        ha="center",
        fontsize=9.4,
        fontweight="bold",
        color=COL_DARK,
    )
    _add_credit(fig, landscape.crs_label)
    fig.savefig(out_dir / "ripley_and_cluster_utility.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_region(region: RegionKey, fig_out: Path, csv_out: Path) -> pd.DataFrame:
    """Run full 50-replicate benchmark and figure generation for one OSM region."""
    fig_out.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    landscape = load_osm_landscape(region, seed=42)
    subj_xy = landscape.subjects[["x", "y"]].to_numpy()
    area_m2 = (landscape.x_max - landscape.x_min) * (landscape.y_max - landscape.y_min)

    radii = np.linspace(150.0, 3_000.0, 30)
    true_l = ripley_l_centered(subj_xy, radii, area_m2)
    true_cluster = kulldorff_poisson_scan(subj_xy, landscape)

    n_rep = 50
    results: dict[str, MaskingResult] = {
        "rp_500": mask_random_perturbation(
            landscape, radius_m=500.0, exclusion_m=50.0, n_replicates=n_rep, target_k_eval=100, seed=100
        ),
        "rp_905": mask_random_perturbation(
            landscape, radius_m=905.0, exclusion_m=50.0, n_replicates=n_rep, target_k_eval=100, seed=101
        ),
        "dg_100": mask_donut_geomasking(
            landscape, k_min=100, k_max_factor=10.0, exclusion_m=50.0, env_match=True, n_replicates=n_rep, seed=201
        ),
        "vn_all_50": mask_verified_neighbor(
            landscape, k=50, exclusion_m=50.0, env_match=False, n_replicates=n_rep, seed=301
        ),
        "vn_all_100": mask_verified_neighbor(
            landscape, k=100, exclusion_m=50.0, env_match=False, n_replicates=n_rep, seed=302
        ),
        "vn_all_200": mask_verified_neighbor(
            landscape, k=200, exclusion_m=50.0, env_match=False, n_replicates=n_rep, seed=303
        ),
        "vn_all_350": mask_verified_neighbor(
            landscape, k=350, exclusion_m=50.0, env_match=False, n_replicates=n_rep, seed=304
        ),
        "vn_env_50": mask_verified_neighbor(
            landscape, k=50, exclusion_m=50.0, env_match=True, n_replicates=n_rep, seed=401
        ),
        "vn_env_100": mask_verified_neighbor(
            landscape, k=100, exclusion_m=50.0, env_match=True, n_replicates=n_rep, seed=402
        ),
        "vn_env_200": mask_verified_neighbor(
            landscape, k=200, exclusion_m=50.0, env_match=True, n_replicates=n_rep, seed=403
        ),
    }

    rows = []
    l_reps_map: dict[str, np.ndarray] = {}
    bg_freq_map: dict[str, np.ndarray] = {}
    for key, res in results.items():
        summary, l_reps, bg_freq = evaluate_method(
            landscape, res, radii, true_l, true_cluster
        )
        rows.append(summary)
        l_reps_map[key] = l_reps
        bg_freq_map[key] = bg_freq

    df = pd.DataFrame(rows)
    df.to_csv(csv_out, index=False)

    plot_verified_neighbor_pipeline(landscape, fig_out)
    plot_spatial_methods_comparison(landscape, results, fig_out)
    plot_privacy_vs_displacement(landscape, results, fig_out)
    plot_ripley_and_cluster_utility(
        landscape, radii, true_l, l_reps_map, bg_freq_map, true_cluster, fig_out
    )

    n_res = int(landscape.parcels["is_residential"].sum())
    print(f"\n=== {landscape.region_label.upper()} ({landscape.crs_label}) ===")
    print(
        f"OSM buildings: {len(landscape.parcels):,} | Residential: {n_res:,} | "
        f"Non-residential: {len(landscape.parcels) - n_res:,}"
    )
    print(
        f"Extent: {(landscape.x_max - landscape.x_min) / 1000:.2f} km x "
        f"{(landscape.y_max - landscape.y_min) / 1000:.2f} km | "
        f"Grid: {landscape.grid_nx} x {landscape.grid_ny} | "
        f"True cluster: {len(true_cluster.primary_block_groups)} tiles (LLR={true_cluster.llr:.2f})"
    )
    print(df.to_string(index=False))
    return df


def run_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    _configure_matplotlib()
    df_do = run_region(
        "dortmund",
        fig_out=FIG_DIR,
        csv_out=RES_DIR / "dortmund_benchmark_summary.csv",
    )
    df_erie = run_region(
        "erie_county",
        fig_out=FIG_DIR / "erie_county",
        csv_out=RES_DIR / "erie_county" / "benchmark_summary.csv",
    )
    return df_do, df_erie


if __name__ == "__main__":
    run_all()
