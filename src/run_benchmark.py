"""Run the full georeferenced microdata anonymization benchmark and generate plots."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from scipy.spatial import cKDTree

from .landscape import StudyLandscape, _water_boundary_y, generate_landscape
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
COL_DG = "#d97706"
COL_VN50 = "#059669"
COL_VN100 = "#0d9488"
COL_VN200 = "#7c3aed"
COL_VN_ENV = "#dc2626"


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 0.9,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.3,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#e2e8f0",
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.7,
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        }
    )


def plot_verified_neighbor_pipeline(landscape: StudyLandscape) -> None:
    """Figure 1: Step-by-step Verified Neighbor winnowing + urban/rural contrast."""
    parcels = landscape.parcels
    res_df = parcels.loc[parcels["is_residential"]].reset_index(drop=True)

    # Pick a residential subject near the meandering water-district boundary in dense-to-mid area
    b_dist = np.abs(res_df["y"] - _water_boundary_y(res_df["x"].to_numpy()))
    cand = res_df.loc[
        (res_df["water_zone"] == "Water_District_A")
        & (b_dist > 70.0)
        & (b_dist < 220.0)
        & (res_df["x"] > 2_800.0)
        & (res_df["x"] < 5_200.0)
    ]
    subj = cand.iloc[len(cand) // 2]
    sx, sy = float(subj["x"]), float(subj["y"])

    # Pick a genuinely sparse rural-edge subject with < 35 residences inside 905 m
    res_tree_all = cKDTree(res_df[["x", "y"]].to_numpy())
    rural_cand = res_df.loc[
        (res_df["density_tier"] == "rural")
        & (res_df["x"] > 1_800.0)
        & (res_df["x"] < 10_200.0)
        & (res_df["y"] > 1_800.0)
        & (res_df["y"] < 10_200.0)
    ].copy()
    rp_counts = [
        len(res_tree_all.query_ball_point([row.x, row.y], r=905.0)) - 1
        for row in rural_cand.itertuples()
    ]
    rural_cand["rp_k"] = rp_counts
    sparse_match = rural_cand.loc[rural_cand["rp_k"].between(22, 34)]
    r_subj = sparse_match.iloc[0] if len(sparse_match) else rural_cand.sort_values("rp_k").iloc[0]
    rx, ry = float(r_subj["x"]), float(r_subj["y"])

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), dpi=220)

    # --- Panel A: Step-by-step Verified Neighbor winnowing near boundary ---
    ax = axes[0]
    win = 520.0
    local = parcels.loc[
        (parcels["x"].between(sx - win, sx + win))
        & (parcels["y"].between(sy - win, sy + win))
    ].copy()
    local["dist"] = np.hypot(local["x"] - sx, local["y"] - sy)

    non_res = local.loc[~local["is_residential"]]
    other_zone = local.loc[
        local["is_residential"] & (local["water_zone"] != "Water_District_A")
    ]
    same_zone = local.loc[
        local["is_residential"] & (local["water_zone"] == "Water_District_A")
    ].sort_values("dist")
    qual = same_zone.loc[same_zone["dist"] > 50.0]
    pool_50 = qual.iloc[:50]
    chosen = pool_50.iloc[17]
    r_pool = float(pool_50["dist"].max())

    # Plot water boundary
    bx = np.linspace(sx - win, sx + win, 200)
    ax.plot(bx, _water_boundary_y(bx), "--", color="#64748b", lw=1.4, label="Water district boundary")

    ax.scatter(non_res["x"], non_res["y"], s=18, marker="s", color="#cbd5e1", label="Non-residential parcel")
    ax.scatter(other_zone["x"], other_zone["y"], s=20, color="#93c5fd", alpha=0.85, label="Residential (other zone)")
    ax.scatter(same_zone["x"], same_zone["y"], s=22, color="#64748b", alpha=0.7, label="Residential (same zone)")
    ax.scatter(
        pool_50["x"],
        pool_50["y"],
        s=42,
        facecolors="none",
        edgecolors=COL_VN50,
        linewidths=1.4,
        label="Verified k=50 pool (>50 m)",
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
        s=190,
        marker="*",
        color=COL_DARK,
        edgecolors="white",
        linewidths=0.8,
        zorder=7,
        label="True subject parcel",
    )

    ax.add_patch(Circle((sx, sy), 50.0, fill=False, ls=":", ec=COL_VN_ENV, lw=1.4))
    ax.add_patch(Circle((sx, sy), r_pool, fill=False, ls="-", ec=COL_VN50, lw=1.2, alpha=0.85))
    ax.annotate("50 m exclusion", xy=(sx + 38, sy + 38), xytext=(sx + 105, sy + 120),
                arrowprops=dict(arrowstyle="->", color=COL_VN_ENV, lw=1.0),
                fontsize=8.2, color=COL_VN_ENV, fontweight="bold")
    ax.annotate(f"Exact k=50 radius ({r_pool:.0f} m)", xy=(sx - r_pool * 0.7, sy + r_pool * 0.7),
                xytext=(sx - win + 30, sy + win - 70),
                arrowprops=dict(arrowstyle="->", color=COL_VN50, lw=1.0),
                fontsize=8.2, color=COL_VN50, fontweight="bold")

    ax.set_xlim(sx - win, sx + win)
    ax.set_ylim(sy - win, sy + win)
    ax.set_aspect("equal")
    ax.set_title("A. Verified Neighbor Winnowing (Env-Matched, k = 50)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.95)
    ax.grid(True, alpha=0.45)

    # --- Panel B: Rural edge vulnerability of Fixed-Radius RP vs. Verified Neighbor (target k = 100) ---
    ax2 = axes[1]
    r_win = 1_850.0
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
    r_lim = max(1_450.0, vn_rad * 1.16)

    ax2.scatter(r_non_res["x"], r_non_res["y"], s=14, marker="s", color="#cbd5e1", label="Non-residential parcel")
    ax2.scatter(r_res["x"], r_res["y"], s=16, color="#94a3b8", alpha=0.6, label="Other residential parcels")
    ax2.scatter(vn_pool["x"], vn_pool["y"], s=34, facecolors="none", edgecolors=COL_VN50, lw=1.2,
                label=f"Verified Neighbor pool (k = {len(vn_pool)}, r = {vn_rad:.0f} m)")
    ax2.scatter(rp_in["x"], rp_in["y"], s=28, color=COL_RP, alpha=0.9,
                label=f"Inside fixed 905 m RP (only {len(rp_in)} res.)")
    ax2.scatter([rx], [ry], s=190, marker="*", color=COL_DARK, edgecolors="white", lw=0.8, zorder=7,
                label="Rural-edge subject")

    ax2.add_patch(Circle((rx, ry), 905.0, fill=False, ls="--", ec=COL_RP, lw=1.8))
    ax2.add_patch(Circle((rx, ry), vn_rad, fill=False, ls="-", ec=COL_VN50, lw=1.5))

    ax2.set_xlim(rx - r_lim, rx + r_lim)
    ax2.set_ylim(ry - r_lim, ry + r_lim)
    ax2.set_aspect("equal")
    ax2.set_title(f"B. Rural Edge: Fixed 905 m RP (k={len(rp_in)}) vs. Adaptive VN (k=100)")
    ax2.set_xlabel("Easting (m)")
    ax2.set_ylabel("Northing (m)")
    ax2.legend(loc="lower left", fontsize=7.6, framealpha=0.95)
    ax2.grid(True, alpha=0.45)

    fig.tight_layout(pad=1.6)
    fig.savefig(FIG_DIR / "verified_neighbor_pipeline.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_methods_comparison(
    landscape: StudyLandscape,
    results: dict[str, MaskingResult],
) -> None:
    """Figure 2: 2x2 spatial comparison of masked surrogates across dense & sparse areas."""
    parcels = landscape.parcels
    res_df = parcels.loc[parcels["is_residential"]]
    subj_xy = landscape.subjects[["x", "y"]].to_numpy()

    # Focus on a 7.5 km x 6.5 km region spanning urban core, water boundary, suburb, and rural fringe
    x_min, x_max = 1_500.0, 9_500.0
    y_min, y_max = 2_000.0, 9_500.0

    panels = [
        ("rp_905", "Random Perturbation (r = 905 m)", COL_RP),
        ("dg_100", "Donut Geomasking (k_min = 100, Env)", COL_DG),
        ("vn_all_50", "Verified Neighbor (All Residences, k = 50)", COL_VN50),
        ("vn_env_100", "Verified Neighbor (Env-Matched, k = 100)", COL_VN_ENV),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 10.6), dpi=220, sharex=True, sharey=True)
    bx = np.linspace(x_min, x_max, 250)
    by = _water_boundary_y(bx)

    # Lake ellipse outline
    ang = np.linspace(0, 2 * np.pi, 160)
    lake_x = 8_700.0 + 1_350.0 * np.cos(ang)
    lake_y = 3_200.0 + 950.0 * np.sin(ang)

    in_view_res = res_df.loc[
        res_df["x"].between(x_min, x_max) & res_df["y"].between(y_min, y_max)
    ]

    for ax, (key, title, color) in zip(axes.ravel(), panels):
        res = results[key]
        # Background parcels
        ax.scatter(in_view_res["x"], in_view_res["y"], s=2.5, color="#cbd5e1", alpha=0.45, rasterized=True)
        # Lake void
        ax.fill(lake_x, lake_y, color="#e0f2fe", edgecolor="#7dd3fc", lw=1.0, alpha=0.85, label="Uninhabited lake")
        # Water district boundary
        ax.plot(bx, by, "--", color="#64748b", lw=1.1, alpha=0.85, label="Water district boundary")
        # 15 surrogates per subject for visual clarity
        pts = res.surrogates[:, :15, :].reshape(-1, 2)
        ax.scatter(pts[:, 0], pts[:, 1], s=7.5, color=color, alpha=0.32, edgecolors="none", label="Masked surrogates")
        # True subjects
        ax.scatter(subj_xy[:, 0], subj_xy[:, 1], s=22, color=COL_DARK, edgecolors="white", lw=0.45, zorder=5,
                   label="True subjects (n=160)")

        valid_pct = 100.0 * float(res.on_residential_parcel.mean())
        env_pct = 100.0 * float(res.env_preserved.mean())
        ax.text(
            0.02,
            0.97,
            f"Valid res.: {valid_pct:.1f}%  |  Same water zone: {env_pct:.1f}%",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.2,
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="#cbd5e1", alpha=0.94),
        )
        ax.set_title(title)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    axes[0, 0].legend(loc="lower left", fontsize=7.6, framealpha=0.94)
    for ax in axes[1, :]:
        ax.set_xlabel("Easting (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Northing (m)")

    fig.suptitle("Spatial Footprint of Masked Surrogates Across Urban, Suburban & Rural Regimes",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(FIG_DIR / "spatial_methods_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_privacy_vs_displacement(
    landscape: StudyLandscape,
    results: dict[str, MaskingResult],
) -> None:
    """Figure 3: Rank-ordered empirical k-anonymity + cumulative net displacement CDF."""
    subj_xy = landscape.subjects[["x", "y"]].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=220)

    # --- Panel A: Rank-ordered realized k across 160 subjects (cf. Richter Fig. 3) ---
    ax = axes[0]
    ranks = np.arange(1, len(subj_xy) + 1)

    rp_k = np.sort(results["rp_905"].realized_k)[::-1]
    dg_k = np.sort(results["dg_100"].realized_k)[::-1]
    vn100_k = np.sort(results["vn_all_100"].realized_k)[::-1]
    vn200_k = np.sort(results["vn_all_200"].realized_k)[::-1]

    ax.plot(ranks, rp_k, color=COL_RP, lw=2.2, label="Random Perturbation (r = 905 m)")
    ax.plot(ranks, dg_k, color=COL_DG, lw=2.0, ls="--", label="Donut Geomasking (target k_min = 100)")
    ax.plot(ranks, vn100_k, color=COL_VN100, lw=2.2, label="Verified Neighbor (k = 100, exact)")
    ax.plot(ranks, vn200_k, color=COL_VN200, lw=1.9, ls=":", label="Verified Neighbor (k = 200, exact)")

    # Shade vulnerable region where RP < 100
    below_100 = int(np.sum(rp_k < 100))
    ax.axhline(100, color="#64748b", lw=1.0, ls="-.")
    ax.fill_between(ranks, rp_k, 100, where=(rp_k < 100), color="#ef4444", alpha=0.22,
                    label=f"RP shortfall below k=100 ({below_100}/160 subjects)")

    ax.set_yscale("log")
    ax.set_ylim(8, max(float(dg_k.max()), float(rp_k.max())) * 1.4)
    ax.set_xlim(1, len(subj_xy))
    ax.set_xlabel("Subject Rank Order (Highest to Lowest Realized Pool Size)")
    ax.set_ylabel("Realized Potential Surrogates (k, log scale)")
    ax.set_title("A. Realized Spatial k-Anonymity Across 160 Subjects")
    ax.legend(loc="lower left", fontsize=7.8)
    ax.grid(True, which="both", alpha=0.35)

    # --- Panel B: Cumulative percent of subjects vs. net mean-center displacement (cf. Richter Fig. 5) ---
    ax2 = axes[1]
    cdf_specs = [
        ("vn_all_50", "Verified Neighbor (All, k=50)", COL_VN50, "-", 2.2),
        ("vn_all_100", "Verified Neighbor (All, k=100)", COL_VN100, "--", 2.0),
        ("vn_env_100", "Verified Neighbor (Env, k=100)", COL_VN_ENV, "-.", 2.0),
        ("vn_all_200", "Verified Neighbor (All, k=200)", COL_VN200, ":", 2.1),
        ("rp_905", "Random Perturbation (r=905 m)", COL_RP, "-", 2.2),
        ("dg_100", "Donut Geomasking (k_min=100)", COL_DG, "-", 2.3),
    ]
    for key, label, col, ls, lw in cdf_specs:
        net_d = np.sort(compute_net_displacements(subj_xy, results[key].surrogates))
        pct = 100.0 * np.arange(1, len(net_d) + 1) / len(net_d)
        ax2.plot(np.maximum(net_d, 1.0), pct, color=col, ls=ls, lw=lw, label=label)

    ax2.set_xscale("log")
    ax2.set_xlim(5, 3_500)
    ax2.set_ylim(0, 102)
    ax2.set_xlabel("Net Mean-Center Displacement Across 50 Replicates (m, log scale)")
    ax2.set_ylabel("Cumulative Percent of Subjects (%)")
    ax2.set_title("B. Net Displacement Distribution (Smaller = Less Spatial Bias)")
    ax2.legend(loc="lower right", fontsize=7.8)
    ax2.grid(True, which="both", alpha=0.35)

    fig.tight_layout(pad=1.6)
    fig.savefig(FIG_DIR / "privacy_vs_displacement.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ripley_and_cluster_utility(
    landscape: StudyLandscape,
    radii: np.ndarray,
    true_l: np.ndarray,
    l_reps_map: dict[str, np.ndarray],
    bg_freq_map: dict[str, np.ndarray],
    true_cluster: ClusterScanResult,
) -> None:
    """Figure 4: Ripley L(r)-r preservation + Kulldorff Poisson hotspot heatmaps."""
    fig = plt.figure(figsize=(13.6, 5.6), dpi=220)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.55, 1.55, 1.0, 1.0], wspace=0.28, hspace=0.36)

    # Left half: Besag L(r) - r comparison across scales (full height across 2 rows)
    ax_l = fig.add_subplot(gs[:, :2])
    ax_l.plot(radii, true_l, color=COL_DARK, lw=2.6, zorder=6, label="Unmasked Ground Truth")

    l_specs = [
        ("vn_all_50", "Verified Neighbor (All, k=50)", COL_VN50, "-"),
        ("vn_env_100", "Verified Neighbor (Env, k=100)", COL_VN_ENV, "-."),
        ("rp_905", "Random Perturbation (r=905 m)", COL_RP, "--"),
        ("dg_100", "Donut Geomasking (k_min=100)", COL_DG, ":"),
    ]
    for key, label, col, ls in l_specs:
        arr = l_reps_map[key]
        mean_curve = arr.mean(axis=0)
        lo = np.percentile(arr, 5, axis=0)
        hi = np.percentile(arr, 95, axis=0)
        ax_l.plot(radii, mean_curve, color=col, ls=ls, lw=2.0, label=label)
        ax_l.fill_between(radii, lo, hi, color=col, alpha=0.11)

    ax_l.axhline(0.0, color="#94a3b8", lw=0.9, ls=":")
    ax_l.set_xlabel("Distance Scale r (m)")
    ax_l.set_ylabel("Besag's Centered L(r) - r (m)")
    ax_l.set_title("A. Spatial Pattern Fidelity: L(r) - r (Mean & 90% Envelope)")
    ax_l.legend(loc="upper left", fontsize=8.0)
    ax_l.grid(True, alpha=0.35)

    # Right half: 2x2 Kulldorff Poisson Primary Cluster inclusion heatmaps (out of 50 runs)
    hm_specs = [
        ("vn_all_50", "VN All (k=50)", gs[0, 2]),
        ("vn_env_100", "VN Env (k=100)", gs[0, 3]),
        ("rp_905", "Random Perturb.", gs[1, 2]),
        ("dg_100", "Donut Geomask.", gs[1, 3]),
    ]
    gn = landscape.grid_n
    for key, title, slot in hm_specs:
        ax_h = fig.add_subplot(slot)
        grid = bg_freq_map[key].reshape(gn, gn)
        ax_h.imshow(grid, origin="lower", cmap="Greys", vmin=0, vmax=50, aspect="equal")
        # Highlight ground-truth primary cluster block groups with red outlines
        for bg_id in true_cluster.primary_block_groups:
            gy, gx = divmod(bg_id, gn)
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
        ax_h.set_title(title, fontsize=9.8)
        ax_h.set_xticks([])
        ax_h.set_yticks([])
        # Focus nicely around northern urban/suburban region where primary cluster sits
        ax_h.set_xlim(-0.5, gn - 0.5)
        ax_h.set_ylim(-0.5, gn - 0.5)

    fig.text(
        0.72,
        0.96,
        "B. Kulldorff Spatial Scan Primary Cluster (50 Runs; Red = Ground Truth)",
        ha="center",
        fontsize=10.0,
        fontweight="bold",
        color=COL_DARK,
    )
    fig.savefig(FIG_DIR / "ripley_and_cluster_utility.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_all() -> pd.DataFrame:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()

    landscape = generate_landscape(seed=42)
    subj_xy = landscape.subjects[["x", "y"]].to_numpy()

    radii = np.linspace(150.0, 3_000.0, 30)
    true_l = ripley_l_centered(subj_xy, radii, landscape.extent_m**2)
    true_cluster = kulldorff_poisson_scan(
        subj_xy, landscape.block_groups, landscape.extent_m, landscape.grid_n
    )

    n_rep = 50
    results: dict[str, MaskingResult] = {
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
    df.to_csv(RES_DIR / "benchmark_summary.csv", index=False)

    plot_verified_neighbor_pipeline(landscape)
    plot_spatial_methods_comparison(landscape, results)
    plot_privacy_vs_displacement(landscape, results)
    plot_ripley_and_cluster_utility(
        landscape, radii, true_l, l_reps_map, bg_freq_map, true_cluster
    )

    print("=== LANDSCAPE SUMMARY ===")
    n_res = int(landscape.parcels["is_residential"].sum())
    print(f"Total parcels: {len(landscape.parcels)} | Residential: {n_res} | Non-residential: {len(landscape.parcels) - n_res}")
    print(f"Subjects: {len(landscape.subjects)} | Ground-truth primary cluster block groups: {len(true_cluster.primary_block_groups)} (LLR={true_cluster.llr:.2f})")
    print("\n=== BENCHMARK RESULTS ===")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    run_all()
