"""Generate large-font, single-panel standalone Dortmund OSM figures for web and print."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from src.geomasking import (
    MaskingResult,
    mask_donut_geomasking,
    mask_random_perturbation,
    mask_verified_neighbor,
)
from src.landscape import eval_corridor_boundary, load_osm_landscape
from src.metrics import (
    compute_net_displacements,
    evaluate_method,
    kulldorff_poisson_scan,
    ripley_l_centered,
)
from src.run_benchmark import (
    COL_DARK,
    COL_DG,
    COL_RP,
    COL_RP500,
    COL_VN50,
    COL_VN100,
    COL_VN200,
    COL_VN_ENV,
    _km_formatter,
)


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 1.0,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.titlesize": 12.8,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 9.3,
            "grid.color": "#e2e8f0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.75,
        }
    )


def _save(fig: plt.Figure, png_path: Path, webp_dir: Path | None, webp_name: str) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    if webp_dir is not None:
        webp_dir.mkdir(parents=True, exist_ok=True)
        Image.open(png_path).convert("RGB").save(
            webp_dir / webp_name, format="WEBP", quality=92, method=6
        )


def build_dortmund_panels(png_dir: Path, webp_dir: Path | None = None) -> None:
    _apply_style()
    landscape = load_osm_landscape("dortmund", seed=42)
    parcels = landscape.parcels
    res_df = parcels.loc[parcels["is_residential"]].reset_index(drop=True)
    subj_xy = landscape.subjects[["x", "y"]].to_numpy()
    area_m2 = (landscape.x_max - landscape.x_min) * (landscape.y_max - landscape.y_min)
    radii = np.linspace(150.0, 3_000.0, 30)
    true_l = ripley_l_centered(subj_xy, radii, area_m2)
    true_cluster = kulldorff_poisson_scan(subj_xy, landscape)

    n_rep = 50
    results: dict[str, MaskingResult] = {
        "rp_500": mask_random_perturbation(landscape, radius_m=500.0, exclusion_m=50.0, n_replicates=n_rep, target_k_eval=100, seed=100),
        "rp_905": mask_random_perturbation(landscape, radius_m=905.0, exclusion_m=50.0, n_replicates=n_rep, target_k_eval=100, seed=101),
        "dg_100": mask_donut_geomasking(landscape, k_min=100, k_max_factor=10.0, exclusion_m=50.0, env_match=True, n_replicates=n_rep, seed=201),
        "vn_all_50": mask_verified_neighbor(landscape, k=50, exclusion_m=50.0, env_match=False, n_replicates=n_rep, seed=301),
        "vn_all_100": mask_verified_neighbor(landscape, k=100, exclusion_m=50.0, env_match=False, n_replicates=n_rep, seed=302),
        "vn_all_200": mask_verified_neighbor(landscape, k=200, exclusion_m=50.0, env_match=False, n_replicates=n_rep, seed=303),
        "vn_env_100": mask_verified_neighbor(landscape, k=100, exclusion_m=50.0, env_match=True, n_replicates=n_rep, seed=402),
    }

    l_reps_map: dict[str, np.ndarray] = {}
    bg_freq_map: dict[str, np.ndarray] = {}
    for key in ["vn_all_50", "vn_env_100", "rp_500", "rp_905", "dg_100"]:
        _, l_reps, bg_freq = evaluate_method(landscape, results[key], radii, true_l, true_cluster)
        l_reps_map[key] = l_reps
        bg_freq_map[key] = bg_freq

    # 1. Verified Neighbor Step-by-Step Winnowing (Dortmund Env k = 50)
    b_y_all = eval_corridor_boundary(res_df["x"].to_numpy(), landscape.x_min, landscape.x_max, landscape.boundary_y_fn_params)
    b_dist = np.abs(res_df["y"].to_numpy() - b_y_all)
    x_span = landscape.x_max - landscape.x_min
    cand = res_df.loc[
        (res_df["water_zone"] == "North_Service_Zone")
        & (b_dist > 55.0) & (b_dist < 190.0)
        & (res_df["x"] > landscape.x_min + 0.18 * x_span)
        & (res_df["x"] < landscape.x_max - 0.18 * x_span)
    ].copy()
    cand["density_diff"] = np.abs(cand["local_res_500m"] - 280)
    subj = cand.sort_values("density_diff").iloc[0]
    sx, sy = float(subj["x"]), float(subj["y"])

    win = 420.0
    local = parcels.loc[(parcels["x"].between(sx - win, sx + win)) & (parcels["y"].between(sy - win, sy + win))].copy()
    local["dist"] = np.hypot(local["x"] - sx, local["y"] - sy)
    non_res = local.loc[~local["is_residential"]]
    other_zone = local.loc[local["is_residential"] & (local["water_zone"] != "North_Service_Zone")]
    same_zone = local.loc[local["is_residential"] & (local["water_zone"] == "North_Service_Zone")].sort_values("dist")
    pool_50 = same_zone.loc[same_zone["dist"] > 50.0].iloc[:50]
    chosen = pool_50.iloc[min(18, len(pool_50) - 1)]
    r_pool = float(pool_50["dist"].max())
    win_a = max(360.0, r_pool * 1.48)

    fig, ax = plt.subplots(figsize=(8.0, 6.4), dpi=220)
    bx = np.linspace(sx - win_a, sx + win_a, 200)
    by = eval_corridor_boundary(bx, landscape.x_min, landscape.x_max, landscape.boundary_y_fn_params)
    ax.plot(bx, by, "--", color="#475569", lw=1.6, label="Service-zone boundary")
    ax.scatter(non_res["x"], non_res["y"], s=20, marker="s", color="#cbd5e1", alpha=0.85, label="Non-residential OSM")
    ax.scatter(other_zone["x"], other_zone["y"], s=24, color="#93c5fd", alpha=0.85, label="Residential (other zone)")
    ax.scatter(same_zone["x"], same_zone["y"], s=26, color="#64748b", alpha=0.72, label="Residential (same zone)")
    ax.scatter(pool_50["x"], pool_50["y"], s=58, facecolors="none", edgecolors=COL_VN50, linewidths=1.7, label="Verified k=50 pool (>50 m)")
    ax.scatter([chosen["x"]], [chosen["y"]], s=160, color=COL_VN_ENV, edgecolors="white", linewidths=1.4, zorder=6, label="Selected surrogate")
    ax.scatter([sx], [sy], s=240, marker="*", color=COL_DARK, edgecolors="white", linewidths=0.9, zorder=7, label="True subject residence")
    ax.add_patch(Circle((sx, sy), 50.0, fill=False, ls=":", ec=COL_VN_ENV, lw=1.7))
    ax.add_patch(Circle((sx, sy), r_pool, fill=False, ls="-", ec=COL_VN50, lw=1.6, alpha=0.92))
    ax.annotate("50 m exclusion", xy=(sx + 36, sy + 36), xytext=(sx + win_a * 0.25, sy + win_a * 0.36),
                arrowprops=dict(arrowstyle="->", color=COL_VN_ENV, lw=1.3), fontsize=9.8, color=COL_VN_ENV, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.92), zorder=9)
    ax.annotate(f"Exact k=50 radius ({r_pool:.0f} m)", xy=(sx - r_pool * 0.72, sy + r_pool * 0.69),
                xytext=(sx - win_a * 0.94, sy + win_a * 0.78),
                arrowprops=dict(arrowstyle="->", color=COL_VN50, lw=1.3), fontsize=9.8, color=COL_VN50, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.92), zorder=9)
    ax.set_xlim(sx - win_a, sx + win_a)
    ax.set_ylim(sy - win_a, sy + win_a)
    ax.xaxis.set_major_formatter(_km_formatter(2))
    ax.yaxis.set_major_formatter(_km_formatter(2))
    ax.set_aspect("equal")
    ax.set_title("Verified Neighbor Winnowing (Dortmund, Env k = 50)", loc="left", pad=10)
    ax.set_xlabel("UTM Easting (km, EPSG:25832)")
    ax.set_ylabel("UTM Northing (km)")
    ax.legend(loc="lower left", fontsize=8.8, framealpha=0.96)
    ax.grid(True, alpha=0.45)
    _save(fig, png_dir / "vn_winnowing_dortmund.png", webp_dir, "vn-winnowing-dortmund.webp")

    # 2. Sparse Outskirt Comparison (Fixed 905 m RP vs Adaptive VN k = 100)
    res_xy = res_df[["x", "y"]].to_numpy()
    res_tree_all = cKDTree(res_xy)
    pad = 1_400.0
    interior_mask = (
        (res_df["x"] > landscape.x_min + pad) & (res_df["x"] < landscape.x_max - pad)
        & (res_df["y"] > landscape.y_min + pad) & (res_df["y"] < landscape.y_max - pad)
    )
    sparse_cand = res_df.loc[interior_mask & (res_df["local_res_500m"] < 90)].copy()
    sc_xy = sparse_cand[["x", "y"]].to_numpy()
    sparse_cand["rp_k"] = np.maximum(
        0,
        np.asarray(res_tree_all.query_ball_point(sc_xy, r=905.0, return_length=True), dtype=int)
        - np.asarray(res_tree_all.query_ball_point(sc_xy, r=50.0, return_length=True), dtype=int),
    )
    ideal = sparse_cand.loc[sparse_cand["rp_k"].between(22, 65)]
    r_subj = ideal.sort_values("rp_k").iloc[len(ideal) // 2]
    rx, ry = float(r_subj["x"]), float(r_subj["y"])
    r_win = 2_400.0
    local_r = parcels.loc[(parcels["x"].between(rx - r_win, rx + r_win)) & (parcels["y"].between(ry - r_win, ry + r_win))].copy()
    local_r["dist"] = np.hypot(local_r["x"] - rx, local_r["y"] - ry)
    r_non_res = local_r.loc[~local_r["is_residential"]]
    r_res = local_r.loc[local_r["is_residential"]].sort_values("dist")
    rp_in = r_res.loc[(r_res["dist"] > 50.0) & (r_res["dist"] <= 905.0)]
    vn_pool = r_res.loc[r_res["dist"] > 50.0].iloc[:100]
    vn_rad = float(vn_pool["dist"].max())
    r_lim = max(1_300.0, vn_rad * 1.18)

    fig, ax = plt.subplots(figsize=(8.0, 6.4), dpi=220)
    ax.scatter(r_non_res["x"], r_non_res["y"], s=16, marker="s", color="#cbd5e1", alpha=0.78, label="Non-residential OSM")
    ax.scatter(r_res["x"], r_res["y"], s=20, color="#94a3b8", alpha=0.58, label="Other residential buildings")
    ax.scatter(vn_pool["x"], vn_pool["y"], s=48, facecolors="none", edgecolors=COL_VN50, lw=1.5, label=f"Adaptive VN pool (k=100, r={vn_rad:.0f} m)")
    ax.scatter(rp_in["x"], rp_in["y"], s=36, color=COL_RP, alpha=0.92, label=f"Inside fixed 905 m RP (only {len(rp_in)} res.)")
    ax.scatter([rx], [ry], s=240, marker="*", color=COL_DARK, edgecolors="white", lw=0.9, zorder=7, label="Low-density fringe subject")
    ax.add_patch(Circle((rx, ry), 905.0, fill=False, ls="--", ec=COL_RP, lw=2.1))
    ax.add_patch(Circle((rx, ry), vn_rad, fill=False, ls="-", ec=COL_VN50, lw=1.8))
    ax.set_xlim(rx - r_lim, rx + r_lim)
    ax.set_ylim(ry - r_lim, ry + r_lim)
    ax.xaxis.set_major_formatter(_km_formatter(1))
    ax.yaxis.set_major_formatter(_km_formatter(1))
    ax.set_aspect("equal")
    ax.set_title(f"Sparse Outskirts: Fixed 905 m RP (k={len(rp_in)}) vs. Adaptive VN (k=100)", loc="left", pad=10)
    ax.set_xlabel("UTM Easting (km, EPSG:25832)")
    ax.set_ylabel("UTM Northing (km)")
    ax.legend(loc="lower left", fontsize=8.8, framealpha=0.96)
    ax.grid(True, alpha=0.45)
    _save(fig, png_dir / "sparse_outskirt_comparison.png", webp_dir, "sparse-outskirt-comparison.webp")

    # 3. Realized Spatial k-Anonymity Distribution
    fig, ax = plt.subplots(figsize=(8.2, 5.9), dpi=220)
    ranks = np.arange(1, len(subj_xy) + 1)
    rp905_k = np.sort(results["rp_905"].realized_k)[::-1]
    rp500_k = np.sort(results["rp_500"].realized_k)[::-1]
    dg_k = np.sort(results["dg_100"].realized_k)[::-1]
    vn100_k = np.sort(results["vn_all_100"].realized_k)[::-1]
    vn200_k = np.sort(results["vn_all_200"].realized_k)[::-1]
    ax.plot(ranks, rp905_k, color=COL_RP, lw=2.5, label="Random Perturbation (r = 905 m)")
    ax.plot(ranks, rp500_k, color=COL_RP500, lw=2.2, ls="-.", label="Random Perturbation (r = 500 m)")
    ax.plot(ranks, dg_k, color=COL_DG, lw=2.3, ls="--", label="Donut Geomasking (target k_min = 100)")
    ax.plot(ranks, vn100_k, color=COL_VN100, lw=2.6, label="Verified Neighbor (k = 100, exact)")
    ax.plot(ranks, vn200_k, color=COL_VN200, lw=2.2, ls=":", label="Verified Neighbor (k = 200, exact)")
    below_500 = int(np.sum(rp500_k < 100))
    below_905 = int(np.sum(rp905_k < 100))
    ax.axhline(100, color="#64748b", lw=1.1, ls=":")
    ax.fill_between(ranks, np.maximum(rp500_k, 1), 100, where=(rp500_k < 100), color="#ef4444", alpha=0.22,
                    label=f"Below k=100: {below_500}/160 (500 m), {below_905}/160 (905 m)")
    ax.set_yscale("log")
    ax.set_ylim(5, max(float(dg_k.max()), float(rp905_k.max())) * 1.45)
    ax.set_xlim(1, len(subj_xy))
    ax.set_xlabel("Subject Rank Order (Highest to Lowest Pool Size)")
    ax.set_ylabel("Realized Residential Surrogates (k, log scale)")
    ax.set_title("Realized Spatial k-Anonymity Across 160 Dortmund Subjects", loc="left", pad=10)
    ax.legend(loc="lower left", fontsize=8.8, framealpha=0.96)
    ax.grid(True, which="both", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, png_dir / "realized_k_anonymity.png", webp_dir, "realized-k-anonymity.webp")

    # 4. Net Mean-Center Displacement CDF
    fig, ax = plt.subplots(figsize=(8.2, 5.9), dpi=220)
    cdf_specs = [
        ("vn_all_50", "Verified Neighbor (All, k=50)", COL_VN50, "-", 2.6),
        ("vn_all_100", "Verified Neighbor (All, k=100)", COL_VN100, "--", 2.3),
        ("vn_env_100", "Verified Neighbor (Env, k=100)", COL_VN_ENV, "-.", 2.3),
        ("vn_all_200", "Verified Neighbor (All, k=200)", COL_VN200, ":", 2.4),
        ("rp_500", "Random Perturbation (r=500 m)", COL_RP500, "-.", 2.2),
        ("rp_905", "Random Perturbation (r=905 m)", COL_RP, "-", 2.4),
        ("dg_100", "Donut Geomasking (k_min=100)", COL_DG, "-", 2.5),
    ]
    for key, lbl, col, ls, lw in cdf_specs:
        net_d = np.sort(compute_net_displacements(subj_xy, results[key].surrogates))
        pct = 100.0 * np.arange(1, len(net_d) + 1) / len(net_d)
        ax.plot(np.maximum(net_d, 1.0), pct, color=col, ls=ls, lw=lw, label=lbl)
    ax.set_xscale("log")
    ax.set_xlim(4, 2500)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Net Mean-Center Displacement Across 50 Replicates (m, log)")
    ax.set_ylabel("Cumulative Percent of Subjects (%)")
    ax.set_title("Net Displacement Distribution (Smaller = Less Spatial Bias)", loc="left", pad=10)
    ax.legend(loc="lower right", fontsize=8.8, framealpha=0.96)
    ax.grid(True, which="both", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, png_dir / "net_displacement_cdf.png", webp_dir, "net-displacement-cdf.webp")

    # 5. Besag L(r) - r Spatial Pattern Fidelity
    fig, ax = plt.subplots(figsize=(8.2, 5.9), dpi=220)
    ax.plot(radii, true_l, color=COL_DARK, lw=3.0, zorder=6, label="Unmasked Ground Truth")
    l_specs = [
        ("vn_all_50", "Verified Neighbor (All, k=50)", COL_VN50, "-"),
        ("vn_env_100", "Verified Neighbor (Env, k=100)", COL_VN_ENV, "-."),
        ("rp_500", "Random Perturbation (r=500 m)", COL_RP500, "--"),
        ("rp_905", "Random Perturbation (r=905 m)", COL_RP, "--"),
        ("dg_100", "Donut Geomasking (k_min=100)", COL_DG, ":"),
    ]
    for key, lbl, col, ls in l_specs:
        arr = l_reps_map[key]
        ax.plot(radii, arr.mean(axis=0), color=col, ls=ls, lw=2.3, label=lbl)
        ax.fill_between(radii, np.percentile(arr, 5, axis=0), np.percentile(arr, 95, axis=0), color=col, alpha=0.10)
    ax.axhline(0.0, color="#94a3b8", lw=1.0, ls=":")
    ax.set_xlabel("Distance Scale r (m)")
    ax.set_ylabel("Besag's Centered L(r) - r (m)")
    ax.set_title("Spatial Point-Pattern Fidelity on Dortmund: Besag L(r) - r", loc="left", pad=10)
    ax.legend(loc="lower right", fontsize=9.0, framealpha=0.96)
    ax.grid(True, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, png_dir / "besag_l_pattern_fidelity.png", webp_dir, "besag-l-pattern-fidelity.webp")

    # 6. Kulldorff Discrete Poisson Primary Cluster Recovery (2x2 crisp standalone heatmaps)
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.4), dpi=220)
    hm_specs = [
        ("vn_all_50", "VN All (k=50)\nJaccard 0.645"),
        ("vn_env_100", "VN Env (k=100)\nJaccard 0.808"),
        ("rp_905", "Random Perturb. (905 m)\nJaccard 0.339"),
        ("dg_100", "Donut Geomask. (k=100)\nJaccard 0.355"),
    ]
    gnx, gny = landscape.grid_nx, landscape.grid_ny
    for ax_h, (key, title) in zip(axes.ravel(), hm_specs):
        grid = bg_freq_map[key].reshape(gny, gnx)
        im = ax_h.imshow(grid, origin="lower", cmap="Greys", vmin=0, vmax=50, aspect="equal")
        for bg_id in true_cluster.primary_block_groups:
            gy, gx = divmod(bg_id, gnx)
            ax_h.add_patch(Rectangle((gx - 0.5, gy - 0.5), 1.0, 1.0, fill=False, edgecolor="#dc2626", lw=1.8))
        ax_h.set_title(title, fontsize=10.4, fontweight="bold")
        ax_h.set_xticks([])
        ax_h.set_yticks([])
    fig.suptitle("Kulldorff Poisson Cluster Recovery (50 Runs; Red = 3 True Tiles)", fontsize=12.2, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    cax = fig.add_axes([0.93, 0.14, 0.024, 0.72])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Runs Flagging Tile (out of 50)", fontsize=9.8)
    _save(fig, png_dir / "kulldorff_cluster_recovery.png", webp_dir, "kulldorff-cluster-recovery.webp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--png-dir", type=Path, default=Path("figures/panels"))
    parser.add_argument("--webp-dir", type=Path, default=None)
    args = parser.parse_args()
    build_dortmund_panels(args.png_dir, args.webp_dir)
    print("Built all 6 standalone Dortmund OSM panels cleanly.")
