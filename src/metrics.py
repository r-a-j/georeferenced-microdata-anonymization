"""Geospatial privacy, displacement, point-pattern, and hotspot scan metrics."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import f as f_dist

from .geomasking import MaskingResult
from .landscape import StudyLandscape


@dataclass(frozen=True)
class ClusterScanResult:
    primary_block_groups: set[int]
    llr: float
    center_bg: int
    radius_m: float


def compute_displacement_stats(
    subjects_xy: np.ndarray, surrogates: np.ndarray
) -> dict[str, float]:
    """Compute displacement distance, Hotelling's T^2 net shift, and Rayleigh bias."""
    # vectors: (n_subj, n_rep, 2)
    diff = surrogates - subjects_xy[:, None, :]
    dists = np.hypot(diff[:, :, 0], diff[:, :, 1])

    # Per-subject net mean-center shift across replicates
    mean_vec = diff.mean(axis=1)  # (n_subj, 2)
    net_disp = np.hypot(mean_vec[:, 0], mean_vec[:, 1])

    n_subj, n_rep, _ = diff.shape
    hotelling_sig = 0
    rayleigh_sig = 0

    for i in range(n_subj):
        # Hotelling's T^2 test for H0: E[diff_i] = (0, 0)
        d_i = diff[i]
        cov = np.cov(d_i.T, ddof=1)
        det = np.linalg.det(cov)
        if det > 1e-12:
            inv_cov = np.linalg.inv(cov)
            t2 = n_rep * float(mean_vec[i] @ inv_cov @ mean_vec[i])
            f_stat = (n_rep - 2) / (2.0 * (n_rep - 1)) * t2
            p_val = float(f_dist.sf(f_stat, 2, n_rep - 2))
            if p_val < 0.05:
                hotelling_sig += 1

        # Rayleigh test for directional uniformity of displacement vectors
        angles = np.arctan2(d_i[:, 1], d_i[:, 0])
        c_sum = np.sum(np.cos(angles))
        s_sum = np.sum(np.sin(angles))
        r_bar = np.hypot(c_sum, s_sum) / n_rep
        z = n_rep * (r_bar**2)
        p_ray = np.exp(-z) * (
            1.0
            + (2.0 * z - z**2) / (4.0 * n_rep)
            - (24.0 * z - 132.0 * z**2 + 76.0 * z**3 - 9.0 * z**4)
            / (288.0 * n_rep**2)
        )
        if p_ray < 0.05:
            rayleigh_sig += 1

    # Global mean center displacement across all subjects
    global_shift = float(
        np.hypot(
            surrogates[:, :, 0].mean() - subjects_xy[:, 0].mean(),
            surrogates[:, :, 1].mean() - subjects_xy[:, 1].mean(),
        )
    )

    return {
        "mean_disp_m": float(dists.mean()),
        "median_disp_m": float(np.median(dists)),
        "p90_disp_m": float(np.percentile(dists, 90)),
        "max_disp_m": float(dists.max()),
        "global_center_shift_m": global_shift,
        "median_net_disp_m": float(np.median(net_disp)),
        "q75_net_disp_m": float(np.percentile(net_disp, 75)),
        "pct_hotelling_sig": 100.0 * hotelling_sig / n_subj,
        "pct_rayleigh_sig": 100.0 * rayleigh_sig / n_subj,
    }


def compute_net_displacements(
    subjects_xy: np.ndarray, surrogates: np.ndarray
) -> np.ndarray:
    """Magnitude of each subject's mean displacement vector across replicates."""
    mean_vec = (surrogates - subjects_xy[:, None, :]).mean(axis=1)
    return np.hypot(mean_vec[:, 0], mean_vec[:, 1])


def ripley_l_centered(
    points: np.ndarray, radii: np.ndarray, area_m2: float
) -> np.ndarray:
    """Compute Besag's centered L(r) - r = sqrt(K(r)/pi) - r across distance scales."""
    n = len(points)
    if n < 2:
        return np.zeros_like(radii, dtype=float)
    dists = pdist(points)
    # Count ordered pairs within each radius r
    counts = np.array([(dists <= r).sum() * 2.0 for r in radii], dtype=float)
    k_vals = (area_m2 / (n * (n - 1))) * counts
    return np.sqrt(k_vals / np.pi) - radii


def kulldorff_poisson_scan(
    points_xy: np.ndarray,
    block_groups: pd.DataFrame,
    extent_m: float,
    grid_n: int,
    max_pop_fraction: float = 0.25,
) -> ClusterScanResult:
    """Kulldorff purely spatial discrete Poisson scan statistic over block groups.

    Mirrors the SaTScan discrete Poisson hotspot detector used in Richter (2018):
    evaluates circular scanning windows centered at every block group up to
    max_pop_fraction of total residential parcels and returns the primary cluster
    maximizing the Poisson log-likelihood ratio (LLR).
    """
    cell_size = extent_m / grid_n
    gx = np.clip((points_xy[:, 0] // cell_size).astype(int), 0, grid_n - 1)
    gy = np.clip((points_xy[:, 1] // cell_size).astype(int), 0, grid_n - 1)
    bg_ids = gy * grid_n + gx

    n_bg = len(block_groups)
    cases = np.bincount(bg_ids, minlength=n_bg).astype(float)
    pop = block_groups["res_count"].to_numpy(dtype=float)
    centers = block_groups[["cx", "cy"]].to_numpy(dtype=float)

    total_cases = float(cases.sum())
    total_pop = float(pop.sum())
    max_pop = total_pop * max_pop_fraction

    dist_mat = squareform(pdist(centers))

    best_llr = -1.0
    best_set: set[int] = set()
    best_center = 0
    best_radius = 0.0

    for center_idx in range(n_bg):
        order = np.argsort(dist_mat[center_idx])
        ordered_dists = dist_mat[center_idx, order]
        cum_pop = np.cumsum(pop[order])
        cum_cases = np.cumsum(cases[order])

        # Group equal-distance ties so circular windows are exact
        for k_end in range(len(order)):
            if k_end + 1 < len(order) and np.isclose(
                ordered_dists[k_end], ordered_dists[k_end + 1]
            ):
                continue

            p_in = cum_pop[k_end]
            if p_in <= 0.0 or p_in > max_pop:
                if p_in > max_pop:
                    break
                continue

            c_in = cum_cases[k_end]
            mu_in = total_cases * (p_in / total_pop)
            if c_in <= mu_in or c_in >= total_cases:
                continue

            c_out = total_cases - c_in
            mu_out = total_cases - mu_in
            llr = c_in * np.log(c_in / mu_in) + c_out * np.log(c_out / mu_out)
            if llr > best_llr:
                best_llr = float(llr)
                best_set = set(int(x) for x in order[: k_end + 1])
                best_center = center_idx
                best_radius = float(ordered_dists[k_end])

    return ClusterScanResult(
        primary_block_groups=best_set,
        llr=best_llr,
        center_bg=best_center,
        radius_m=best_radius,
    )


def evaluate_method(
    landscape: StudyLandscape,
    result: MaskingResult,
    radii: np.ndarray,
    true_l: np.ndarray,
    true_cluster: ClusterScanResult,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Run full privacy + spatial utility evaluation for one geomasking method."""
    subj_xy = landscape.subjects[["x", "y"]].to_numpy()
    area_m2 = landscape.extent_m**2
    n_rep = result.surrogates.shape[1]

    disp = compute_displacement_stats(subj_xy, result.surrogates)

    # Ripley L(r) - r across all 50 replicates
    l_reps = np.zeros((n_rep, len(radii)), dtype=float)
    in_true_counts = np.zeros(n_rep, dtype=float)
    outside_counts = np.zeros(n_rep, dtype=float)
    jaccards = np.zeros(n_rep, dtype=float)
    bg_hit_freq = np.zeros(len(landscape.block_groups), dtype=int)

    for rep in range(n_rep):
        pts_rep = result.surrogates[:, rep, :]
        l_reps[rep] = ripley_l_centered(pts_rep, radii, area_m2)

        scan = kulldorff_poisson_scan(
            pts_rep, landscape.block_groups, landscape.extent_m, landscape.grid_n
        )
        for bg in scan.primary_block_groups:
            bg_hit_freq[bg] += 1

        inter = len(scan.primary_block_groups & true_cluster.primary_block_groups)
        union = len(scan.primary_block_groups | true_cluster.primary_block_groups)
        outside = len(scan.primary_block_groups - true_cluster.primary_block_groups)
        in_true_counts[rep] = inter
        outside_counts[rep] = outside
        jaccards[rep] = inter / union if union > 0 else 0.0

    mean_l = l_reps.mean(axis=0)
    ripley_rmse = float(np.sqrt(np.mean((mean_l - true_l) ** 2)))
    ripley_rmse_1km = float(
        np.sqrt(np.mean((mean_l[radii <= 1000.0] - true_l[radii <= 1000.0]) ** 2))
    )

    target_k = result.target_k or 100
    pct_achieving_k = 100.0 * float(np.mean(result.realized_k >= target_k))

    summary = {
        "method": result.method_name,
        "target_k": target_k,
        "env_matched": result.env_matched,
        "pct_achieving_k": round(pct_achieving_k, 1),
        "min_realized_k": int(result.realized_k.min()),
        "median_realized_k": int(np.median(result.realized_k)),
        "max_realized_k": int(result.realized_k.max()),
        "pct_valid_residential": round(100.0 * float(result.on_residential_parcel.mean()), 1),
        "pct_env_preserved": round(100.0 * float(result.env_preserved.mean()), 1),
        "mean_disp_m": round(disp["mean_disp_m"], 1),
        "median_disp_m": round(disp["median_disp_m"], 1),
        "max_disp_m": round(disp["max_disp_m"], 1),
        "median_net_disp_m": round(disp["median_net_disp_m"], 1),
        "pct_hotelling_sig": round(disp["pct_hotelling_sig"], 1),
        "pct_rayleigh_sig": round(disp["pct_rayleigh_sig"], 1),
        "ripley_l_rmse_m": round(ripley_rmse, 1),
        "ripley_l_rmse_1km_m": round(ripley_rmse_1km, 1),
        "cluster_true_bg_size": len(true_cluster.primary_block_groups),
        "cluster_mean_in": round(float(in_true_counts.mean()), 2),
        "cluster_mean_outside": round(float(outside_counts.mean()), 2),
        "cluster_mean_jaccard": round(float(jaccards.mean()), 3),
    }
    return summary, l_reps, bg_hit_freq
