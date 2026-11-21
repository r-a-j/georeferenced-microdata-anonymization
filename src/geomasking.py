"""Core geographic masking algorithms operating in projected UTM meters.

Implements three spatial anonymization methods benchmarked in Richter (2018):
1. Random Perturbation (fixed-radius uniform-area annulus with exclusion buffer)
2. Donut Geomasking (Hampton et al., 2010: density-adaptive annulus)
3. Verified Neighbor (Richter, 2018: k-nearest verified residential building centroids)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree

from .landscape import StudyLandscape, eval_corridor_boundary


@dataclass
class MaskingResult:
    method_name: str
    target_k: int | None
    env_matched: bool
    surrogates: np.ndarray           # (n_subjects, n_replicates, 2) UTM meters
    realized_k: np.ndarray           # (n_subjects,) empirical k-anonymity pool size
    on_residential_parcel: np.ndarray  # (n_subjects, n_replicates) bool
    env_preserved: np.ndarray          # (n_subjects, n_replicates) bool


def _infer_zone_at_coords(
    coords: np.ndarray,
    landscape: StudyLandscape,
    all_parcels_tree: cKDTree,
    parcel_has_public: np.ndarray,
) -> np.ndarray:
    """Assign environmental service zone to arbitrary continuous UTM coordinates."""
    flat = coords.reshape(-1, 2)
    _, nn_idx = all_parcels_tree.query(flat, k=1)
    has_pub = parcel_has_public[nn_idx]
    b_y = eval_corridor_boundary(
        flat[:, 0], landscape.x_min, landscape.x_max, landscape.boundary_y_fn_params
    )
    zones = np.where(
        ~has_pub,
        "Rural_Fringe_Zone",
        np.where(flat[:, 1] >= b_y, "North_Service_Zone", "South_Service_Zone"),
    )
    return zones.reshape(coords.shape[:-1])


def mask_random_perturbation(
    landscape: StudyLandscape,
    *,
    radius_m: float = 905.0,
    exclusion_m: float = 50.0,
    n_replicates: int = 50,
    target_k_eval: int = 100,
    seed: int = 100,
) -> MaskingResult:
    """Exact uniform-area random perturbation inside annulus [exclusion_m, radius_m]."""
    rng = np.random.default_rng(seed)
    subjects = landscape.subjects
    parcels = landscape.parcels
    subj_xy = subjects[["x", "y"]].to_numpy()
    n_subj = len(subj_xy)

    res_mask = parcels["is_residential"].to_numpy()
    res_xy = parcels.loc[res_mask, ["x", "y"]].to_numpy()
    res_tree = cKDTree(res_xy)
    all_tree = cKDTree(parcels[["x", "y"]].to_numpy())

    u = rng.uniform(0.0, 1.0, size=(n_subj, n_replicates))
    radii = np.sqrt(exclusion_m**2 + u * (radius_m**2 - exclusion_m**2))
    angles = rng.uniform(0.0, 2.0 * np.pi, size=(n_subj, n_replicates))

    sx = np.clip(subj_xy[:, [0]] + radii * np.cos(angles), landscape.x_min, landscape.x_max)
    sy = np.clip(subj_xy[:, [1]] + radii * np.sin(angles), landscape.y_min, landscape.y_max)
    surrogates = np.stack([sx, sy], axis=-1)

    # Empirical k: residential building centroids inside (exclusion_m, radius_m]
    n_outer = np.asarray(res_tree.query_ball_point(subj_xy, r=radius_m, return_length=True), dtype=int)
    n_inner = np.asarray(res_tree.query_ball_point(subj_xy, r=exclusion_m, return_length=True), dtype=int)
    realized_k = np.maximum(0, n_outer - n_inner)

    # Check whether continuous perturbed coordinate falls on/near an actual residential building (<= 25 m)
    flat = surrogates.reshape(-1, 2)
    nn_dist, _ = res_tree.query(flat, k=1)
    on_res = (nn_dist <= 25.0).reshape(n_subj, n_replicates)

    inferred_zones = _infer_zone_at_coords(
        surrogates, landscape, all_tree, parcels["has_public_water"].to_numpy()
    )
    env_preserved = inferred_zones == subjects["water_zone"].to_numpy()[:, None]

    return MaskingResult(
        method_name=f"Random Perturbation (r={int(radius_m)}m)",
        target_k=target_k_eval,
        env_matched=False,
        surrogates=surrogates,
        realized_k=realized_k,
        on_residential_parcel=on_res,
        env_preserved=env_preserved,
    )


def mask_donut_geomasking(
    landscape: StudyLandscape,
    *,
    k_min: int = 100,
    k_max_factor: float = 10.0,
    exclusion_m: float = 50.0,
    env_match: bool = True,
    n_replicates: int = 50,
    seed: int = 200,
) -> MaskingResult:
    """Density-adaptive Donut Geomasking (Hampton et al., 2010) in UTM meters."""
    rng = np.random.default_rng(seed)
    subjects = landscape.subjects
    parcels = landscape.parcels
    bg_density = landscape.block_groups.set_index("block_group_id")[
        "res_density_per_m2"
    ].to_dict()

    subj_xy = subjects[["x", "y"]].to_numpy()
    n_subj = len(subj_xy)
    k_max = k_min * k_max_factor
    max_cap_r = 0.45 * min(landscape.x_max - landscape.x_min, landscape.y_max - landscape.y_min)

    res_df = parcels.loc[parcels["is_residential"]].copy()
    res_tree = cKDTree(res_df[["x", "y"]].to_numpy())
    all_tree = cKDTree(parcels[["x", "y"]].to_numpy())
    has_pub_arr = parcels["has_public_water"].to_numpy()

    zone_trees: dict[str, cKDTree] = {
        str(z): cKDTree(grp[["x", "y"]].to_numpy())
        for z, grp in res_df.groupby("water_zone")
    }

    surrogates = np.zeros((n_subj, n_replicates, 2), dtype=float)
    realized_k = np.zeros(n_subj, dtype=int)

    for i in range(n_subj):
        bg_id = int(subjects.loc[i, "block_group_id"])
        rho = max(bg_density.get(bg_id, 1e-6), 2.5e-6)
        r1 = max(exclusion_m, np.sqrt(k_min / (np.pi * rho)))
        r2 = min(max_cap_r, max(r1 + 50.0, np.sqrt(k_max / (np.pi * rho))))
        true_zone = str(subjects.loc[i, "water_zone"])

        accepted = []
        attempts = 0
        while len(accepted) < n_replicates and attempts < 45:
            attempts += 1
            batch = max((n_replicates - len(accepted)) * 4, 64)
            u = rng.uniform(0.0, 1.0, size=batch)
            rad = np.sqrt(r1**2 + u * (r2**2 - r1**2))
            ang = rng.uniform(0.0, 2.0 * np.pi, size=batch)
            cand = np.column_stack(
                [subj_xy[i, 0] + rad * np.cos(ang), subj_xy[i, 1] + rad * np.sin(ang)]
            )
            in_bounds = (
                (cand[:, 0] >= landscape.x_min)
                & (cand[:, 0] <= landscape.x_max)
                & (cand[:, 1] >= landscape.y_min)
                & (cand[:, 1] <= landscape.y_max)
            )
            cand = cand[in_bounds]
            if len(cand) == 0:
                continue

            if env_match:
                cand_zones = _infer_zone_at_coords(cand, landscape, all_tree, has_pub_arr)
                cand = cand[cand_zones == true_zone]

            for pt in cand:
                accepted.append(pt)
                if len(accepted) == n_replicates:
                    break

        while len(accepted) < n_replicates:
            u = rng.uniform(0.0, 1.0)
            rad = np.sqrt(r1**2 + u * (r2**2 - r1**2))
            ang = rng.uniform(0.0, 2.0 * np.pi)
            pt = np.array(
                [
                    np.clip(subj_xy[i, 0] + rad * np.cos(ang), landscape.x_min, landscape.x_max),
                    np.clip(subj_xy[i, 1] + rad * np.sin(ang), landscape.y_min, landscape.y_max),
                ]
            )
            accepted.append(pt)

        surrogates[i] = np.asarray(accepted[:n_replicates])

        tree_eval = zone_trees[true_zone] if env_match else res_tree
        n_outer = len(tree_eval.query_ball_point(subj_xy[i], r=r2))
        n_inner = len(tree_eval.query_ball_point(subj_xy[i], r=exclusion_m))
        realized_k[i] = max(0, n_outer - n_inner)

    flat = surrogates.reshape(-1, 2)
    nn_dist, _ = res_tree.query(flat, k=1)
    on_res = (nn_dist <= 25.0).reshape(n_subj, n_replicates)

    inferred_zones = _infer_zone_at_coords(surrogates, landscape, all_tree, has_pub_arr)
    env_preserved = inferred_zones == subjects["water_zone"].to_numpy()[:, None]

    return MaskingResult(
        method_name=f"Donut Geomasking (k_min={k_min})",
        target_k=k_min,
        env_matched=env_match,
        surrogates=surrogates,
        realized_k=realized_k,
        on_residential_parcel=on_res,
        env_preserved=env_preserved,
    )


def mask_verified_neighbor(
    landscape: StudyLandscape,
    *,
    k: int = 100,
    exclusion_m: float = 50.0,
    env_match: bool = False,
    max_search_m: float | None = None,
    n_replicates: int = 50,
    seed: int = 300,
) -> MaskingResult:
    """Verified Neighbor geomasking (Richter, 2018) on real OSM residential centroids."""
    rng = np.random.default_rng(seed)
    subjects = landscape.subjects
    parcels = landscape.parcels
    res_df = parcels.loc[parcels["is_residential"]].reset_index(drop=True)

    subj_xy = subjects[["x", "y"]].to_numpy()
    n_subj = len(subj_xy)

    all_res_xy = res_df[["x", "y"]].to_numpy()
    all_res_zones = res_df["water_zone"].to_numpy()
    all_res_tree = cKDTree(all_res_xy)

    zone_pools: dict[str, tuple[cKDTree, np.ndarray, np.ndarray]] = {
        str(z): (cKDTree(grp[["x", "y"]].to_numpy()), grp[["x", "y"]].to_numpy(), grp["water_zone"].to_numpy())
        for z, grp in res_df.groupby("water_zone")
    }

    surrogates = np.zeros((n_subj, n_replicates, 2), dtype=float)
    realized_k = np.zeros(n_subj, dtype=int)
    env_preserved = np.zeros((n_subj, n_replicates), dtype=bool)

    for i in range(n_subj):
        true_zone = str(subjects.loc[i, "water_zone"])
        if env_match:
            tree, pool_xy, pool_zones = zone_pools[true_zone]
        else:
            tree, pool_xy, pool_zones = all_res_tree, all_res_xy, all_res_zones

        n_available = len(pool_xy)
        n_excl_approx = len(tree.query_ball_point(subj_xy[i], r=exclusion_m))
        q_k = min(n_available, k + n_excl_approx + 10)
        dists, idxs = tree.query(subj_xy[i], k=q_k)
        dists = np.atleast_1d(dists)
        idxs = np.atleast_1d(idxs)

        valid = dists > exclusion_m
        if max_search_m is not None:
            valid = valid & (dists <= max_search_m)

        valid_idxs = idxs[valid][:k]
        realized_k[i] = len(valid_idxs)

        if len(valid_idxs) == 0:
            fallback = idxs[dists > 0.0][:1]
            valid_idxs = fallback if len(fallback) else idxs[:1]

        chosen = rng.choice(valid_idxs, size=n_replicates, replace=True)
        surrogates[i] = pool_xy[chosen]
        env_preserved[i] = pool_zones[chosen] == true_zone

    label_scope = "Env-Matched" if env_match else "All Residences"
    return MaskingResult(
        method_name=f"Verified Neighbor ({label_scope}, k={k})",
        target_k=k,
        env_matched=env_match,
        surrogates=surrogates,
        realized_k=realized_k,
        on_residential_parcel=np.ones((n_subj, n_replicates), dtype=bool),
        env_preserved=env_preserved,
    )
