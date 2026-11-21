"""Core geographic masking algorithms for georeferenced microdata.

Implements three spatial anonymization methods benchmarked in Richter (2018):
1. Random Perturbation (fixed-radius annulus with exclusion buffer)
2. Donut Geomasking (Hampton et al., 2010: density-adaptive annulus)
3. Verified Neighbor (Richter, 2018: k-nearest verified residential parcel centroids)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .landscape import StudyLandscape, _in_lake_void, _water_boundary_y


@dataclass
class MaskingResult:
    method_name: str
    target_k: int | None
    env_matched: bool
    # Surrogate coordinates across M replicates: shape (n_subjects, n_replicates, 2)
    surrogates: np.ndarray
    # Realized pool size (empirical k-anonymity) per subject: shape (n_subjects,)
    realized_k: np.ndarray
    # Whether each surrogate coordinate lands directly on a verified residential parcel
    on_residential_parcel: np.ndarray
    # Whether each surrogate retains the subject's true environmental exposure zone
    env_preserved: np.ndarray


def _infer_water_zone_at_coords(
    coords: np.ndarray,
    all_parcels_tree: cKDTree,
    parcel_has_public_water: np.ndarray,
) -> np.ndarray:
    """Assign environmental water zone to arbitrary continuous (x, y) coordinates."""
    flat = coords.reshape(-1, 2)
    _, nn_idx = all_parcels_tree.query(flat, k=1)
    has_pw = parcel_has_public_water[nn_idx]
    boundary = _water_boundary_y(flat[:, 0])
    zones = np.where(
        ~has_pw,
        "Private_Well",
        np.where(flat[:, 1] >= boundary, "Water_District_A", "Water_District_B"),
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
    """Uniform-area random perturbation inside annulus [exclusion_m, radius_m]."""
    rng = np.random.default_rng(seed)
    subjects = landscape.subjects
    parcels = landscape.parcels
    subj_xy = subjects[["x", "y"]].to_numpy()
    n_subj = len(subj_xy)

    res_mask = parcels["is_residential"].to_numpy()
    res_xy = parcels.loc[res_mask, ["x", "y"]].to_numpy()
    res_tree = cKDTree(res_xy)
    all_tree = cKDTree(parcels[["x", "y"]].to_numpy())

    # Exact area-uniform polar sampling inside [exclusion_m, radius_m]
    u = rng.uniform(0.0, 1.0, size=(n_subj, n_replicates))
    radii = np.sqrt(exclusion_m**2 + u * (radius_m**2 - exclusion_m**2))
    angles = rng.uniform(0.0, 2.0 * np.pi, size=(n_subj, n_replicates))

    sx = np.clip(subj_xy[:, [0]] + radii * np.cos(angles), 0.0, landscape.extent_m)
    sy = np.clip(subj_xy[:, [1]] + radii * np.sin(angles), 0.0, landscape.extent_m)
    surrogates = np.stack([sx, sy], axis=-1)

    # Realized k-anonymity: count of residential parcels inside (exclusion_m, radius_m]
    realized_k = np.zeros(n_subj, dtype=int)
    for i in range(n_subj):
        n_outer = len(res_tree.query_ball_point(subj_xy[i], r=radius_m))
        n_inner = len(res_tree.query_ball_point(subj_xy[i], r=exclusion_m))
        realized_k[i] = max(0, n_outer - n_inner)

    # Check if continuous perturbed coordinate lands on plausible residential space
    # (outside lake void AND within 35 m of an actual residential parcel centroid)
    flat = surrogates.reshape(-1, 2)
    nn_dist, _ = res_tree.query(flat, k=1)
    on_res = (~_in_lake_void(flat[:, 0], flat[:, 1])) & (nn_dist <= 35.0)
    on_res = on_res.reshape(n_subj, n_replicates)

    # Check environmental exposure preservation
    inferred_zones = _infer_water_zone_at_coords(
        surrogates, all_tree, parcels["has_public_water"].to_numpy()
    )
    true_zones = subjects["water_zone"].to_numpy()[:, None]
    env_preserved = inferred_zones == true_zones

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
    """Density-adaptive Donut Geomasking (Hampton et al., 2010).

    Computes inner and outer displacement radii from each subject's census
    block-group residential density to target [k_min, k_max] residences.
    """
    rng = np.random.default_rng(seed)
    subjects = landscape.subjects
    parcels = landscape.parcels
    bg_density = landscape.block_groups.set_index("block_group_id")[
        "res_density_per_m2"
    ].to_dict()

    subj_xy = subjects[["x", "y"]].to_numpy()
    n_subj = len(subj_xy)
    k_max = k_min * k_max_factor

    res_df = parcels.loc[parcels["is_residential"]].copy()
    res_tree = cKDTree(res_df[["x", "y"]].to_numpy())
    all_tree = cKDTree(parcels[["x", "y"]].to_numpy())
    has_pw_arr = parcels["has_public_water"].to_numpy()

    # Build zone-specific KD-trees for exact realized-k evaluation
    zone_trees: dict[str, cKDTree] = {}
    for z, grp in res_df.groupby("water_zone"):
        zone_trees[str(z)] = cKDTree(grp[["x", "y"]].to_numpy())

    surrogates = np.zeros((n_subj, n_replicates, 2), dtype=float)
    realized_k = np.zeros(n_subj, dtype=int)

    for i in range(n_subj):
        bg_id = int(subjects.loc[i, "block_group_id"])
        rho = max(bg_density.get(bg_id, 1e-6), 2.5e-6)  # guard against empty cell
        r1 = max(exclusion_m, np.sqrt(k_min / (np.pi * rho)))
        r2 = min(landscape.extent_m * 0.45, max(r1 + 50.0, np.sqrt(k_max / (np.pi * rho))))
        true_zone = str(subjects.loc[i, "water_zone"])

        accepted = []
        attempts = 0
        while len(accepted) < n_replicates and attempts < 40:
            attempts += 1
            batch = max((n_replicates - len(accepted)) * 4, 64)
            u = rng.uniform(0.0, 1.0, size=batch)
            rad = np.sqrt(r1**2 + u * (r2**2 - r1**2))
            ang = rng.uniform(0.0, 2.0 * np.pi, size=batch)
            cand = np.column_stack(
                [subj_xy[i, 0] + rad * np.cos(ang), subj_xy[i, 1] + rad * np.sin(ang)]
            )
            in_bounds = (
                (cand[:, 0] >= 0.0)
                & (cand[:, 0] <= landscape.extent_m)
                & (cand[:, 1] >= 0.0)
                & (cand[:, 1] <= landscape.extent_m)
            )
            cand = cand[in_bounds]
            if len(cand) == 0:
                continue

            if env_match:
                cand_zones = _infer_water_zone_at_coords(cand, all_tree, has_pw_arr)
                cand = cand[cand_zones == true_zone]

            for pt in cand:
                accepted.append(pt)
                if len(accepted) == n_replicates:
                    break

        # Fallback if boundary geometry prevented full batch
        while len(accepted) < n_replicates:
            u = rng.uniform(0.0, 1.0)
            rad = np.sqrt(r1**2 + u * (r2**2 - r1**2))
            ang = rng.uniform(0.0, 2.0 * np.pi)
            pt = np.clip(
                subj_xy[i] + np.array([rad * np.cos(ang), rad * np.sin(ang)]),
                0.0,
                landscape.extent_m,
            )
            accepted.append(pt)

        surrogates[i] = np.asarray(accepted[:n_replicates])

        # Empirical k: qualifying residences within outer donut radius r2 outside exclusion_m
        tree_eval = zone_trees[true_zone] if env_match else res_tree
        n_outer = len(tree_eval.query_ball_point(subj_xy[i], r=r2))
        n_inner = len(tree_eval.query_ball_point(subj_xy[i], r=exclusion_m))
        realized_k[i] = max(0, n_outer - n_inner)

    flat = surrogates.reshape(-1, 2)
    nn_dist, _ = res_tree.query(flat, k=1)
    on_res = (~_in_lake_void(flat[:, 0], flat[:, 1])) & (nn_dist <= 35.0)
    on_res = on_res.reshape(n_subj, n_replicates)

    inferred_zones = _infer_water_zone_at_coords(surrogates, all_tree, has_pw_arr)
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
    """Verified Neighbor geomasking (Richter, 2018) using cKDTree spatial indexing.

    Filters real-property parcels to verified residential centroids (and optionally
    matching environmental exposure zones), removes centroids within exclusion_m,
    and samples surrogates uniformly from the exact k nearest qualifying parcels.
    """
    rng = np.random.default_rng(seed)
    subjects = landscape.subjects
    parcels = landscape.parcels
    res_df = parcels.loc[parcels["is_residential"]].reset_index(drop=True)

    subj_xy = subjects[["x", "y"]].to_numpy()
    n_subj = len(subj_xy)

    # Pre-build KD-trees for all residential parcels and per environmental zone
    all_res_xy = res_df[["x", "y"]].to_numpy()
    all_res_zones = res_df["water_zone"].to_numpy()
    all_res_tree = cKDTree(all_res_xy)

    zone_pools: dict[str, tuple[cKDTree, np.ndarray, np.ndarray]] = {}
    for z, grp in res_df.groupby("water_zone"):
        z_xy = grp[["x", "y"]].to_numpy()
        z_zones = grp["water_zone"].to_numpy()
        zone_pools[str(z)] = (cKDTree(z_xy), z_xy, z_zones)

    surrogates = np.zeros((n_subj, n_replicates, 2), dtype=float)
    realized_k = np.zeros(n_subj, dtype=int)
    env_preserved = np.zeros((n_subj, n_replicates), dtype=bool)

    for i in range(n_subj):
        true_zone = str(subjects.loc[i, "water_zone"])
        if env_match:
            tree, pool_xy, pool_zones = zone_pools[true_zone]
        else:
            tree, pool_xy, pool_zones = all_res_tree, all_res_xy, all_res_zones

        # Query enough neighbors to exclude parcels inside exclusion_m
        n_available = len(pool_xy)
        n_excl_approx = len(tree.query_ball_point(subj_xy[i], r=exclusion_m))
        q_k = min(n_available, k + n_excl_approx + 5)
        dists, idxs = tree.query(subj_xy[i], k=q_k)
        dists = np.atleast_1d(dists)
        idxs = np.atleast_1d(idxs)

        valid = dists > exclusion_m
        if max_search_m is not None:
            valid = valid & (dists <= max_search_m)

        valid_idxs = idxs[valid][:k]
        realized_k[i] = len(valid_idxs)

        if len(valid_idxs) == 0:
            # Edge fallback: nearest parcel outside 0 m
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
