"""Synthetic heterogeneous urban-suburban-rural parcel landscape generator.

Models a 12 km x 12 km projected coordinate study region (meters) with:
- 144 census block groups (1 km x 1 km grid)
- 12,000 property parcel centroids (residential & non-residential)
- Uneven urban core, secondary town, suburban belts, rural outskirts, and an unpopulated park/lake void
- Environmental exposure zones (District A public water, District B public water, private rural wells)
- 160 georeferenced subjects (100 background + 35 primary hotspot + 25 secondary hotspot)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StudyLandscape:
    parcels: pd.DataFrame
    subjects: pd.DataFrame
    block_groups: pd.DataFrame
    extent_m: float = 12_000.0
    grid_n: int = 12


def _water_boundary_y(x: np.ndarray) -> np.ndarray:
    """Smooth meandering boundary separating Water District A and District B."""
    return 6_100.0 + 1_150.0 * np.sin(2.0 * np.pi * x / 12_000.0 + 0.4)


def _in_lake_void(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Uninhabited lake / nature reserve polygon (elliptical void)."""
    dx = (x - 8_700.0) / 1_350.0
    dy = (y - 3_200.0) / 950.0
    return (dx * dx + dy * dy) < 1.0


def _sample_clamped_normal(
    rng: np.random.Generator,
    n: int,
    center: tuple[float, float],
    cov: list[list[float]],
    extent_m: float,
) -> np.ndarray:
    pts = []
    remaining = n
    while remaining > 0:
        draw = rng.multivariate_normal(center, cov, size=max(remaining * 2, 256))
        valid = (
            (draw[:, 0] >= 50.0)
            & (draw[:, 0] <= extent_m - 50.0)
            & (draw[:, 1] >= 50.0)
            & (draw[:, 1] <= extent_m - 50.0)
            & (~_in_lake_void(draw[:, 0], draw[:, 1]))
        )
        kept = draw[valid]
        take = kept[:remaining]
        pts.append(take)
        remaining -= len(take)
    return np.vstack(pts)


def generate_landscape(seed: int = 42) -> StudyLandscape:
    """Generate a reproducible parcel centroid dataset and 160 study subjects."""
    rng = np.random.default_rng(seed)
    extent_m = 12_000.0
    grid_n = 12
    cell_size = extent_m / grid_n

    # 1. Generate 12,000 property parcel centroids across heterogeneous density regimes
    urban_core = _sample_clamped_normal(
        rng, 4_500, (3_800.0, 7_600.0), [[1_050**2, 180**2], [180**2, 950**2]], extent_m
    )
    town_center = _sample_clamped_normal(
        rng, 2_400, (8_400.0, 8_200.0), [[950**2, -120**2], [-120**2, 850**2]], extent_m
    )
    south_suburb = _sample_clamped_normal(
        rng, 2_400, (4_400.0, 3_600.0), [[1_350**2, 220**2], [220**2, 1_100**2]], extent_m
    )

    # Sparse rural parcels across the entire extent (excluding lake void)
    rural_pts = []
    rem = 2_700
    while rem > 0:
        cand = rng.uniform(50.0, extent_m - 50.0, size=(rem * 2, 2))
        cand = cand[~_in_lake_void(cand[:, 0], cand[:, 1])]
        rural_pts.append(cand[:rem])
        rem -= min(rem, len(cand))
    rural = np.vstack(rural_pts)

    coords = np.vstack([urban_core, town_center, south_suburb, rural])
    n_parcels = len(coords)
    x = coords[:, 0]
    y = coords[:, 1]

    # Assign density tier label based on generation source
    density_tier = np.array(
        ["urban"] * len(urban_core)
        + ["town"] * len(town_center)
        + ["suburban"] * len(south_suburb)
        + ["rural"] * len(rural)
    )

    # Commercial/industrial corridor along y ~ 6200 m + random non-residential parcels (~22% total)
    corridor_dist = np.abs(y - (6_200.0 + 0.12 * (x - 6_000.0)))
    non_res_prob = np.where(corridor_dist < 420.0, 0.58, 0.16)
    non_res_prob = np.where(density_tier == "rural", 0.24, non_res_prob)
    is_residential = rng.random(n_parcels) >= non_res_prob

    # Environmental variable: water supply system (District A, District B, or Private Well)
    # Public water covers denser regions and most of the study area; outer rural fringe uses private wells
    dist_urban = np.hypot(x - 3_800.0, y - 7_600.0)
    dist_town = np.hypot(x - 8_400.0, y - 8_200.0)
    dist_suburb = np.hypot(x - 4_400.0, y - 3_600.0)
    min_hub_dist = np.minimum(np.minimum(dist_urban, dist_town), dist_suburb)

    has_public_water = (min_hub_dist < 3_600.0) | (rng.random(n_parcels) < 0.35)
    boundary = _water_boundary_y(x)
    water_zone = np.where(
        ~has_public_water,
        "Private_Well",
        np.where(y >= boundary, "Water_District_A", "Water_District_B"),
    )

    # Census block group IDs (12 x 12 grid, 1 km^2 each)
    gx = np.clip((x // cell_size).astype(int), 0, grid_n - 1)
    gy = np.clip((y // cell_size).astype(int), 0, grid_n - 1)
    block_group_id = gy * grid_n + gx

    parcels = pd.DataFrame(
        {
            "parcel_id": np.arange(n_parcels, dtype=int),
            "x": x,
            "y": y,
            "is_residential": is_residential,
            "has_public_water": has_public_water,
            "water_zone": water_zone,
            "density_tier": density_tier,
            "block_group_id": block_group_id,
        }
    )

    # 2. Build census block-group table with exact residential parcel counts & densities
    bg_rows = []
    res_counts = (
        parcels.loc[parcels["is_residential"]]
        .groupby("block_group_id")
        .size()
        .to_dict()
    )
    bg_area_m2 = cell_size * cell_size
    for gy_i in range(grid_n):
        for gx_i in range(grid_n):
            bg_id = gy_i * grid_n + gx_i
            cnt = int(res_counts.get(bg_id, 0))
            bg_rows.append(
                {
                    "block_group_id": bg_id,
                    "gx": gx_i,
                    "gy": gy_i,
                    "cx": (gx_i + 0.5) * cell_size,
                    "cy": (gy_i + 0.5) * cell_size,
                    "res_count": cnt,
                    "res_density_per_m2": cnt / bg_area_m2,
                }
            )
    block_groups = pd.DataFrame(bg_rows)

    # 3. Select 160 study subjects mirroring Richter (2018):
    #    100 background subjects across all residential parcels +
    #    35 primary cluster subjects + 25 secondary cluster subjects
    res_df = parcels.loc[parcels["is_residential"]].copy()
    res_indices = res_df.index.to_numpy()

    # Background sample stratified slightly so rural/edge subjects are well represented
    bg_weights = np.where(res_df["density_tier"] == "rural", 1.8, 1.0)
    bg_weights /= bg_weights.sum()
    bg_choice = rng.choice(res_indices, size=100, replace=False, p=bg_weights)

    remaining_res = res_df.drop(index=bg_choice)

    # Primary cluster: 35 subjects concentrated within 900 m of (3750, 7850)
    d_primary = np.hypot(remaining_res["x"] - 3_750.0, remaining_res["y"] - 7_850.0)
    cand_primary = remaining_res.loc[d_primary <= 900.0].index.to_numpy()
    c1_choice = rng.choice(cand_primary, size=35, replace=False)

    remaining_res = remaining_res.drop(index=c1_choice)

    # Secondary cluster: 25 subjects within 950 m of (8400, 8100) in town center
    d_secondary = np.hypot(remaining_res["x"] - 8_400.0, remaining_res["y"] - 8_100.0)
    cand_secondary = remaining_res.loc[d_secondary <= 950.0].index.to_numpy()
    c2_choice = rng.choice(cand_secondary, size=25, replace=False)

    subj_idx = np.concatenate([bg_choice, c1_choice, c2_choice])
    subjects = parcels.loc[subj_idx].copy().reset_index(drop=True)
    subjects["subject_id"] = np.arange(len(subjects), dtype=int)
    subjects["group"] = (
        ["background"] * 100 + ["primary_cluster"] * 35 + ["secondary_cluster"] * 25
    )

    return StudyLandscape(
        parcels=parcels,
        subjects=subjects,
        block_groups=block_groups,
        extent_m=extent_m,
        grid_n=grid_n,
    )
