"""Real OpenStreetMap building-centroid landscape loader for Dortmund & Erie County.

Loads compressed OSM building centroids projected to exact metric UTM coordinates:
- Dortmund, Germany: EPSG:25832 (UTM Zone 32N), 125,612 real buildings
- Erie County, NY:   EPSG:26918 (UTM Zone 18N), 53,394 real buildings (Richter 2018 study county)

Partitions each study area into a metric 1 km-class block-group grid, assigns a
geographic environmental service-zone layer (North/South corridor + sparse rural
fringe), and samples 160 residential building centroids (100 background + 35 primary
cluster + 25 secondary cluster) with zero personal data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

RegionKey = Literal["dortmund", "erie_county"]


@dataclass(frozen=True)
class StudyLandscape:
    region_key: RegionKey
    region_label: str
    crs_label: str
    parcels: pd.DataFrame
    subjects: pd.DataFrame
    block_groups: pd.DataFrame
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    grid_nx: int
    grid_ny: int
    boundary_y_fn_params: tuple[float, float, float]  # (base_y, amp_y, phase)


def eval_corridor_boundary(
    x: np.ndarray, x_min: float, x_max: float, params: tuple[float, float, float]
) -> np.ndarray:
    """Smooth east-west geographic boundary dividing North and South service zones."""
    base_y, amp_y, phase = params
    span = max(x_max - x_min, 1.0)
    return base_y + amp_y * np.sin(2.0 * np.pi * (x - x_min) / span + phase)


def load_osm_landscape(region: RegionKey = "dortmund", seed: int = 42) -> StudyLandscape:
    """Load real OSM building centroids and construct the 160-subject benchmark."""
    rng = np.random.default_rng(seed)

    if region == "dortmund":
        csv_path = DATA_DIR / "dortmund_osm_buildings.csv.gz"
        region_label = "Dortmund, Germany"
        crs_label = "EPSG:25832 (UTM 32N)"
        # B1 / Emscher east-west corridor dividing central/north Dortmund from southern districts
        boundary_params = (5_705_400.0, 420.0, 0.35)
        rural_neighbor_thresh = 110
        # Real Dortmund neighborhood clusters (UTM 32N meters):
        # Primary: Hoerde / Phoenix-See South; Secondary: Barop / Kreuzviertel-West
        primary_center = (394_500.0, 5_704_200.0)
        primary_radius = 1_150.0
        secondary_center = (390_800.0, 5_706_800.0)
        secondary_radius = 1_000.0
        grid_nx, grid_ny = 12, 12
    elif region == "erie_county":
        csv_path = DATA_DIR / "erie_county_osm_buildings.csv.gz"
        region_label = "Erie County, NY"
        crs_label = "EPSG:26918 (UTM 18N)"
        # East-west service-district boundary across Buffalo / Cheektowaga / Lancaster
        boundary_params = (4_759_600.0, 620.0, 0.5)
        rural_neighbor_thresh = 60
        # Primary: Cheektowaga / Amherst border; Secondary: Depew / Lancaster corridor
        primary_center = (189_500.0, 4_761_500.0)
        primary_radius = 1_250.0
        secondary_center = (193_500.0, 4_756_500.0)
        secondary_radius = 1_150.0
        grid_nx, grid_ny = 14, 13
    else:
        raise ValueError(f"Unsupported region: {region}")

    raw = pd.read_csv(csv_path)

    # Region-aware OSM residential classification:
    # - Dortmund (Germany ALKIS-rich OSM): explicit residential tags + addressed building=yes
    # - Erie County (NY GIS building=yes import): exclude explicit commercial/industrial/civic/garage tags
    if region == "dortmund":
        is_res = raw["tag_source"].isin(["explicit_res", "yes_addr"]).to_numpy()
    else:
        is_res = (raw["tag_source"] != "explicit_non_res").to_numpy()

    x = raw["utm_easting"].to_numpy(dtype=float)
    y = raw["utm_northing"].to_numpy(dtype=float)
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    # Local 500 m residential density around every building via cKDTree
    res_xy = np.column_stack([x[is_res], y[is_res]])
    res_tree = cKDTree(res_xy)
    all_xy = np.column_stack([x, y])
    local_counts = np.asarray(res_tree.query_ball_point(all_xy, r=500.0, return_length=True), dtype=int)

    # Environmental service zones: North vs. South district for denser parcels, Rural_Fringe for low-density outskirts
    has_public_service = local_counts >= rural_neighbor_thresh
    boundary_y = eval_corridor_boundary(x, x_min, x_max, boundary_params)
    env_zone = np.where(
        ~has_public_service,
        "Rural_Fringe_Zone",
        np.where(y >= boundary_y, "North_Service_Zone", "South_Service_Zone"),
    )

    density_tier = np.where(
        local_counts < rural_neighbor_thresh,
        "rural",
        np.where(local_counts < 180, "suburban", "urban"),
    )

    # Metric census block-group grid
    dx = (x_max - x_min) / grid_nx
    dy = (y_max - y_min) / grid_ny
    gx = np.clip(((x - x_min) / dx).astype(int), 0, grid_nx - 1)
    gy = np.clip(((y - y_min) / dy).astype(int), 0, grid_ny - 1)
    block_group_id = gy * grid_nx + gx

    parcels = pd.DataFrame(
        {
            "parcel_id": raw["osm_id"].to_numpy(dtype=np.int64),
            "x": x,
            "y": y,
            "lat": raw["lat"].to_numpy(dtype=float),
            "lon": raw["lon"].to_numpy(dtype=float),
            "building_tag": raw["building_tag"].astype(str).to_numpy(),
            "is_residential": is_res,
            "has_public_water": has_public_service,
            "water_zone": env_zone,
            "local_res_500m": local_counts,
            "density_tier": density_tier,
            "block_group_id": block_group_id,
        }
    )

    # Block-group summary table with exact residential building density
    res_counts = (
        parcels.loc[parcels["is_residential"]]
        .groupby("block_group_id")
        .size()
        .to_dict()
    )
    bg_area_m2 = dx * dy
    bg_rows = []
    for gy_i in range(grid_ny):
        for gx_i in range(grid_nx):
            bg_id = gy_i * grid_nx + gx_i
            cnt = int(res_counts.get(bg_id, 0))
            bg_rows.append(
                {
                    "block_group_id": bg_id,
                    "gx": gx_i,
                    "gy": gy_i,
                    "cx": x_min + (gx_i + 0.5) * dx,
                    "cy": y_min + (gy_i + 0.5) * dy,
                    "res_count": cnt,
                    "res_density_per_m2": cnt / bg_area_m2,
                }
            )
    block_groups = pd.DataFrame(bg_rows)

    # Sample 160 study subjects (100 uniform background + 35 primary cluster + 25 secondary cluster)
    # Pure uniform random background sampling across verified residences matches Kulldorff's Poisson null
    res_df = parcels.loc[parcels["is_residential"]].copy()
    res_indices = res_df.index.to_numpy()
    bg_choice = rng.choice(res_indices, size=100, replace=False)

    remaining = res_df.drop(index=bg_choice)
    d_p = np.hypot(remaining["x"] - primary_center[0], remaining["y"] - primary_center[1])
    cand_p = remaining.loc[d_p <= primary_radius].index.to_numpy()
    if len(cand_p) < 35:
        cand_p = remaining.loc[d_p.nsmallest(max(80, 35)).index].index.to_numpy()
    c1_choice = rng.choice(cand_p, size=35, replace=False)

    remaining = remaining.drop(index=c1_choice)
    d_s = np.hypot(remaining["x"] - secondary_center[0], remaining["y"] - secondary_center[1])
    cand_s = remaining.loc[d_s <= secondary_radius].index.to_numpy()
    if len(cand_s) < 25:
        cand_s = remaining.loc[d_s.nsmallest(max(80, 25)).index].index.to_numpy()
    c2_choice = rng.choice(cand_s, size=25, replace=False)

    subj_idx = np.concatenate([bg_choice, c1_choice, c2_choice])
    subjects = parcels.loc[subj_idx].copy().reset_index(drop=True)
    subjects["subject_id"] = np.arange(len(subjects), dtype=int)
    subjects["group"] = (
        ["background"] * 100 + ["primary_cluster"] * 35 + ["secondary_cluster"] * 25
    )

    return StudyLandscape(
        region_key=region,
        region_label=region_label,
        crs_label=crs_label,
        parcels=parcels,
        subjects=subjects,
        block_groups=block_groups,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        grid_nx=grid_nx,
        grid_ny=grid_ny,
        boundary_y_fn_params=boundary_params,
    )
