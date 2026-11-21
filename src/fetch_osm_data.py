"""Fetch real OpenStreetMap building centroids for Dortmund (DE) and Erie County (NY).

Uses the public Overpass API (`out center tags;`) to retrieve real building
footprints as centroids, classifies residential vs. non-residential land use from
OSM `building` tags, projects WGS84 (EPSG:4326) into exact metric UTM coordinates
(EPSG:25832 for Dortmund, EPSG:26918 for Erie County) using exact transverse
Mercator (Karney/Krueger series in pure NumPy), and saves compact `.csv.gz` files.

Data attribution: Map data (c) OpenStreetMap contributors, available under ODbL 1.0.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

# Explicit non-residential OSM building values (commercial, industrial, civic, garages, sheds)
NON_RESIDENTIAL_TAGS = {
    "commercial", "industrial", "retail", "warehouse", "office", "supermarket",
    "kiosk", "service", "garage", "garages", "carport", "shed", "roof",
    "greenhouse", "barn", "farm_auxiliary", "stable", "silo", "hangar",
    "school", "university", "college", "kindergarten", "hospital", "clinic",
    "church", "cathedral", "chapel", "mosque", "synagogue", "temple",
    "public", "civic", "government", "fire_station", "police", "train_station",
    "transportation", "bridge", "bunker", "ruins", "construction", "sports_hall",
    "stadium", "grandstand", "pavilion", "toilets", "transformer_tower",
    "water_tower", "substation", "storage_tank", "parking", "hotel", "motel",
    "allotment_house", "hut", "cabin", "cowshed", "digester", "slurry_tank",
}

EXPLICIT_RESIDENTIAL_TAGS = {
    "residential", "house", "apartments", "detached", "semidetached_house",
    "terrace", "bungalow", "dormitory", "static_caravan", "farm", "duplex",
    "townhouse", "condominium", "villa",
}


def wgs84_to_utm(lat_deg: np.ndarray, lon_deg: np.ndarray, zone: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert WGS84 lat/lon (degrees) to UTM Easting/Northing (meters, northern hemisphere).

    Uses standard WGS84 ellipsoid parameters (a = 6378137.0 m, f = 1/298.257223563)
    with 4th-order transverse Mercator series (sub-millimeter accuracy within UTM zone).
    """
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    ep2 = e2 / (1.0 - e2)
    k0 = 0.9996

    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    tan_lat = np.tan(lat)

    n = a / np.sqrt(1.0 - e2 * sin_lat**2)
    t = tan_lat**2
    c = ep2 * cos_lat**2
    aa = cos_lat * (lon - lon0)

    m = a * (
        (1.0 - e2 / 4.0 - 3.0 * e2**2 / 64.0 - 5.0 * e2**3 / 256.0) * lat
        - (3.0 * e2 / 8.0 + 3.0 * e2**2 / 32.0 + 45.0 * e2**3 / 1024.0) * np.sin(2.0 * lat)
        + (15.0 * e2**2 / 256.0 + 45.0 * e2**3 / 1024.0) * np.sin(4.0 * lat)
        - (35.0 * e2**3 / 3072.0) * np.sin(6.0 * lat)
    )

    easting = 500000.0 + k0 * n * (
        aa
        + (1.0 - t + c) * aa**3 / 6.0
        + (5.0 - 18.0 * t + t**2 + 72.0 * c - 58.0 * ep2) * aa**5 / 120.0
    )
    northing = k0 * (
        m
        + n * tan_lat * (
            aa**2 / 2.0
            + (5.0 - t + 9.0 * c + 4.0 * c**2) * aa**4 / 24.0
            + (61.0 - 58.0 * t + t**2 + 600.0 * c - 330.0 * ep2) * aa**6 / 720.0
        )
    )
    return easting, northing


def _query_overpass(ql: str, timeout_s: int = 120) -> dict:
    payload = urllib.parse.urlencode({"data": ql}).encode("utf-8")
    last_err: Exception | None = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            print(f"Querying {endpoint} ...", flush=True)
            req = urllib.request.Request(
                endpoint,
                data=payload,
                headers={"User-Agent": "TU-Dortmund-Geoprivacy-Benchmark/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8")
            return json.loads(raw)
        except Exception as exc:
            print(f"  Endpoint failed ({exc}); trying next...", flush=True)
            last_err = exc
            time.sleep(2.0)
    raise RuntimeError(f"All Overpass endpoints failed: {last_err}")


def fetch_region(
    name: str,
    bbox: tuple[float, float, float, float],
    utm_zone: int,
    out_path: Path,
    *,
    treat_yes_as_residential: bool = False,
) -> pd.DataFrame:
    """Fetch OSM building centroids inside (south, west, north, east) bbox."""
    s, w, n, e = bbox
    ql = (
        f"[out:json][timeout:90];\n"
        f"(\n"
        f'  way["building"]({s},{w},{n},{e});\n'
        f");\n"
        f"out center tags;\n"
    )
    data = _query_overpass(ql)
    elements = data.get("elements", [])
    print(f"[{name}] Retrieved {len(elements)} raw OSM building ways.", flush=True)

    rows = []
    for el in elements:
        center = el.get("center")
        if not center:
            continue
        tags = el.get("tags", {})
        btag = str(tags.get("building", "yes")).strip().lower()
        amenity = str(tags.get("amenity", "")).strip().lower()
        shop = str(tags.get("shop", "")).strip().lower()
        landuse = str(tags.get("landuse", "")).strip().lower()

        if btag in EXPLICIT_RESIDENTIAL_TAGS:
            is_res = True
            tag_source = "explicit_res"
        elif btag in NON_RESIDENTIAL_TAGS or amenity or shop or landuse in {"industrial", "commercial", "retail"}:
            is_res = False
            tag_source = "explicit_non_res"
        elif btag == "yes":
            # In some US counties (e.g., Erie County GIS import), residential buildings carry building=yes + addr:housenumber
            has_addr = "addr:housenumber" in tags or "addr:street" in tags
            is_res = True if (treat_yes_as_residential and has_addr) else (False if not treat_yes_as_residential else True)
            tag_source = "yes_addr" if has_addr else "yes_generic"
        else:
            is_res = False
            tag_source = "other"

        rows.append(
            {
                "osm_id": int(el["id"]),
                "lat": float(center["lat"]),
                "lon": float(center["lon"]),
                "building_tag": btag,
                "has_addr": "addr:housenumber" in tags,
                "is_residential": bool(is_res),
                "tag_source": tag_source,
            }
        )

    df = pd.DataFrame(rows)
    easting, northing = wgs84_to_utm(df["lat"].to_numpy(), df["lon"].to_numpy(), utm_zone)
    df["utm_easting"] = np.round(easting, 2)
    df["utm_northing"] = np.round(northing, 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, compression="gzip")

    n_res = int(df["is_residential"].sum())
    print(
        f"[{name}] Saved {len(df)} centroids -> {out_path.name} "
        f"(residential={n_res}, non_res={len(df) - n_res}, "
        f"top tags={df['building_tag'].value_counts().head(6).to_dict()})",
        flush=True,
    )
    return df


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dortmund, Germany (EPSG:25832 / UTM 32N)
    # Spans Dortmund Innenstadt, Kreuzviertel, TU Dortmund, Phoenix-See, and southern rural outskirts (Syburg/Bittermark)
    fetch_region(
        name="Dortmund, Germany",
        bbox=(51.425, 7.390, 51.535, 7.555),
        utm_zone=32,
        out_path=DATA_DIR / "dortmund_osm_buildings.csv.gz",
        treat_yes_as_residential=False,
    )
    time.sleep(2.0)

    # 2. Erie County, New York (EPSG:26918 / UTM 18N; Richter 2018 study county)
    # Spans Buffalo urban core, Cheektowaga/Amherst suburbs, and sparse eastern rural fringe (Lancaster/Clarence)
    fetch_region(
        name="Erie County, NY",
        bbox=(42.860, -78.880, 42.980, -78.700),
        utm_zone=18,
        out_path=DATA_DIR / "erie_county_osm_buildings.csv.gz",
        treat_yes_as_residential=True,
    )


if __name__ == "__main__":
    main()
