# OpenStreetMap Data Attribution & Processing Notes

The real-world building centroid datasets in this folder were retrieved from OpenStreetMap via the public Overpass API (`src/fetch_osm_data.py`).

- **License & Credit:** Map data © OpenStreetMap contributors, available under the [Open Database License (ODbL 1.0)](https://www.openstreetmap.org/copyright).
- **Datasets:**
  1. `dortmund_osm_buildings.csv.gz` (Dortmund, Germany; `51.425-51.535 deg N, 7.390-7.555 deg E`, projected to **EPSG:25832 / ETRS89 UTM Zone 32N**): `125,612` building centroids (`58,462` verified residential addresses via explicit residential tags or addressed `building=yes` without non-residential tags; `67,150` non-residential or auxiliary structures such as garages, allotment houses, commercial, industrial, and university buildings).
  2. `erie_county_osm_buildings.csv.gz` (Erie County, New York; `42.860-42.980 deg N, 78.880-78.700 deg W`, projected to **EPSG:26918 / NAD83 UTM Zone 18N**, mirroring Richter 2018's study county): `53,394` building centroids (`50,252` residential candidates excluding explicit commercial, industrial, civic, and garage tags; `3,142` non-residential structures).

All study subjects (`n = 160` per region) are randomly sampled building centroids with synthetic cluster assignments; **no real personal, medical, or household records are present in this repository**.
