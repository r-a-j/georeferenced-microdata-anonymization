"""Unit tests for geographic masking invariants on real OpenStreetMap UTM datasets."""

from __future__ import annotations

import unittest
import numpy as np

from src.landscape import load_osm_landscape
from src.geomasking import (
    mask_donut_geomasking,
    mask_random_perturbation,
    mask_verified_neighbor,
)
from src.metrics import kulldorff_poisson_scan, ripley_l_centered


class TestGeomaskingInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dortmund = load_osm_landscape("dortmund", seed=42)
        cls.do_xy = cls.dortmund.subjects[["x", "y"]].to_numpy()
        cls.erie = load_osm_landscape("erie_county", seed=42)
        cls.erie_xy = cls.erie.subjects[["x", "y"]].to_numpy()

    def test_osm_datasets_loaded_with_expected_utm_scales(self) -> None:
        self.assertGreater(len(self.dortmund.parcels), 100_000)
        self.assertGreater(int(self.dortmund.parcels["is_residential"].sum()), 50_000)
        self.assertGreater(len(self.erie.parcels), 50_000)
        # Metric UTM ranges
        self.assertTrue(300_000 < self.dortmund.x_min < 500_000)
        self.assertTrue(5_600_000 < self.dortmund.y_min < 5_800_000)
        self.assertTrue(100_000 < self.erie.x_min < 300_000)
        self.assertTrue(4_700_000 < self.erie.y_min < 4_800_000)

    def test_verified_neighbor_guarantees_k_and_exclusion_dortmund(self) -> None:
        res = mask_verified_neighbor(
            self.dortmund, k=50, exclusion_m=50.0, env_match=False, n_replicates=10, seed=7
        )
        self.assertTrue(np.all(res.realized_k == 50))
        self.assertTrue(np.all(res.on_residential_parcel))

        diff = res.surrogates - self.do_xy[:, None, :]
        dists = np.hypot(diff[:, :, 0], diff[:, :, 1])
        self.assertTrue(np.all(dists > 50.0))

    def test_verified_neighbor_env_matching_preserves_100_pct_both_regions(self) -> None:
        for ls in (self.dortmund, self.erie):
            res = mask_verified_neighbor(
                ls, k=100, exclusion_m=50.0, env_match=True, n_replicates=10, seed=8
            )
            self.assertTrue(np.all(res.env_preserved))
            self.assertTrue(np.all(res.realized_k == 100))
            self.assertTrue(np.all(res.on_residential_parcel))

    def test_random_perturbation_respects_annulus_bounds(self) -> None:
        res = mask_random_perturbation(
            self.dortmund, radius_m=905.0, exclusion_m=50.0, n_replicates=15, seed=9
        )
        diff = res.surrogates - self.do_xy[:, None, :]
        dists = np.hypot(diff[:, :, 0], diff[:, :, 1])
        self.assertLessEqual(float(dists.max()), 905.0 + 1e-6)
        self.assertGreater(float(np.median(dists)), 500.0)

    def test_donut_geomasking_runs_and_returns_valid_shapes(self) -> None:
        res = mask_donut_geomasking(
            self.dortmund, k_min=100, exclusion_m=50.0, env_match=True, n_replicates=5, seed=10
        )
        self.assertEqual(res.surrogates.shape, (len(self.do_xy), 5, 2))
        self.assertEqual(res.realized_k.shape, (len(self.do_xy),))

    def test_kulldorff_scan_detects_embedded_hotspot_both_regions(self) -> None:
        for ls, xy in ((self.dortmund, self.do_xy), (self.erie, self.erie_xy)):
            scan = kulldorff_poisson_scan(xy, ls)
            self.assertGreater(scan.llr, 10.0)
            self.assertGreaterEqual(len(scan.primary_block_groups), 2)

    def test_ripley_l_positive_for_clustered_subjects(self) -> None:
        radii = np.array([500.0, 1000.0, 1500.0])
        area_m2 = (self.dortmund.x_max - self.dortmund.x_min) * (
            self.dortmund.y_max - self.dortmund.y_min
        )
        l_vals = ripley_l_centered(self.do_xy, radii, area_m2)
        self.assertTrue(np.all(l_vals > 0.0))


if __name__ == "__main__":
    unittest.main()
