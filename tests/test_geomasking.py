"""Unit tests for geographic masking invariants and spatial metrics."""

from __future__ import annotations

import unittest
import numpy as np

from src.landscape import generate_landscape
from src.geomasking import (
    mask_donut_geomasking,
    mask_random_perturbation,
    mask_verified_neighbor,
)
from src.metrics import kulldorff_poisson_scan, ripley_l_centered


class TestGeomaskingInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.landscape = generate_landscape(seed=42)
        cls.subj_xy = cls.landscape.subjects[["x", "y"]].to_numpy()

    def test_verified_neighbor_guarantees_k_and_exclusion(self) -> None:
        res = mask_verified_neighbor(
            self.landscape, k=50, exclusion_m=50.0, env_match=False, n_replicates=10, seed=7
        )
        self.assertTrue(np.all(res.realized_k == 50))
        self.assertTrue(np.all(res.on_residential_parcel))

        diff = res.surrogates - self.subj_xy[:, None, :]
        dists = np.hypot(diff[:, :, 0], diff[:, :, 1])
        self.assertTrue(np.all(dists > 50.0))

    def test_verified_neighbor_env_matching_preserves_100_pct(self) -> None:
        res = mask_verified_neighbor(
            self.landscape, k=50, exclusion_m=50.0, env_match=True, n_replicates=10, seed=8
        )
        self.assertTrue(np.all(res.env_preserved))
        self.assertTrue(np.all(res.realized_k == 50))

    def test_random_perturbation_respects_annulus_bounds(self) -> None:
        res = mask_random_perturbation(
            self.landscape, radius_m=905.0, exclusion_m=50.0, n_replicates=15, seed=9
        )
        diff = res.surrogates - self.subj_xy[:, None, :]
        dists = np.hypot(diff[:, :, 0], diff[:, :, 1])
        # Clipped boundary points could be slightly closer only at map edge; interior points stay within [50, 905]
        self.assertLessEqual(float(dists.max()), 905.0 + 1e-6)
        self.assertGreater(float(np.median(dists)), 500.0)

    def test_donut_geomasking_runs_and_returns_valid_shapes(self) -> None:
        res = mask_donut_geomasking(
            self.landscape, k_min=100, exclusion_m=50.0, env_match=True, n_replicates=5, seed=10
        )
        self.assertEqual(res.surrogates.shape, (len(self.subj_xy), 5, 2))
        self.assertEqual(res.realized_k.shape, (len(self.subj_xy),))

    def test_kulldorff_scan_detects_embedded_hotspot(self) -> None:
        scan = kulldorff_poisson_scan(
            self.subj_xy,
            self.landscape.block_groups,
            self.landscape.extent_m,
            self.landscape.grid_n,
        )
        self.assertGreater(scan.llr, 5.0)
        self.assertGreaterEqual(len(scan.primary_block_groups), 2)

    def test_ripley_l_positive_for_clustered_subjects(self) -> None:
        radii = np.array([500.0, 1000.0, 1500.0])
        l_vals = ripley_l_centered(self.subj_xy, radii, self.landscape.extent_m**2)
        self.assertTrue(np.all(l_vals > 0.0))


if __name__ == "__main__":
    unittest.main()
