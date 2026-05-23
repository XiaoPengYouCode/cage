from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.build_fjw_golden_case import build_minimal_mma_golden


class FJWGoldenCasesTest(unittest.TestCase):
    def test_build_minimal_mma_golden_case(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = build_minimal_mma_golden(Path(temp_dir) / "minimal_mma")
            fixture = np.load(output_dir / "mma_one_step.npz")

            self.assertTrue((output_dir / "manifest.json").exists())
            np.testing.assert_allclose(
                fixture["xmma"],
                np.array([0.42677313265564956, 0.001001091740939041, 0.6119509389187682], dtype=np.float64),
                rtol=1e-7,
                atol=1e-9,
            )
            self.assertGreater(float(fixture["delta"]), 0.0)


if __name__ == "__main__":
    unittest.main()
