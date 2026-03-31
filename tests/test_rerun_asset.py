from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cage.cli import main
from cage.helix_stl import export_helix_edges_to_stl
from cage.models import RowGeometry
from cage.rerun_asset import log_helix_edges_to_rerun, log_stl_asset


SIMPLE_STL = """solid triangle
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 0 1 0
  endloop
endfacet
endsolid triangle
"""


class RerunAssetTest(unittest.TestCase):
    def test_log_stl_asset_reads_stl_bytes_and_can_save_rrd(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            stl_path = temp_path / "triangle.stl"
            save_path = temp_path / "triangle.rrd"
            stl_path.write_text(SIMPLE_STL, encoding="ascii")

            with (
                patch("cage.rerun_asset.rr.init") as rr_init,
                patch("cage.rerun_asset.rr.save") as rr_save,
                patch("cage.rerun_asset.rr.log") as rr_log,
                patch("cage.rerun_asset.rr.disconnect") as rr_disconnect,
            ):
                log_stl_asset(
                    stl_path=stl_path,
                    app_id="cage-test",
                    spawn=False,
                    save_path=save_path,
                )

                rr_init.assert_called_once_with("cage-test", spawn=False)
                rr_save.assert_called_once_with(save_path)
                self.assertEqual(rr_log.call_count, 2)
                rr_disconnect.assert_called_once()


class HelixStlExportTest(unittest.TestCase):
    def test_export_helix_edges_to_stl_writes_ascii_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "helix.stl"
            summary = export_helix_edges_to_stl(
                [
                    (
                        np.array([0.1, 0.2, 0.3], dtype=float),
                        np.array([0.9, 0.2, 0.3], dtype=float),
                    )
                ],
                radius=0.05,
                output_path=output_path,
            )

            content = output_path.read_text(encoding="ascii")
            self.assertTrue(content.startswith("solid cage_helix"))
            self.assertIn("facet normal", content)
            self.assertEqual(summary.edge_count, 1)
            self.assertGreater(summary.triangle_count, 0)


class RerunHelixTest(unittest.TestCase):
    def test_log_helix_edges_to_rerun_uses_temp_stl_when_no_output_is_requested(
        self,
    ) -> None:
        with (
            patch("cage.rerun_asset.export_helix_edges_to_stl") as export_mock,
            patch("cage.rerun_asset.log_stl_asset") as log_mock,
        ):
            export_mock.return_value = type(
                "Summary",
                (),
                {"triangle_count": 42, "edge_count": 7},
            )()
            summary = log_helix_edges_to_rerun(
                edges=[(np.zeros(3), np.ones(3))],
                radius=0.02,
                app_id="cage-test",
                spawn=False,
                save_path=Path("/tmp/test.rrd"),
                stl_path=None,
            )

            export_mock.assert_called_once()
            log_mock.assert_called_once()
            self.assertEqual(summary.triangle_count, 42)
            self.assertEqual(summary.edge_count, 7)
            self.assertIsNone(summary.stl_path)
            self.assertEqual(summary.rrd_path, Path("/tmp/test.rrd"))


class RerunStlCliTest(unittest.TestCase):
    def test_rerun_stl_command_routes_to_asset_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            stl_path = temp_path / "triangle.stl"
            save_path = temp_path / "triangle.rrd"
            stl_path.write_text(SIMPLE_STL, encoding="ascii")

            with patch("cage.cli.log_stl_asset") as log_stl_asset_mock:
                main(
                    [
                        "rerun-stl",
                        str(stl_path),
                        "--rerun-app-id",
                        "cage-test",
                        "--save-rrd",
                        str(save_path),
                        "--no-rerun-spawn",
                    ]
                )

                log_stl_asset_mock.assert_called_once_with(
                    stl_path=stl_path,
                    app_id="cage-test",
                    spawn=False,
                    save_path=save_path,
                )


class RerunHelixCliTest(unittest.TestCase):
    def test_rerun_helix_command_routes_to_helix_loader(self) -> None:
        row = RowGeometry(
            rng_seed=55,
            seeds=np.array([[0.5, 0.5, 0.5]], dtype=float),
            cells=[],
            halfspace_sets=[],
            edges=[(np.zeros(3), np.ones(3))],
        )

        with (
            patch("cage.cli.VoronoiPipeline") as pipeline_cls,
            patch("cage.cli.log_helix_edges_to_rerun") as helix_mock,
        ):
            pipeline_cls.return_value.build_row.return_value = row
            helix_mock.return_value = type(
                "Summary",
                (),
                {
                    "edge_count": 1,
                    "triangle_count": 24,
                    "stl_path": Path("/tmp/model.stl"),
                    "rrd_path": Path("/tmp/model.rrd"),
                },
            )()

            main(
                [
                    "rerun-helix",
                    "--seed",
                    "55",
                    "--num-seeds",
                    "8",
                    "--radius",
                    "0.02",
                    "--stl-output",
                    "/tmp/model.stl",
                    "--save-rrd",
                    "/tmp/model.rrd",
                    "--rerun-app-id",
                    "cage-test",
                    "--no-rerun-spawn",
                ]
            )

            pipeline_cls.return_value.build_row.assert_called_once_with(
                num_seeds=8, rng_seed=55
            )
            helix_mock.assert_called_once_with(
                row.edges,
                radius=0.02,
                app_id="cage-test",
                spawn=False,
                save_path=Path("/tmp/model.rrd"),
                stl_path=Path("/tmp/model.stl"),
            )


if __name__ == "__main__":
    unittest.main()
