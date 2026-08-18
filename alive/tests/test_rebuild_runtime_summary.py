import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from alive.tools.rebuild_runtime_summary import (
    discover_run_dirs,
    rebuild_runtime_summary,
    rebuild_runtime_summary_batch,
)


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_true_bulk(path: Path):
    true_df = pd.DataFrame(
        {
            "g1": [0.0, 1.0, 3.0],
            "g2": [0.0, 2.0, 2.0],
            "g3": [0.0, 3.0, 1.0],
        },
        index=["control", "gene_a", "gene_b"],
    )
    true_df.index.name = "perturbation"
    true_df.to_csv(path, compression="gzip")


def _write_true_bulk_with_positional_columns(path: Path):
    true_df = pd.DataFrame(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
        ],
        index=["control", "gene_a", "gene_b"],
        columns=["0", "1", "2"],
    )
    true_df.index.name = "perturbation"
    true_df.to_csv(path, compression="gzip")


def _write_pred_bulk(path: Path):
    pred_df = pd.DataFrame(
        {
            "g1": [3.0, 3.0],
            "g2": [2.0, 2.0],
            "g3": [1.0, 1.0],
        },
        index=["gene_a", "gene_b"],
    )
    pred_df.to_csv(path, compression="gzip")


def _write_mock_run(run_dir: Path, atlas_dir: Path, model_name: str):
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "config.json",
        {
            "atlas_dir": str(atlas_dir),
            "dataset_cell_type": "mock_ds_target",
            "model_name": model_name,
        },
    )
    _write_pred_bulk(run_dir / "repeat_0_50.0_test_pred_bulk.csv.gz")


class RebuildRuntimeSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.tmp_path = Path(self.temp_dir.name)

    def test_rebuild_runtime_summary_recomputes_metrics_from_saved_pred_bulk(self):
        atlas_dir = self.tmp_path / "atlas"
        atlas_dir.mkdir()
        _write_json(atlas_dir / "mock_ds_target_hvg.json", ["g1", "g3"])
        _write_json(
            atlas_dir / "mock_ds_target_de.json",
            {"gene_a": ["g1", "g2", "g3"], "gene_b": ["g1", "g2", "g3"]},
        )
        _write_true_bulk(atlas_dir / "mock_ds_target_true_bulk.csv.gz")

        run_dir = self.tmp_path / "results" / "mock"
        _write_mock_run(run_dir=run_dir, atlas_dir=atlas_dir, model_name="mock_method")
        _write_pred_bulk(run_dir / "repeat_0_50.0_cluster_test_pred_bulk.csv.gz")

        output_path = rebuild_runtime_summary(run_dir=run_dir)

        self.assertEqual(output_path, run_dir / "achievement_summary.csv")
        self.assertTrue(output_path.exists())

        written = pd.read_csv(output_path)
        self.assertEqual(
            written.columns.tolist(),
            ["method", "context", "repeat", "x", "set", "metric_name", "gene", "value"],
        )
        self.assertEqual(set(written["set"]), {"test", "cluster_test"})
        self.assertEqual(
            set(written["metric_name"]),
            {
                "mse",
                "mae",
                "correlation",
                "systema_correlation",
                "systema_bias",
            },
        )
        self.assertEqual(set(written["context"]), {"mock_ds_target"})
        self.assertEqual(set(written["method"]), {"mock_method"})
        self.assertEqual(set(written["repeat"]), {0})
        self.assertEqual(set(written["x"]), {50.0})

    def test_discover_run_dirs_filters_methods_and_ignores_nested_non_runs(self):
        atlas_dir = self.tmp_path / "atlas"
        atlas_dir.mkdir()
        _write_json(atlas_dir / "mock_ds_target_hvg.json", ["g1", "g3"])
        _write_json(
            atlas_dir / "mock_ds_target_de.json",
            {"gene_a": ["g1", "g2", "g3"], "gene_b": ["g1", "g2", "g3"]},
        )
        _write_true_bulk(atlas_dir / "mock_ds_target_true_bulk.csv.gz")

        root_dir = self.tmp_path / "results"
        gears_run = root_dir / "mock_context" / "GEARS" / "random_sampling_56_3"
        scgen_run = root_dir / "mock_context" / "scGen" / "random_sampling_56_3"
        _write_mock_run(run_dir=gears_run, atlas_dir=atlas_dir, model_name="GEARS")
        _write_mock_run(run_dir=scgen_run, atlas_dir=atlas_dir, model_name="scGen")

        nested_round_dir = scgen_run / "round_0"
        nested_round_dir.mkdir()
        _write_json(nested_round_dir / "config.json", {"ignored": True})

        discovered_all = discover_run_dirs(root_dir=root_dir)
        discovered_gears_only = discover_run_dirs(root_dir=root_dir, methods=["GEARS"])

        self.assertEqual(discovered_all, [gears_run, scgen_run])
        self.assertEqual(discovered_gears_only, [gears_run])

    def test_rebuild_runtime_summary_batch_writes_each_discovered_run(self):
        atlas_dir = self.tmp_path / "atlas"
        atlas_dir.mkdir()
        _write_json(atlas_dir / "mock_ds_target_hvg.json", ["g1", "g3"])
        _write_json(
            atlas_dir / "mock_ds_target_de.json",
            {"gene_a": ["g1", "g2", "g3"], "gene_b": ["g1", "g2", "g3"]},
        )
        _write_true_bulk(atlas_dir / "mock_ds_target_true_bulk.csv.gz")

        root_dir = self.tmp_path / "results"
        gears_run = root_dir / "mock_context" / "GEARS" / "random_sampling_56_3"
        scgen_run = root_dir / "mock_context" / "scGen" / "random_sampling_56_3"
        _write_mock_run(run_dir=gears_run, atlas_dir=atlas_dir, model_name="GEARS")
        _write_mock_run(run_dir=scgen_run, atlas_dir=atlas_dir, model_name="scGen")

        written_paths = rebuild_runtime_summary_batch(root_dir=root_dir, methods=["GEARS", "scGen"])

        self.assertEqual(
            written_paths,
            [
                gears_run / "achievement_summary.csv",
                scgen_run / "achievement_summary.csv",
            ],
        )
        self.assertTrue((gears_run / "achievement_summary.csv").exists())
        self.assertTrue((scgen_run / "achievement_summary.csv").exists())

    def test_rebuild_runtime_summary_aligns_positional_true_bulk_columns_to_pred_gene_names(self):
        atlas_dir = self.tmp_path / "atlas"
        atlas_dir.mkdir()
        _write_json(atlas_dir / "mock_ds_target_hvg.json", ["g1", "g3"])
        _write_json(
            atlas_dir / "mock_ds_target_de.json",
            {"gene_a": ["g1", "g2", "g3"], "gene_b": ["g1", "g2", "g3"]},
        )
        _write_true_bulk_with_positional_columns(atlas_dir / "mock_ds_target_true_bulk.csv.gz")

        run_dir = self.tmp_path / "results" / "mock_context" / "Biolord" / "random_sampling_56_3"
        _write_mock_run(run_dir=run_dir, atlas_dir=atlas_dir, model_name="Biolord")

        output_path = rebuild_runtime_summary(run_dir=run_dir)

        written = pd.read_csv(output_path)
        self.assertEqual(
            set(written["metric_name"]),
            {
                "mse",
                "mae",
                "correlation",
                "systema_correlation",
                "systema_bias",
            },
        )
        self.assertEqual(set(written["method"]), {"Biolord"})

    def test_rebuild_runtime_summary_can_resolve_stale_atlas_dir_from_run_path(self):
        benchmark_root = self.tmp_path / "benchmark"
        atlas_dir = benchmark_root / "data" / "essential_atlas"
        atlas_dir.mkdir(parents=True)
        _write_json(atlas_dir / "mock_ds_target_hvg.json", ["g1", "g3"])
        _write_json(
            atlas_dir / "mock_ds_target_de.json",
            {"gene_a": ["g1", "g2", "g3"], "gene_b": ["g1", "g2", "g3"]},
        )
        _write_true_bulk(atlas_dir / "mock_ds_target_true_bulk.csv.gz")

        run_dir = benchmark_root / "results" / "mock_context" / "scGPT" / "random_sampling_56_3"
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            run_dir / "config.json",
            {
                "atlas_dir": "/stale/server/path/essential_atlas",
                "data_dir": "/stale/server/path",
                "atlas": "essential_atlas",
                "dataset_cell_type": "mock_ds_target",
                "model_name": "scGPT",
            },
        )
        _write_pred_bulk(run_dir / "repeat_0_50.0_test_pred_bulk.csv.gz")

        output_path = rebuild_runtime_summary(run_dir=run_dir)
        written = pd.read_csv(output_path)
        self.assertEqual(
            set(written["metric_name"]),
            {
                "mse",
                "mae",
                "correlation",
                "systema_correlation",
                "systema_bias",
            },
        )
        self.assertEqual(set(written["method"]), {"scGPT"})


if __name__ == "__main__":
    unittest.main()
