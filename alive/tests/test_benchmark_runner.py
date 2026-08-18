import json
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from alive.benchmark import (
    build_runtime_benchmark_config,
    run_benchmark,
    save_runtime_metric_summary,
)


def _make_fixture():
    obs = pd.DataFrame(
        {
            "dataset_cell_type": ["target_ct", "target_ct", "target_ct"],
            "split": ["control", "pool", "val"],
            "perturbation": ["control", "gene_a", "gene_b"],
        },
        index=["cell_0", "cell_1", "cell_2"],
    )
    var = pd.DataFrame(index=["g1", "g2"])
    X = np.array([[0.0, 0.0], [1.0, 0.5], [0.5, 1.0]], dtype=float)
    return ad.AnnData(X=X, obs=obs, var=var)


class _DummyTask:
    def __init__(self, adata, model_capacity, model_wrapper, config):
        self.adata = adata
        self.model_capacity = model_capacity
        self.model_wrapper = model_wrapper
        self.config = config
        self.run_calls = []

    def run(self, repeat):
        self.run_calls.append(repeat)
        return {
            "runtime_metric_rows": [
                {
                    "method": self.config["model_name"],
                    "context": self.config["dataset_cell_type"],
                    "repeat": repeat,
                    "x": 0.0,
                    "set": "test",
                    "metric_name": "correlation",
                    "gene": "all",
                    "value": float(repeat),
                }
            ]
        }


class BenchmarkRunnerTests(unittest.TestCase):
    def test_save_runtime_metric_summary_accepts_shuffled_field_order(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        shuffled_row = {
            "value": 0.5,
            "gene": "all",
            "metric_name": "mse",
            "set": "test",
            "x": 10.0,
            "repeat": 0,
            "context": "target_ct",
            "method": "mock-model",
        }
        output_path = save_runtime_metric_summary(temp_dir.name, [shuffled_row])
        written = pd.read_csv(output_path)

        self.assertEqual(
            written.columns.tolist(),
            ["method", "context", "repeat", "x", "set", "metric_name", "gene", "value"],
        )

    def test_save_runtime_metric_summary_rejects_missing_or_extra_fields(self):
        valid_row = {
            "method": "mock-model",
            "context": "target_ct",
            "repeat": 0,
            "x": 10.0,
            "set": "test",
            "metric_name": "mse",
            "gene": "all",
            "value": 0.5,
        }
        missing_field_row = dict(valid_row)
        missing_field_row.pop("gene")
        extra_field_row = {**valid_row, "unexpected": True}

        with self.assertRaises(ValueError):
            save_runtime_metric_summary("/tmp", [missing_field_row])
        with self.assertRaises(ValueError):
            save_runtime_metric_summary("/tmp", [extra_field_row])

    def test_build_runtime_benchmark_config_derives_standard_paths(self):
        config = build_runtime_benchmark_config(
            {
                "data_dir": "/tmp/data",
                "results_dir": "/tmp/results",
                "atlas": "essential_atlas",
                "dataset": "replogle_k562_essential",
                "cell_type": "k562",
                "model_name": "PRESAGE",
                "task": "active_learning",
                "strategy": "diversity_sampling",
                "start_mode": "zero-shot",
                "seed": 3,
                "repeat": 2,
                "budget_plan": [10.0],
                "use_support_data": False,
            }
        )

        self.assertEqual(config["atlas_dir"], "/tmp/data/essential_atlas")
        self.assertEqual(config["dataset_cell_type"], "replogle_k562_essential_k562")
        self.assertEqual(config["max_budget_gap"], 5.0)
        self.assertEqual(config["max_representative_candidates"], 5000)
        self.assertEqual(
            config["saved_dir"],
            "/tmp/results/active_learning/replogle_k562_essential_k562/PRESAGE/diversity_sampling_3_2",
        )

    def test_run_benchmark_saves_config_and_executes_repeats(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        tmp_path = Path(temp_dir.name)

        config = build_runtime_benchmark_config(
            {
                "data_dir": str(tmp_path / "data"),
                "results_dir": str(tmp_path / "results"),
                "atlas": "essential_atlas",
                "dataset": "mock_ds",
                "cell_type": "target",
                "model_name": "mock-model",
                "task": "active_learning",
                "strategy": "diversity_sampling",
                "start_mode": "zero-shot",
                "seed": 10,
                "repeat": 3,
                "budget_plan": [10.0],
                "use_support_data": False,
            }
        )

        task = run_benchmark(
            config=config,
            task_cls=_DummyTask,
            adata_loader=lambda cfg: _make_fixture(),
            model_capacity_builder=lambda adata, cfg: ["gene_a", "gene_b"],
            model_wrapper_factory=lambda cfg: type(
                "Wrapper",
                (),
                {
                    "config": dict(cfg),
                    "fit": lambda self, adata, split_key: None,
                    "predict": lambda self, adata, split_key, split: None,
                },
            )(),
        )

        config_path = Path(config["saved_dir"]) / "config.json"
        self.assertTrue(config_path.exists())
        written = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(written["model_name"], "mock-model")
        self.assertEqual(written["task"], "active_learning")
        self.assertEqual(task.run_calls, [0, 1, 2])
        self.assertEqual(task.model_capacity, ["gene_a", "gene_b"])
        self.assertEqual(task.model_wrapper.config["model_name"], "mock-model")

        achievement_path = Path(config["saved_dir"]) / "achievement_summary.csv"
        self.assertTrue(achievement_path.exists())
        achievement_df = pd.read_csv(achievement_path)
        self.assertEqual(
            achievement_df.columns.tolist(),
            ["method", "context", "repeat", "x", "set", "metric_name", "gene", "value"],
        )
        self.assertEqual(set(achievement_df["context"]), {"mock_ds_target"})
        self.assertEqual(achievement_df["repeat"].tolist(), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
