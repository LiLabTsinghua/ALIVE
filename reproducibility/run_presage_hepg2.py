"""Run the full current PRESAGE InDistribution hepg2 reproducibility job."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PRESAGE_DIR = REPO_ROOT / "baseline" / "perturbation_generalization" / "PRESAGE"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PRESAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PRESAGE_DIR))

from alive.benchmark import build_benchmark_config
from in_distribution import alive_config, main


config = dict(alive_config)
config.update(
    {
        "data_dir": "/data/home/shengkun/benchmark/data",
        "results_dir": str(REPO_ROOT / "reproducibility" / "results"),
        "model_name": "PRESAGE",
        "task": "in_distribution",
        "budget_plan": [0.5, 1, 10, 30, 50, 80, 100],
        "start_mode": "random",
        "seed": 56,
        "repeat": 3,
    }
)

runtime_config = build_benchmark_config(
    config,
    enable_predefined_condition=True,
)
main(config=runtime_config)
