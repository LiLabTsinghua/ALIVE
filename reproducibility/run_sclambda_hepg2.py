"""Run the full current scLambda InDistribution HepG2 reproducibility job."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
METHOD_DIR = REPO_ROOT / "baseline" / "perturbation_generalization" / "scLambda"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from alive import InDistribution, set_seed
from alive.benchmark import (
    build_benchmark_config,
    load_benchmark_adata,
    run_benchmark,
)
from alive_adapter import build_model_capacity, build_model_wrapper


config = {
    "data_dir": "/data/home/shengkun/benchmark/data",
    "results_dir": str(REPO_ROOT / "reproducibility" / "sclambda_results"),
    "atlas": "essential_atlas",
    "dataset": "nadig_hepg2_essential",
    "cell_type": "hepg2",
    "model_name": "scLambda",
    "use_support_data": False,
    "task": "in_distribution",
    "strategy": "random_sampling",
    "budget_plan": [0.5, 1, 10, 30, 50, 80, 100],
    "start_mode": "random",
    "seed": 56,
    "repeat": 3,
    "model_config": {
        "embedding_path": "/home/shengkun/aivc/embedding/gene/text_based/genept/genept_1.pkl",
    },
}

runtime_config = build_benchmark_config(
    config,
    enable_predefined_condition=True,
)
run_benchmark(
    config=runtime_config,
    task_cls=InDistribution,
    adata_loader=load_benchmark_adata,
    model_capacity_builder=build_model_capacity,
    model_wrapper_factory=build_model_wrapper,
    seed_setter=set_seed,
)
