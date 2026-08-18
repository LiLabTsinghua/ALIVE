"""Shared benchmark orchestration utilities used by baseline entry scripts."""

import json
from collections.abc import MutableMapping
from pathlib import Path

import pandas as pd

from .base import TASK_ALLOWED_STRATEGIES
from .utils import set_seed

SUPPORTED_TASKS = {"in_distribution", "out_of_distribution", "active_learning"}

__all__ = [
    "build_runtime_benchmark_config",
    "save_benchmark_config",
    "enable_default_predefined_condition_path",
    "enable_default_cluster_condition_path",
    "build_benchmark_config",
    "load_benchmark_adata",
    "save_runtime_metric_summary",
    "run_benchmark",
]


def _validate_benchmark_config(config):
    """Validate the shared config contract before paths or models are created.

    Args:
        config: Benchmark configuration mapping to validate.

    Raises:
        KeyError: If a required field is missing.
        TypeError: If a field has an invalid type.
        ValueError: If task, strategy, repeat, or budget values are unsupported.
    """
    if "objection" in config:
        raise ValueError("config['objection'] is not supported; use config['task'].")

    required_fields = [
        "data_dir",
        "results_dir",
        "atlas",
        "dataset",
        "cell_type",
        "model_name",
        "task",
        "strategy",
        "start_mode",
        "seed",
        "repeat",
        "budget_plan",
        "use_support_data",
    ]
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        raise KeyError(f"Missing required benchmark config fields: {missing_fields}")

    string_fields = [
        "data_dir",
        "results_dir",
        "atlas",
        "dataset",
        "cell_type",
        "model_name",
        "task",
        "strategy",
        "start_mode",
    ]
    invalid_string_fields = [
        field for field in string_fields if not isinstance(config[field], str) or not config[field]
    ]
    if invalid_string_fields:
        raise TypeError(
            "Benchmark config fields must be non-empty strings: "
            f"{invalid_string_fields}"
        )

    if config["task"] not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported task: {config['task']!r}. Allowed: {sorted(SUPPORTED_TASKS)}"
        )

    if config["task"] == "in_distribution":
        allowed_strategies = set(TASK_ALLOWED_STRATEGIES["in_distribution"])
    elif config["task"] == "out_of_distribution":
        allowed_strategies = set(TASK_ALLOWED_STRATEGIES["out_of_distribution"])
    elif config["task"] == "active_learning":
        allowed_strategies = set(TASK_ALLOWED_STRATEGIES["active_learning"])
    else:
        raise ValueError(f"Unexpected validated task: {config['task']!r}")

    if config["strategy"] not in allowed_strategies:
        raise ValueError(
            f"Unsupported strategy for task {config['task']!r}: "
            f"{config['strategy']!r}. Allowed: {sorted(allowed_strategies)}"
        )

    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int):
        raise TypeError("config['seed'] must be an integer.")

    if isinstance(config["repeat"], bool) or not isinstance(config["repeat"], int):
        raise TypeError("config['repeat'] must be a positive integer.")
    if config["repeat"] <= 0:
        raise ValueError("config['repeat'] must be positive.")

    if not isinstance(config["use_support_data"], bool):
        raise TypeError("config['use_support_data'] must be a boolean.")

    if not isinstance(config["budget_plan"], list):
        raise TypeError("config['budget_plan'] must be a list.")
    if len(config["budget_plan"]) == 0:
        raise ValueError("config['budget_plan'] cannot be empty.")
    invalid_budget_indices = [
        index
        for index, budget in enumerate(config["budget_plan"])
        if isinstance(budget, bool) or not isinstance(budget, (int, float))
    ]
    if invalid_budget_indices:
        raise TypeError(
            "config['budget_plan'] values must be numeric and cannot be booleans. "
            f"Invalid indices: {invalid_budget_indices}."
        )


def _validate_model_wrapper(model_wrapper):
    """Validate the adapter interface required by shared orchestration.

    Args:
        model_wrapper: Adapter expected to expose ``fit``, ``predict``, and config.

    Raises:
        TypeError: If methods are missing or ``config`` is not mutable.
    """
    missing_methods = [
        method_name
        for method_name in ("fit", "predict")
        if not callable(getattr(model_wrapper, method_name, None))
    ]
    if missing_methods:
        raise TypeError(f"Model wrapper is missing callable methods: {missing_methods}")

    if not isinstance(getattr(model_wrapper, "config", None), MutableMapping):
        raise TypeError("Model wrapper must expose a mutable mapping as 'config'.")


def build_runtime_benchmark_config(config):
    """Expand a script-owned config into the standard runtime benchmark layout.

    The input config is expected to already carry task-specific defaults such as
    ``task``, ``strategy``, and ``start_mode``. This helper only derives the
    canonical filesystem-oriented runtime fields shared by every baseline.

    Args:
        config: Script-owned config with at least ``data_dir``, ``atlas``,
            ``dataset``, ``cell_type``, ``results_dir``, ``task``,
            ``model_name``, ``strategy``, ``seed``, and ``repeat``.

    Returns:
        dict: A copy of ``config`` with ``atlas_dir``, ``dataset_cell_type``,
        and ``saved_dir`` added.

    Raises:
        KeyError: If a required configuration field is missing.
        TypeError: If a configuration field has an invalid type.
        ValueError: If a configuration value violates the runtime contract.
    """
    runtime_config = dict(config)
    _validate_benchmark_config(runtime_config)
    runtime_config.setdefault("max_budget_gap", 5.0)
    runtime_config.setdefault("max_representative_candidates", 5000)
    runtime_config["atlas_dir"] = f'{runtime_config["data_dir"]}/{runtime_config["atlas"]}'
    runtime_config["dataset_cell_type"] = (
        f'{runtime_config["dataset"]}_{runtime_config["cell_type"]}'
    )
    runtime_config["saved_dir"] = (
        f'{runtime_config["results_dir"]}/{runtime_config["task"]}/'
        f'{runtime_config["dataset_cell_type"]}/{runtime_config["model_name"]}/'
        f'{runtime_config["strategy"]}_{runtime_config["seed"]}_{runtime_config["repeat"]}'
    )
    return runtime_config


def save_benchmark_config(config):
    """Persist the resolved runtime config under the benchmark output directory.

    Args:
        config: Resolved runtime config with a ``saved_dir`` entry.

    Returns:
        pathlib.Path: Path to the written ``config.json``.
    """
    saved_dir = Path(config["saved_dir"])
    saved_dir.mkdir(parents=True, exist_ok=True)
    config_path = saved_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=4, ensure_ascii=False)
    print(f"✅ Configuration saved to: {config_path}")
    return config_path


def enable_default_predefined_condition_path(config):
    """Attach the default predefined-condition file for data-scaling tasks.

    The path is only injected when the runtime config does not already provide
    either an inline predefined condition payload or an explicit file path. The
    split files live under the atlas split tree, named after the task, so each
    task's schedule can coexist.

    Args:
        config: Runtime config carrying ``task``, ``atlas_dir``, and
            ``dataset_cell_type``.

    Returns:
        dict: A copy of ``config`` with ``predefined_condition_path`` added when
        applicable.

    Raises:
        ValueError: If ``task`` is not ``in_distribution`` or
            ``out_of_distribution``.
    """
    runtime_config = dict(config)
    if (
        "predefined_condition" not in runtime_config
        and "predefined_condition_path" not in runtime_config
    ):
        task = runtime_config.get("task")
        dataset_cell_type = runtime_config["dataset_cell_type"]

        if task == "in_distribution":
            runtime_config["predefined_condition_path"] = (
                f'{runtime_config["atlas_dir"]}/split/in_distribution/{dataset_cell_type}_cond.json'
            )
        if task == "out_of_distribution":
            runtime_config["predefined_condition_path"] = (
                f'{runtime_config["atlas_dir"]}/split/out_of_distribution/{dataset_cell_type}_cond.json'
            )
        if task not in {"in_distribution", "out_of_distribution"}:
            raise ValueError(
                "Default predefined-condition path is supported only for "
                f"'in_distribution' and 'out_of_distribution'. Got {task!r}."
            )
    return runtime_config


def enable_default_cluster_condition_path(config):
    """Attach the default fixed cluster-holdout file for out-of-distribution.

    Args:
        config: Runtime config carrying ``task``, ``atlas_dir``, and
            ``dataset_cell_type``.

    Returns:
        dict: A copy of ``config`` with ``cluster_condition_path`` added when the
        task is ``out_of_distribution`` and no cluster condition is set.
    """
    runtime_config = dict(config)
    if runtime_config.get("task") != "out_of_distribution":
        return runtime_config

    if (
        "cluster_condition" not in runtime_config
        and "cluster_condition_path" not in runtime_config
    ):
        runtime_config["cluster_condition_path"] = (
            f'{runtime_config["atlas_dir"]}/split/out_of_distribution/'
            f'{runtime_config["dataset_cell_type"]}_cluster_cond.json'
        )

    return runtime_config


def build_benchmark_config(default_config, enable_predefined_condition=True):
    """Build the runtime config used by the entry scripts.

    Baseline scripts own the full default config and explicitly decide whether
    predefined-condition behavior should be enabled. When disabled, any
    predefined-condition fields already present in the script-owned config are
    removed so active tasks keep their semantics stable.

    Args:
        default_config: Script-owned default config.
        enable_predefined_condition: Whether to attach default predefined- and
            cluster-condition paths.

    Returns:
        dict: The resolved runtime config.
    """
    config = dict(default_config)
    runtime_config = build_runtime_benchmark_config(config)
    if enable_predefined_condition:
        runtime_config = enable_default_predefined_condition_path(runtime_config)
        runtime_config = enable_default_cluster_condition_path(runtime_config)
    else:
        # Active tasks must not inherit any predefined-condition scheduling.
        runtime_config.pop("predefined_condition_path", None)
        runtime_config.pop("predefined_condition", None)
    return runtime_config


def load_benchmark_adata(config):
    """Load the default atlas AnnData used by most baseline entry scripts.

    Args:
        config: Runtime config carrying ``atlas_dir`` and ``atlas``.

    Returns:
        anndata.AnnData: The loaded atlas.
    """
    import scanpy as sc

    return sc.read_h5ad(f"{config['atlas_dir']}/{config['atlas']}.h5ad")


def save_runtime_metric_summary(saved_dir, rows):
    """Persist one long-format runtime metric summary after all repeats finish.

    Args:
        saved_dir: Benchmark run output directory.
        rows: Long-format metric rows; must match the expected column schema.

    Returns:
        pathlib.Path | None: Path to ``achievement_summary.csv``, or ``None`` if
        ``rows`` is empty.

    Raises:
        ValueError: If the row columns do not match the expected schema.
    """
    if not rows:
        return None

    expected_columns = ["method", "context", "repeat", "x", "set", "metric_name", "gene", "value"]
    expected_column_set = set(expected_columns)
    invalid_row_indices = [
        row_index
        for row_index, row in enumerate(rows)
        if set(row) != expected_column_set
    ]
    if invalid_row_indices:
        raise ValueError(
            "Every runtime metric summary row must contain exactly the columns "
            f"{expected_columns}. Invalid row indices: {invalid_row_indices}."
        )

    summary_df = pd.DataFrame(rows, columns=expected_columns)
    summary_df = summary_df.sort_values(
        ["method", "context", "repeat", "x", "set", "metric_name", "gene"],
    ).reset_index(drop=True)
    output_path = Path(saved_dir) / "achievement_summary.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"✅ Runtime metric summary saved to: {output_path}")
    return output_path


def run_benchmark(
    config,
    task_cls,
    adata_loader,
    model_capacity_builder,
    model_wrapper_factory,
    seed_setter=set_seed,
):
    """Run a benchmark task with shared loading, saving, and repeat orchestration.

    The caller is responsible for providing a fully resolved runtime config and
    the method-specific loader, capacity builder, and wrapper factory.

    Args:
        config: Resolved runtime config (must include ``repeat`` and ``seed``).
        task_cls: ALIVE task class instantiated once for all configured repeats.
        adata_loader: Callable ``config -> AnnData``.
        model_capacity_builder: Callable ``(adata, config) -> model_capacity``.
        model_wrapper_factory: Callable ``config -> model_wrapper``.
        seed_setter: Callable ``seed -> None`` applied before each repeat.

    Returns:
        ALIVE: The constructed benchmark task after all repeats run.

    Raises:
        TypeError: If the model wrapper is invalid or ``benchmark.run()``
            returns an unsupported type.
    """
    save_benchmark_config(config)

    print("loading benchmark dataset")
    adata = adata_loader(config)
    model_capacity = model_capacity_builder(adata, config)
    model_wrapper = model_wrapper_factory(config)
    _validate_model_wrapper(model_wrapper)

    benchmark = task_cls(
        adata,
        model_capacity,
        model_wrapper,
        config=config,
    )

    runtime_metric_rows = []
    for i in range(config["repeat"]):
        seed = config["seed"] + i
        model_wrapper.config['model_seed'] = seed
        seed_setter(seed)
        repeat_output = benchmark.run(repeat=i)

        if repeat_output is None:
            continue

        if isinstance(repeat_output, dict):
            repeat_rows = repeat_output.get("runtime_metric_rows", [])
        elif isinstance(repeat_output, list):
            repeat_rows = repeat_output
        else:
            raise TypeError(
                "benchmark.run() must return None, a list of runtime metric rows, "
                f"or a dict containing 'runtime_metric_rows'. Got {type(repeat_output)}."
            )

        runtime_metric_rows.extend(repeat_rows)

    save_runtime_metric_summary(saved_dir=config["saved_dir"], rows=runtime_metric_rows)

    return benchmark
