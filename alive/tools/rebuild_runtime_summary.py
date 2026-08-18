"""Rebuild runtime metric summaries from saved prediction bulk tables."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from alive.benchmark import save_runtime_metric_summary
from alive.metrics import (
    compute_runtime_metric_summary,
    load_runtime_gene_sets,
    metric_record_key,
)


PRED_BULK_PATTERN = re.compile(r"repeat_(\d+)_([\d\.]+)_(cluster_test|test)_pred_bulk\.csv\.gz")
RUN_NAME_PATTERN = re.compile(r".+_\d+_\d+$")


def _load_runtime_config(run_dir: Path) -> dict:
    """Load ``config.json`` from a run directory.

    Args:
        run_dir: Benchmark run directory.

    Returns:
        dict: Parsed runtime configuration.
    """
    config_path = run_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_true_bulk(atlas_dir: Path, dataset_cell_type: str) -> pd.DataFrame:
    """Load the ground-truth pseudobulk table for a context.

    Args:
        atlas_dir: Atlas directory containing reference files.
        dataset_cell_type: Target context identifier.

    Returns:
        pandas.DataFrame: Ground-truth pseudobulk table.
    """
    return pd.read_csv(
        atlas_dir / f"{dataset_cell_type}_true_bulk.csv.gz",
        index_col=0,
        compression="gzip",
    )


def _iter_pred_bulk_files(run_dir: Path):
    """Iterate over saved prediction bulk files in a run directory.

    Args:
        run_dir: Benchmark run directory.

    Yields:
        tuple: ``(path, repeat, x, split)`` parsed from each matching filename.
    """
    for pred_path in sorted(run_dir.iterdir()):
        match = PRED_BULK_PATTERN.fullmatch(pred_path.name)
        if match is None:
            continue
        yield pred_path, int(match.group(1)), float(match.group(2)), str(match.group(3))


def _iter_candidate_atlas_dirs(config: dict, run_dir: Path):
    """Iterate over atlas directories to probe when the configured path is stale.

    Args:
        config: Runtime configuration containing atlas location hints.
        run_dir: Benchmark run directory used to derive relative candidates.

    Yields:
        pathlib.Path: Candidate atlas directory.
    """
    atlas_name = config.get("atlas")
    if atlas_name is None:
        return

    normalized_atlas_name = str(atlas_name)
    data_dir = config.get("data_dir")
    if data_dir is not None:
        yield Path(str(data_dir)) / normalized_atlas_name

    for parent_dir in [run_dir, *run_dir.parents]:
        yield parent_dir / "data" / normalized_atlas_name


def _resolve_atlas_dir(
    config: dict,
    run_dir: Path,
    atlas_dir_override: Path | str | None = None,
) -> Path:
    """Resolve the atlas directory, falling back to candidate locations.

    Args:
        config: Run config carrying ``atlas_dir`` (and optionally ``data_dir`` /
            ``atlas``).
        run_dir: Run directory used to derive candidate atlas locations.
        atlas_dir_override: Explicit override; must exist if given.

    Returns:
        pathlib.Path: The resolved atlas directory.

    Raises:
        FileNotFoundError: If no existing atlas directory can be resolved.
    """
    if atlas_dir_override is not None:
        overridden_atlas_dir = Path(atlas_dir_override)
        if overridden_atlas_dir.exists():
            return overridden_atlas_dir
        raise FileNotFoundError(f"atlas_dir_override does not exist: {overridden_atlas_dir}")

    configured_atlas_dir = Path(config["atlas_dir"])
    if configured_atlas_dir.exists():
        return configured_atlas_dir

    seen_candidates = set()
    candidate_dirs = []
    for candidate_dir in _iter_candidate_atlas_dirs(config=config, run_dir=run_dir):
        normalized_candidate = str(candidate_dir)
        if normalized_candidate in seen_candidates:
            continue
        seen_candidates.add(normalized_candidate)
        candidate_dirs.append(candidate_dir)
        if candidate_dir.exists():
            return candidate_dir

    raise FileNotFoundError(
        "Unable to resolve atlas directory. "
        f"configured_atlas_dir={configured_atlas_dir}, "
        f"candidate_dirs={[str(path) for path in candidate_dirs]}"
    )


def _is_positional_column_index(columns) -> bool:
    """Return whether columns are default positional integer labels.

    Args:
        columns: Column labels to inspect.

    Returns:
        bool: Whether labels are ``"0"`` through ``"n-1"`` in order.
    """
    expected_columns = [str(idx) for idx in range(len(columns))]
    return list(columns) == expected_columns


def _align_true_bulk_to_pred_bulk(true_bulk: pd.DataFrame, pred_bulk: pd.DataFrame) -> pd.DataFrame:
    """Align ``true_bulk`` columns to ``pred_bulk``'s gene order.

    Handles already-aligned tables, positional integer column indices, and
    same-set-different-order columns.

    Args:
        true_bulk: Ground-truth pseudobulk.
        pred_bulk: Predicted pseudobulk defining the target gene order.

    Returns:
        pandas.DataFrame: ``true_bulk`` reindexed to ``pred_bulk``'s columns.

    Raises:
        ValueError: If the two tables cannot be aligned on a common gene axis.
    """
    if true_bulk.columns.tolist() == pred_bulk.columns.tolist():
        return true_bulk

    if _is_positional_column_index(true_bulk.columns):
        if true_bulk.shape[1] != pred_bulk.shape[1]:
            raise ValueError("Cannot align positional true_bulk columns because gene dimensions do not match.")
        aligned_true_bulk = true_bulk.copy()
        aligned_true_bulk.columns = pred_bulk.columns.tolist()
        return aligned_true_bulk

    if set(true_bulk.columns) == set(pred_bulk.columns):
        return true_bulk.loc[:, pred_bulk.columns.tolist()]

    raise ValueError("Prediction and truth bulk tables do not share a compatible gene axis.")


def _is_run_dir(candidate_dir: Path) -> bool:
    """Return whether a directory contains a recognizable benchmark run.

    Args:
        candidate_dir: Directory to inspect.

    Returns:
        bool: Whether the directory has a valid run name, config, and prediction.
    """
    if not candidate_dir.is_dir():
        return False
    if not RUN_NAME_PATTERN.fullmatch(candidate_dir.name):
        return False
    if not (candidate_dir / "config.json").exists():
        return False

    for pred_path in candidate_dir.iterdir():
        if PRED_BULK_PATTERN.fullmatch(pred_path.name) is not None:
            return True
    return False


def discover_run_dirs(root_dir: Path | str, methods: list[str] | None = None) -> list[Path]:
    """Recursively find benchmark run directories under ``root_dir``.

    Args:
        root_dir: Directory to search recursively.
        methods: Optional method-name filter (matched against each run
            directory's parent name).

    Returns:
        list[pathlib.Path]: Discovered run directories, sorted.
    """
    resolved_root_dir = Path(root_dir)
    normalized_methods: set[str] | None
    if methods is None:
        normalized_methods = None
    else:
        normalized_methods = {str(method) for method in methods}

    discovered_run_dirs = []
    for candidate_dir in sorted(resolved_root_dir.rglob("*")):
        if not _is_run_dir(candidate_dir):
            continue
        if normalized_methods is not None:
            method_name = candidate_dir.parent.name
            if method_name not in normalized_methods:
                continue
        discovered_run_dirs.append(candidate_dir)

    return discovered_run_dirs


def rebuild_runtime_summary(
    run_dir: Path | str,
    output_path: Path | str | None = None,
    atlas_dir_override: Path | str | None = None,
) -> Path:
    """Recompute one run's ``achievement_summary.csv`` from saved pred_bulk files.

    Args:
        run_dir: Benchmark run directory containing ``config.json`` and
            ``*_pred_bulk.csv.gz`` files.
        output_path: Optional output CSV path; defaults to
            ``<run_dir>/achievement_summary.csv``.
        atlas_dir_override: Optional atlas directory used when ``config.json``
            points to a stale path.

    Returns:
        pathlib.Path: Path to the written summary CSV.

    Raises:
        FileNotFoundError: If the atlas or prediction bulk files cannot be found.
        ValueError: If prediction and truth gene axes are incompatible.
    """
    resolved_run_dir = Path(run_dir)
    config = _load_runtime_config(resolved_run_dir)

    atlas_dir = _resolve_atlas_dir(
        config=config,
        run_dir=resolved_run_dir,
        atlas_dir_override=atlas_dir_override,
    )
    dataset_cell_type = str(config["dataset_cell_type"])
    true_bulk = _load_true_bulk(atlas_dir=atlas_dir, dataset_cell_type=dataset_cell_type)
    hvg_genes, de_dict = load_runtime_gene_sets(
        atlas_dir=str(atlas_dir),
        dataset_cell_type=dataset_cell_type,
    )

    rows = []
    for pred_path, repeat, x_value, split_name in _iter_pred_bulk_files(resolved_run_dir):
        pred_bulk = pd.read_csv(pred_path, index_col=0, compression="gzip")
        aligned_true_bulk = _align_true_bulk_to_pred_bulk(true_bulk=true_bulk, pred_bulk=pred_bulk)
        ref_control = aligned_true_bulk.loc["control"].to_numpy(dtype=float)
        ref_pert_mean = aligned_true_bulk.loc[aligned_true_bulk.index != "control"].mean(axis=0).to_numpy(dtype=float)
        metric_summary = compute_runtime_metric_summary(
            pred_bulk=pred_bulk,
            true_bulk=aligned_true_bulk,
            ref_control=ref_control,
            ref_pert_mean=ref_pert_mean,
            hvg_genes=hvg_genes,
            de_dict=de_dict,
        )

        for gene_set_name, metric_values in metric_summary.items():
            gene_name = str(gene_set_name).lower()
            for metric_name, value in metric_values.items():
                rows.append(
                    {
                        "method": str(config["model_name"]),
                        "context": dataset_cell_type,
                        "repeat": int(repeat),
                        "x": float(x_value),
                        "set": split_name,
                        "metric_name": metric_record_key(metric_name),
                        "gene": gene_name,
                        "value": float(value),
                    }
                )

    if not rows:
        raise FileNotFoundError(f"No pred_bulk files found under {resolved_run_dir}")

    if output_path is None:
        return save_runtime_metric_summary(saved_dir=resolved_run_dir, rows=rows)

    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(rows).sort_values(
        ["method", "context", "repeat", "x", "set", "metric_name", "gene"],
    ).reset_index(drop=True)
    summary_df.to_csv(resolved_output_path, index=False)
    return resolved_output_path


def rebuild_runtime_summary_batch(
    root_dir: Path | str,
    methods: list[str] | None = None,
    atlas_dir_override: Path | str | None = None,
) -> list[Path]:
    """Rebuild summaries for every run directory discovered under ``root_dir``.

    Args:
        root_dir: Directory searched recursively for run directories.
        methods: Optional method-name filter.
        atlas_dir_override: Optional atlas directory override.

    Returns:
        list[pathlib.Path]: Paths to the rebuilt summary CSVs.

    Raises:
        FileNotFoundError: If no run directories are found under ``root_dir``.
    """
    run_dirs = discover_run_dirs(root_dir=root_dir, methods=methods)
    if not run_dirs:
        raise FileNotFoundError(f"No benchmark run directories found under {Path(root_dir)}")

    written_paths = []
    for run_dir in run_dirs:
        written_paths.append(
            rebuild_runtime_summary(
                run_dir=run_dir,
                atlas_dir_override=atlas_dir_override,
            )
        )
    return written_paths


def parse_args(argv=None):
    """Parse and validate CLI arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv`` when ``None``.

    Returns:
        argparse.Namespace: Parsed arguments.

    Raises:
        ValueError: On mutually exclusive or missing argument combinations.
    """
    parser = argparse.ArgumentParser(
        description="Rebuild benchmark run achievement_summary.csv files from saved pred_bulk files.",
    )
    parser.add_argument("--run-dir", help="One benchmark run directory containing config.json.")
    parser.add_argument("--root-dir", help="Root directory under which benchmark run directories will be discovered recursively.")
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Optional method names used to filter discovered run directories under --root-dir.",
    )
    parser.add_argument(
        "--atlas-dir-override",
        help="Optional atlas directory override used when config.json points to a stale path.",
    )
    parser.add_argument("--output", help="Optional output CSV path. Defaults to <run-dir>/achievement_summary.csv.")
    args = parser.parse_args(argv)

    if args.run_dir is not None and args.root_dir is not None:
        raise ValueError("--run-dir and --root-dir cannot be used together")
    if args.run_dir is None and args.root_dir is None:
        raise ValueError("One of --run-dir or --root-dir must be provided")
    if args.root_dir is None and args.methods is not None:
        raise ValueError("--methods can only be used together with --root-dir")
    if args.root_dir is not None and args.output is not None:
        raise ValueError("--output is only supported together with --run-dir")

    return args


def main(argv=None):
    """CLI entry point: rebuild summaries for a single run or a tree of runs.

    Args:
        argv: Argument list; defaults to ``sys.argv`` when ``None``.

    Returns:
        int: Process exit code (0 on success).

    Raises:
        ValueError: If parsed arguments do not select a supported execution mode.
    """
    args = parse_args(argv)
    if args.run_dir is not None:
        rebuild_runtime_summary(
            run_dir=args.run_dir,
            output_path=args.output,
            atlas_dir_override=args.atlas_dir_override,
        )
        return 0

    if args.root_dir is not None:
        rebuild_runtime_summary_batch(
            root_dir=args.root_dir,
            methods=args.methods,
            atlas_dir_override=args.atlas_dir_override,
        )
        return 0

    raise ValueError("Unsupported argument combination")


if __name__ == "__main__":
    raise SystemExit(main())
