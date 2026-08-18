"""Condition schedule persistence, loading, validation, and split labeling."""

import json
from pathlib import Path

import pandas as pd


def json_converter(o):
    """Convert numpy scalar-like objects into plain Python JSON values.

    Args:
        o: Object to serialize; numpy scalars expose ``.item()``.

    Returns:
        object: JSON-serializable value.
    """
    if hasattr(o, "item"):
        return o.item()
    return o


def write_condition_record(saved_dir, repeat, condition_record):
    """Persist one repeat's selected perturbation schedule.

    Args:
        saved_dir: Benchmark run output directory.
        repeat: Zero-based repeat index used in the filename.
        condition_record: Mapping of budget to selected perturbations.
    """
    with open(f"{saved_dir}/repeat_{repeat}_condition_record.json", "w", encoding="utf-8") as f:
        json.dump(condition_record, f, indent=4, ensure_ascii=False, default=json_converter)


def _budget_key_candidates(budget):
    """Generate stable string candidates for matching a budget in JSON configs.

    Args:
        budget: Budget percentage to look up.

    Returns:
        set[str]: Candidate string keys that may represent ``budget``.
    """
    budget_value = float(budget)
    budget_int = int(budget_value)
    candidates = {
        str(budget),
        str(budget_value),
        f"{budget_value:.1f}",
    }
    if budget_value.is_integer():
        candidates.add(str(budget_int))
    return candidates


def _normalize_budget_keyed_repeat(repeat_schedule, budget_plan):
    """Convert one repeat schedule from ``{budget: perturbations}`` into round order.

    Args:
        repeat_schedule: Mapping of budget key to selected perturbations.
        budget_plan: Ordered budgets defining the round order.

    Returns:
        list: Selected perturbations per round, in budget-plan order.

    Raises:
        KeyError: If a budget in ``budget_plan`` has no matching key.
    """
    normalized_rounds = []
    for budget in budget_plan:
        matched_key = next(
            (candidate for candidate in _budget_key_candidates(budget) if candidate in repeat_schedule),
            None,
        )
        if matched_key is None:
            raise KeyError(
                f"Missing predefined condition for budget {budget}. "
                f"Available keys: {sorted(repeat_schedule.keys())}"
            )
        normalized_rounds.append(repeat_schedule[matched_key])
    return normalized_rounds


def normalize_predefined_conditions(predefined_condition, budget_plan):
    """Normalize predefined-condition config formats into ``[repeat][round]``.

    Supported formats:
        - legacy nested list: ``[[round0, round1, ...], ...]``
        - budget-keyed list: ``[{"0.5": [...], "1": [...]}, ...]``

    Args:
        predefined_condition: Raw predefined condition, or ``None``.
        budget_plan: Ordered budgets defining the round order.

    Returns:
        list | None: Condition normalized to ``[repeat][round]``, or ``None``.
    """
    if predefined_condition is None:
        return None

    if isinstance(predefined_condition, list) and predefined_condition:
        first_entry = predefined_condition[0]
        if isinstance(first_entry, dict):
            return [
                _normalize_budget_keyed_repeat(repeat_schedule, budget_plan)
                for repeat_schedule in predefined_condition
            ]
        return predefined_condition

    return predefined_condition


def load_predefined_conditions_from_config(config, budget_plan):
    """Load predefined conditions from config, optionally resolving a file path.

    Args:
        config: Runtime config that may carry ``predefined_condition`` or
            ``predefined_condition_path``.
        budget_plan: Ordered budgets defining the round order.

    Returns:
        list | None: Normalized ``[repeat][round]`` condition, or ``None``.

    Raises:
        ValueError: If both the inline value and the file path are provided.
    """
    predefined_condition = config.get("predefined_condition")
    predefined_condition_path = config.get("predefined_condition_path")

    if predefined_condition is not None and predefined_condition_path is not None:
        raise ValueError("Use only one of config['predefined_condition'] or config['predefined_condition_path'].")

    if predefined_condition_path is not None:
        with open(Path(predefined_condition_path), "r", encoding="utf-8") as f:
            predefined_condition = json.load(f)

    return normalize_predefined_conditions(predefined_condition, budget_plan)


def load_fixed_conditions_from_config(config, key_name):
    """Load a fixed perturbation set from config, optionally resolving a file path.

    Args:
        config: Runtime config that may carry ``key_name`` or ``{key_name}_path``.
        key_name: Base config key (e.g. ``"cluster_condition"``).

    Returns:
        object | None: Loaded condition value, or ``None`` if unset.

    Raises:
        ValueError: If both the inline value and the file path are provided.
    """
    condition = config.get(key_name)
    condition_path = config.get(f"{key_name}_path")

    if condition is not None and condition_path is not None:
        raise ValueError(f"Use only one of config['{key_name}'] or config['{key_name}_path'].")

    if condition_path is not None:
        with open(Path(condition_path), "r", encoding="utf-8") as f:
            condition = json.load(f)

    return condition


def validate_predefined_conditions(predefined_condition, repeat_count, budget_plan):
    """Validate the predefined condition table used to replay a schedule.

    The expected shape is ``[repeat][round]`` and the round dimension must match
    the configured budget plan exactly.

    Args:
        predefined_condition: Raw predefined condition to validate.
        repeat_count: Expected number of repeats.
        budget_plan: Ordered budgets defining the round dimension.

    Raises:
        ValueError: If the repeat or round dimension does not match.
    """
    normalized_condition = normalize_predefined_conditions(predefined_condition, budget_plan)

    if len(normalized_condition) != repeat_count:
        raise ValueError(
            f"The length of predefined_condition ({len(normalized_condition)}) "
            f"must equal the repeat count ({repeat_count})."
        )

    expected_round_count = len(budget_plan)
    invalid_repeat_indices = [
        repeat_index
        for repeat_index, repeat_schedule in enumerate(normalized_condition)
        if len(repeat_schedule) != expected_round_count
    ]
    if invalid_repeat_indices:
        raise ValueError(
            "Every predefined-condition repeat must match the budget-plan length "
            f"({expected_round_count}). Invalid repeat indices: {invalid_repeat_indices}."
        )


def get_round_predefined_condition(predefined_condition, repeat, round_index):
    """Return the preselected perturbation set for a repeat/round pair.

    Args:
        predefined_condition: Normalized ``[repeat][round]`` condition.
        repeat: Repeat index.
        round_index: Round index within the repeat.

    Returns:
        list: Selected perturbations for that repeat/round.
    """
    return predefined_condition[repeat][round_index]


def build_round_split_labels(adata_model, dataset_cell_type, trained_condition):
    """Build round-specific split labels for the in-distribution loop.

    Invariants:
        - support-context rows default to ``train``
        - target-context rows default to ``test``
        - target validation rows are promoted to ``val``
        - target control rows are promoted to ``train``
        - target rows whose perturbations are already selected become ``train``

    Args:
        adata_model: Model-facing ``AnnData`` whose rows are labeled.
        dataset_cell_type: Target context identifier (``dataset_cell_type``).
        trained_condition: Perturbations already selected for training.

    Returns:
        pandas.Series: Standalone split labels, so callers can assign to any
        split key without mutating other columns implicitly.
    """
    split_labels = pd.Series("train", index=adata_model.obs.index, dtype="object")

    target_mask = adata_model.obs["dataset_cell_type"] == dataset_cell_type
    split_labels.loc[target_mask] = "test"

    val_mask = target_mask & (adata_model.obs["split"] == "val")
    split_labels.loc[val_mask] = "val"

    ctrl_mask = target_mask & (adata_model.obs["perturbation"] == "control")
    split_labels.loc[ctrl_mask] = "train"

    trained_mask = target_mask & adata_model.obs["perturbation"].isin(trained_condition)
    split_labels.loc[trained_mask] = "train"

    return split_labels


def build_cluster_round_split_labels(
    adata_model,
    dataset_cell_type,
    trained_condition,
    cluster_test_condition,
):
    """Build round split labels for the out-of-distribution (cluster) workflow.

    Invariants:
        - support-context rows stay ``train``
        - target-context rows default to ``test``
        - target validation rows are promoted to ``val``
        - target control rows are promoted to ``train``
        - target fixed cluster-holdout rows are promoted to ``cluster_test``
        - target adaptive trained perturbations are promoted to ``train``

    Args:
        adata_model: Model-facing ``AnnData`` whose rows are labeled.
        dataset_cell_type: Target context identifier (``dataset_cell_type``).
        trained_condition: Adaptive perturbations already selected for training.
        cluster_test_condition: Fixed cluster-holdout perturbations.

    Returns:
        pandas.Series: Standalone split labels for the cluster workflow.
    """
    split_labels = pd.Series("train", index=adata_model.obs.index, dtype="object")

    target_mask = adata_model.obs["dataset_cell_type"] == dataset_cell_type
    split_labels.loc[target_mask] = "test"

    val_mask = target_mask & (adata_model.obs["split"] == "val")
    split_labels.loc[val_mask] = "val"

    ctrl_mask = target_mask & (adata_model.obs["perturbation"] == "control")
    split_labels.loc[ctrl_mask] = "train"

    cluster_test_mask = target_mask & adata_model.obs["perturbation"].isin(cluster_test_condition)
    split_labels.loc[cluster_test_mask] = "cluster_test"

    trained_mask = target_mask & adata_model.obs["perturbation"].isin(trained_condition)
    split_labels.loc[trained_mask] = "train"

    return split_labels
