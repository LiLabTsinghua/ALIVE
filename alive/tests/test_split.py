import numpy as np
import pandas as pd
import anndata as ad
import json
import unittest

from alive.split import (
    build_cluster_round_split_labels,
    build_round_split_labels,
    get_round_predefined_condition,
    json_converter,
    load_fixed_conditions_from_config,
    load_predefined_conditions_from_config,
    normalize_predefined_conditions,
    validate_predefined_conditions,
    write_condition_record,
)


def test_build_round_split_labels_assigns_train_val_test_consistently():
    obs = pd.DataFrame(
        {
            "dataset_cell_type": [
                "target_ct",
                "target_ct",
                "target_ct",
                "target_ct",
                "support_ct",
                "support_ct",
            ],
            "split": ["pool", "val", "control", "pool", "pool", "val"],
            "perturbation": [
                "gene_a",
                "gene_val",
                "control",
                "gene_trained",
                "gene_support",
                "gene_support_val",
            ],
        },
        index=[f"cell_{i}" for i in range(6)],
    )
    var = pd.DataFrame(index=["g1", "g2"])
    adata = ad.AnnData(X=np.zeros((6, 2), dtype=float), obs=obs, var=var)

    result = build_round_split_labels(
        adata_model=adata,
        dataset_cell_type="target_ct",
        trained_condition=["gene_trained"],
    )

    expected = pd.Series(
        ["test", "val", "train", "train", "train", "train"],
        index=obs.index,
        dtype="object",
    )

    pd.testing.assert_series_equal(result, expected)


def test_build_cluster_round_split_labels_assigns_train_val_cluster_test_test_consistently():
    obs = pd.DataFrame(
        {
            "dataset_cell_type": [
                "target_ct",
                "target_ct",
                "target_ct",
                "target_ct",
                "target_ct",
                "support_ct",
            ],
            "split": ["pool", "val", "control", "pool", "pool", "pool"],
            "perturbation": [
                "gene_train",
                "gene_val",
                "control",
                "gene_cluster_test",
                "gene_test",
                "gene_support",
            ],
        },
        index=[f"cluster_cell_{i}" for i in range(6)],
    )
    var = pd.DataFrame(index=["g1", "g2"])
    adata = ad.AnnData(X=np.zeros((6, 2), dtype=float), obs=obs, var=var)

    result = build_cluster_round_split_labels(
        adata_model=adata,
        dataset_cell_type="target_ct",
        trained_condition=["gene_train"],
        cluster_test_condition=["gene_cluster_test"],
    )

    expected = pd.Series(
        ["train", "val", "train", "cluster_test", "test", "train"],
        index=obs.index,
        dtype="object",
    )

    pd.testing.assert_series_equal(result, expected)


def test_validate_predefined_conditions_accepts_matching_repeat_and_round_shape():
    predefined_condition = [
        [["gene_a"], ["gene_b"]],
        [["gene_c"], ["gene_d"]],
    ]

    validate_predefined_conditions(
        predefined_condition=predefined_condition,
        repeat_count=2,
        budget_plan=[10, 20],
    )


def test_validate_predefined_conditions_rejects_mismatched_repeat_count():
    predefined_condition = [
        [["gene_a"], ["gene_b"]],
    ]

    with unittest.TestCase().assertRaises(ValueError):
        validate_predefined_conditions(
            predefined_condition=predefined_condition,
            repeat_count=2,
            budget_plan=[10, 20],
        )


def test_validate_predefined_conditions_checks_every_repeat_round_count():
    predefined_condition = [
        [["gene_a"], ["gene_b"]],
        [["gene_c"]],
    ]

    with unittest.TestCase().assertRaises(ValueError):
        validate_predefined_conditions(
            predefined_condition=predefined_condition,
            repeat_count=2,
            budget_plan=[10, 20],
        )


def test_validate_predefined_conditions_accepts_budget_keyed_schedule():
    predefined_condition = [
        {"10": ["gene_a"], "20": ["gene_b"]},
        {"10": ["gene_c"], "20": ["gene_d"]},
    ]

    validate_predefined_conditions(
        predefined_condition=predefined_condition,
        repeat_count=2,
        budget_plan=[10, 20],
    )


def test_get_round_predefined_condition_returns_repeat_specific_schedule():
    predefined_condition = [
        [["gene_a"], ["gene_b"]],
        [["gene_c"], ["gene_d"]],
    ]

    result = get_round_predefined_condition(
        predefined_condition=predefined_condition,
        repeat=1,
        round_index=0,
    )

    assert result == ["gene_c"]


def test_normalize_predefined_conditions_orders_budget_keyed_schedule():
    predefined_condition = [
        {"10": ["gene_a"], "20.0": ["gene_b"]},
        {"10.0": ["gene_c"], "20": ["gene_d"]},
    ]

    normalized = normalize_predefined_conditions(predefined_condition, budget_plan=[10, 20])

    assert normalized == [
        [["gene_a"], ["gene_b"]],
        [["gene_c"], ["gene_d"]],
    ]


def test_load_predefined_conditions_from_config_supports_json_path(tmp_path):
    condition_path = tmp_path / "conditions.json"
    condition_path.write_text(
        json.dumps([{"10": ["gene_a"], "20": ["gene_b"]}]),
        encoding="utf-8",
    )

    result = load_predefined_conditions_from_config(
        config={"predefined_condition_path": str(condition_path)},
        budget_plan=[10, 20],
    )

    assert result == [[["gene_a"], ["gene_b"]]]


def test_load_fixed_conditions_from_config_supports_json_path(tmp_path):
    condition_path = tmp_path / "cluster_test.json"
    condition_path.write_text(json.dumps(["gene_a", "gene_b"]), encoding="utf-8")

    result = load_fixed_conditions_from_config(
        config={"cluster_condition_path": str(condition_path)},
        key_name="cluster_condition",
    )

    assert result == ["gene_a", "gene_b"]


def test_json_converter_converts_numpy_scalar_like_values():
    value = np.array(3.5)[()]
    assert json_converter(value) == 3.5


def test_write_condition_record_persists_expected_json_file(tmp_path):
    write_condition_record(
        saved_dir=str(tmp_path),
        repeat=1,
        condition_record={0.0: ["gene_a"]},
    )

    condition_path = tmp_path / "repeat_1_condition_record.json"

    assert condition_path.exists()

    assert json.loads(condition_path.read_text(encoding="utf-8")) == {"0.0": ["gene_a"]}
