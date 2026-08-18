import unittest
import json
from pathlib import Path
from unittest.mock import patch
import inspect
import warnings

import anndata as ad
import numpy as np
import pandas as pd

from alive.tasks import ActiveLearning, InDistribution, OutOfDistribution


class DummyModelWrapper:
    """
    Minimal model wrapper used to exercise the shared task orchestration path.
    """

    def __init__(self):
        self.fit_calls = []
        self.predict_calls = []

    def fit(self, adata_model, split_key):
        self.fit_calls.append(split_key)

    def predict(self, adata_model, split_key, split):
        self.predict_calls.append((split_key, split))

        split_mask = adata_model.obs[split_key] == split
        perts = [
            pert
            for pert in adata_model.obs.loc[split_mask, "perturbation"].astype(str).unique().tolist()
            if pert != "control"
        ]

        control_mask = adata_model.obs["perturbation"] == "control"
        control_mean = np.asarray(adata_model.X[control_mask].mean(axis=0)).reshape(1, -1)

        X = np.repeat(control_mean, len(perts), axis=0)
        obs = pd.DataFrame({"perturbation": perts}, index=[f"{split}_{pert}" for pert in perts])
        return ad.AnnData(X=X, obs=obs, var=adata_model.var.copy())


def _make_in_distribution_fixture():
    obs = pd.DataFrame(
        {
            "dataset_cell_type": ["target_ct"] * 5,
            "split": ["control", "control", "pool", "pool", "val"],
            "perturbation": ["control", "control", "gene_a", "gene_b", "gene_val"],
        },
        index=[f"cell_{i}" for i in range(5)],
    )
    var = pd.DataFrame(index=["g1", "g2"])
    X = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.5],
            [3.0, 1.0],
            [1.5, 2.5],
        ],
        dtype=float,
    )
    return ad.AnnData(X=X, obs=obs, var=var)


def _make_out_of_distribution_fixture():
    obs = pd.DataFrame(
        {
            "dataset_cell_type": ["target_ct"] * 7,
            "split": ["control", "control", "pool", "pool", "pool", "pool", "val"],
            "perturbation": [
                "control",
                "control",
                "gene_a",
                "gene_b",
                "gene_cluster",
                "gene_pool",
                "gene_val",
            ],
        },
        index=[f"cluster_cell_{i}" for i in range(7)],
    )
    var = pd.DataFrame(index=["g1", "g2"])
    X = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.5],
            [3.0, 1.0],
            [4.0, 0.5],
            [2.5, 2.5],
            [1.5, 2.5],
        ],
        dtype=float,
    )
    return ad.AnnData(X=X, obs=obs, var=var)


def _make_in_distribution_fixture_without_target_control():
    obs = pd.DataFrame(
        {
            "dataset_cell_type": ["other_ct", "target_ct", "target_ct"],
            "split": ["control", "pool", "val"],
            "perturbation": ["control", "gene_a", "gene_val"],
        },
        index=[f"missing_ctrl_cell_{i}" for i in range(3)],
    )
    var = pd.DataFrame(index=["g1", "g2"])
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [1.5, 1.0],
        ],
        dtype=float,
    )
    return ad.AnnData(X=X, obs=obs, var=var)


def _make_in_distribution_fixture_with_empty_pool():
    obs = pd.DataFrame(
        {
            "dataset_cell_type": ["target_ct", "target_ct", "target_ct"],
            "split": ["control", "control", "val"],
            "perturbation": ["control", "control", "gene_val"],
        },
        index=[f"empty_pool_cell_{i}" for i in range(3)],
    )
    var = pd.DataFrame(index=["g1", "g2"])
    X = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.5, 2.5],
        ],
        dtype=float,
    )
    return ad.AnnData(X=X, obs=obs, var=var)


def _make_active_learning_fixture():
    obs = pd.DataFrame(
        {
            "dataset_cell_type": ["target_ct"] * 7,
            "split": ["control", "control", "pool", "pool", "pool", "pool", "val"],
            "perturbation": ["control", "control", "gene_a", "gene_b", "gene_c", "gene_d", "gene_val"],
        },
        index=[f"active_cell_{i}" for i in range(7)],
    )
    var = pd.DataFrame(index=["g1", "g2"])
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 2.0],
            [3.0, 0.0],
            [0.5, 0.5],
        ],
        dtype=float,
    )
    return ad.AnnData(X=X, obs=obs, var=var)


def _make_active_learning_correlation_reference_fixture():
    obs = pd.DataFrame(
        {
            "dataset_cell_type": ["target_ct"] * 7,
            "split": ["control", "control", "pool", "pool", "pool", "pool", "val"],
            "perturbation": ["control", "control", "gene_a", "gene_b", "gene_c", "gene_d", "gene_val"],
        },
        index=[f"active_pcc_cell_{i}" for i in range(7)],
    )
    var = pd.DataFrame(index=["g1", "g2", "g3"])
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    return ad.AnnData(X=X, obs=obs, var=var)


def _write_runtime_metric_reference_files(tmp_path, dataset_cell_type, hvg_genes, de_dict):
    atlas_dir = tmp_path / "atlas"
    atlas_dir.mkdir()
    (atlas_dir / f"{dataset_cell_type}_hvg.json").write_text(
        json.dumps(hvg_genes),
        encoding="utf-8",
    )
    (atlas_dir / f"{dataset_cell_type}_de.json").write_text(
        json.dumps(de_dict),
        encoding="utf-8",
    )
    return atlas_dir


def test_in_distribution_run_persists_round_outputs(tmp_path):
    adata = _make_in_distribution_fixture()
    model_wrapper = DummyModelWrapper()
    atlas_dir = _write_runtime_metric_reference_files(
        tmp_path=tmp_path,
        dataset_cell_type="target_ct",
        hvg_genes=["g1"],
        de_dict={"gene_a": ["g1"], "gene_b": ["g2"], "gene_val": ["g1"]},
    )

    task = InDistribution(
        adata=adata,
        model_capacity=["gene_a", "gene_b", "gene_val"],
        model_wrapper=model_wrapper,
        config={
            "budget_plan": [50.0],
            "strategy": "random_sampling",
            "start_mode": "zero-shot",
            "dataset_cell_type": "target_ct",
            "use_support_data": False,
            "saved_dir": str(tmp_path),
            "atlas_dir": str(atlas_dir),
            "model_name": "dummy-wrapper",
            "repeat": 1,
            "predefined_condition": [{"50.0": ["gene_a"]}],
        },
    )

    run_result = task.run(repeat=0)

    expected_files = [
        "repeat_0_50.0_test_pred_bulk.csv.gz",
        "repeat_0_condition_record.json",
    ]

    for name in expected_files:
        assert (Path(tmp_path) / name).exists()

    assert not (Path(tmp_path) / "repeat_0_val_achievement_record.json").exists()
    assert not (Path(tmp_path) / "repeat_0_test_achievement_record.json").exists()

    runtime_rows = run_result["runtime_metric_rows"]
    assert len(runtime_rows) == 15
    assert {row["context"] for row in runtime_rows} == {"target_ct"}
    assert {row["set"] for row in runtime_rows} == {"test"}
    assert {row["metric_name"] for row in runtime_rows} == {
        "mse",
        "mae",
        "correlation",
        "systema_correlation",
        "systema_bias",
    }
    assert {row["gene"] for row in runtime_rows} == {"all", "hvg", "de"}

    assert len(model_wrapper.fit_calls) == 1
    assert len(model_wrapper.predict_calls) == 1
    round_key = model_wrapper.fit_calls[0]
    assert model_wrapper.predict_calls == [(round_key, "test")]


def test_random_cold_start_evaluates_zero_shot_baseline(tmp_path):
    adata = _make_in_distribution_fixture()
    model_wrapper = DummyModelWrapper()
    atlas_dir = _write_runtime_metric_reference_files(
        tmp_path=tmp_path,
        dataset_cell_type="target_ct",
        hvg_genes=["g1"],
        de_dict={"gene_a": ["g1"], "gene_b": ["g2"]},
    )

    task = InDistribution(
        adata=adata,
        model_capacity=["gene_a", "gene_b", "gene_val"],
        model_wrapper=model_wrapper,
        config={
            "budget_plan": [50.0],
            "strategy": "random_sampling",
            "start_mode": "random",
            "dataset_cell_type": "target_ct",
            "use_support_data": False,
            "saved_dir": str(tmp_path),
            "atlas_dir": str(atlas_dir),
            "model_name": "dummy-wrapper",
            "repeat": 1,
        },
    )

    run_result = task.run(repeat=0)

    # The random cold start fits no model but still scores a zero-shot baseline
    # (all targets padded to the control mean) and writes its pred_bulk.
    assert (Path(tmp_path) / "repeat_0_0.0_test_pred_bulk.csv.gz").exists()
    assert len(model_wrapper.fit_calls) == 0
    assert len(model_wrapper.predict_calls) == 0

    runtime_rows = run_result["runtime_metric_rows"]
    mse_all = next(
        row
        for row in runtime_rows
        if row["x"] == 0.0 and row["set"] == "test" and row["metric_name"] == "mse" and row["gene"] == "all"
    )
    # Control baseline differs from the true perturbed profiles, so MSE is real.
    assert mse_all["value"] > 0.0


def test_out_of_distribution_excludes_cluster_test_from_pool_and_persists_split_outputs(tmp_path):
    adata = _make_out_of_distribution_fixture()
    model_wrapper = DummyModelWrapper()
    atlas_dir = _write_runtime_metric_reference_files(
        tmp_path=tmp_path,
        dataset_cell_type="target_ct",
        hvg_genes=["g1"],
        de_dict={
            "gene_a": ["g1"],
            "gene_b": ["g2"],
            "gene_cluster": ["g1"],
            "gene_pool": ["g2"],
            "gene_val": ["g1"],
        },
    )
    cluster_condition_path = tmp_path / "cluster_condition.json"
    cluster_condition_path.write_text('["gene_cluster"]', encoding="utf-8")

    task = OutOfDistribution(
        adata=adata,
        model_capacity=["gene_a", "gene_b", "gene_cluster", "gene_pool", "gene_val"],
        model_wrapper=model_wrapper,
        config={
            "budget_plan": [50.0],
            "strategy": "random_sampling",
            "start_mode": "zero-shot",
            "dataset_cell_type": "target_ct",
            "use_support_data": False,
            "saved_dir": str(tmp_path),
            "atlas_dir": str(atlas_dir),
            "model_name": "dummy-wrapper",
            "repeat": 1,
            "max_budget_gap": 100.0,
            "cluster_condition_path": str(cluster_condition_path),
            "predefined_condition": [{"50.0": ["gene_a"]}],
        },
    )

    assert task.cluster_test == ["gene_cluster"]
    assert task.pool == ["gene_a", "gene_b", "gene_pool"]
    assert task.total_pool_count == 3

    run_result = task.run(repeat=0)

    expected_files = [
        "repeat_0_33.3_cluster_test_pred_bulk.csv.gz",
        "repeat_0_33.3_test_pred_bulk.csv.gz",
        "repeat_0_condition_record.json",
    ]

    for name in expected_files:
        assert (Path(tmp_path) / name).exists()

    assert not (Path(tmp_path) / "repeat_0_val_achievement_record.json").exists()
    assert not (Path(tmp_path) / "repeat_0_cluster_test_achievement_record.json").exists()
    assert not (Path(tmp_path) / "repeat_0_test_achievement_record.json").exists()

    runtime_rows = run_result["runtime_metric_rows"]
    assert len(runtime_rows) == 30
    assert {row["context"] for row in runtime_rows} == {"target_ct"}
    assert {row["set"] for row in runtime_rows} == {"cluster_test", "test"}
    assert {row["metric_name"] for row in runtime_rows} == {
        "mse",
        "mae",
        "correlation",
        "systema_correlation",
        "systema_bias",
    }
    assert {row["gene"] for row in runtime_rows} == {"all", "hvg", "de"}

    assert len(model_wrapper.fit_calls) == 1
    round_key = model_wrapper.fit_calls[0]
    assert model_wrapper.predict_calls == [
        (round_key, "cluster_test"),
        (round_key, "test"),
    ]


def test_out_of_distribution_requires_cluster_condition_path(tmp_path):
    adata = _make_out_of_distribution_fixture()

    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        "OutOfDistribution requires 'cluster_condition_path'.",
    ):
        OutOfDistribution(
            adata=adata,
            model_capacity=["gene_a", "gene_b", "gene_cluster", "gene_pool", "gene_val"],
            model_wrapper=DummyModelWrapper(),
            config={
                "budget_plan": [50.0],
                "strategy": "random_sampling",
                "start_mode": "zero-shot",
                "dataset_cell_type": "target_ct",
                "use_support_data": False,
                "saved_dir": str(tmp_path),
                "model_name": "dummy-wrapper",
                "repeat": 1,
            },
        )


def test_in_distribution_init_requires_control_in_evaluation_subset(tmp_path):
    adata = _make_in_distribution_fixture_without_target_control()

    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        "Control cells were not found in the evaluation subset.",
    ):
        InDistribution(
            adata=adata,
            model_capacity=["gene_a", "gene_val"],
            model_wrapper=DummyModelWrapper(),
            config={
                "budget_plan": [50.0],
                "strategy": "random_sampling",
                "start_mode": "zero-shot",
                "dataset_cell_type": "target_ct",
                "use_support_data": False,
                "saved_dir": str(tmp_path),
                "model_name": "dummy-wrapper",
                "repeat": 1,
            },
        )


def test_in_distribution_init_rejects_generated_split_values_in_raw_atlas(tmp_path):
    adata = _make_in_distribution_fixture()
    adata.obs.loc[adata.obs.index[2], "split"] = "train"

    with unittest.TestCase().assertRaisesRegex(ValueError, "train"):
        InDistribution(
            adata=adata,
            model_capacity=["gene_a", "gene_b", "gene_val"],
            model_wrapper=DummyModelWrapper(),
            config={
                "budget_plan": [50.0],
                "strategy": "random_sampling",
                "start_mode": "zero-shot",
                "dataset_cell_type": "target_ct",
                "use_support_data": False,
                "saved_dir": str(tmp_path),
                "model_name": "dummy-wrapper",
                "repeat": 1,
            },
        )


def test_in_distribution_init_rejects_inconsistent_control_split(tmp_path):
    adata = _make_in_distribution_fixture()
    adata.obs.loc[adata.obs.index[0], "split"] = "pool"

    with unittest.TestCase().assertRaisesRegex(ValueError, "inconsistent rows"):
        InDistribution(
            adata=adata,
            model_capacity=["gene_a", "gene_b", "gene_val"],
            model_wrapper=DummyModelWrapper(),
            config={
                "budget_plan": [50.0],
                "strategy": "random_sampling",
                "start_mode": "zero-shot",
                "dataset_cell_type": "target_ct",
                "use_support_data": False,
                "saved_dir": str(tmp_path),
                "model_name": "dummy-wrapper",
                "repeat": 1,
            },
        )


def test_validation_rows_remain_in_model_split_without_redundant_attributes(tmp_path):
    adata = _make_in_distribution_fixture()
    task = InDistribution(
        adata=adata,
        model_capacity=["gene_a", "gene_b", "gene_val"],
        model_wrapper=DummyModelWrapper(),
        config={
            "budget_plan": [50.0],
            "strategy": "random_sampling",
            "start_mode": "zero-shot",
            "dataset_cell_type": "target_ct",
            "use_support_data": False,
            "saved_dir": str(tmp_path),
            "model_name": "dummy-wrapper",
            "repeat": 1,
        },
    )

    round_labels = task._build_round_split_labels(["gene_a"])
    validation_mask = task.adata_model.obs["split"] == "val"

    assert validation_mask.any()
    assert set(round_labels[validation_mask]) == {"val"}
    assert not hasattr(task, "adata_val")
    assert not hasattr(task, "val")


def test_prediction_schema_accepts_repeated_requested_perturbations(tmp_path):
    adata = _make_in_distribution_fixture()
    task = InDistribution(
        adata=adata,
        model_capacity=["gene_a", "gene_b", "gene_val"],
        model_wrapper=DummyModelWrapper(),
        config={
            "budget_plan": [50.0],
            "strategy": "random_sampling",
            "start_mode": "zero-shot",
            "dataset_cell_type": "target_ct",
            "use_support_data": False,
            "saved_dir": str(tmp_path),
            "model_name": "dummy-wrapper",
            "repeat": 1,
        },
    )
    prediction = ad.AnnData(
        X=np.ones((2, adata.n_vars)),
        obs=pd.DataFrame(
            {"perturbation": ["gene_b", "gene_b"]},
            index=["pred_0", "pred_1"],
        ),
        var=adata.var.copy(),
    )

    task._validate_prediction_schema(prediction, {"gene_b"}, "test")


def test_prediction_schema_rejects_invalid_predictions(tmp_path):
    adata = _make_in_distribution_fixture()
    task = InDistribution(
        adata=adata,
        model_capacity=["gene_a", "gene_b", "gene_val"],
        model_wrapper=DummyModelWrapper(),
        config={
            "budget_plan": [50.0],
            "strategy": "random_sampling",
            "start_mode": "zero-shot",
            "dataset_cell_type": "target_ct",
            "use_support_data": False,
            "saved_dir": str(tmp_path),
            "model_name": "dummy-wrapper",
            "repeat": 1,
        },
    )

    missing_column = ad.AnnData(
        X=np.ones((1, adata.n_vars)),
        obs=pd.DataFrame(index=["pred_0"]),
        var=adata.var.copy(),
    )
    with unittest.TestCase().assertRaisesRegex(KeyError, r"obs\['perturbation'\]"):
        task._validate_prediction_schema(missing_column, {"gene_b"}, "test")

    missing_label = ad.AnnData(
        X=np.ones((1, adata.n_vars)),
        obs=pd.DataFrame({"perturbation": [None]}, index=["pred_0"]),
        var=adata.var.copy(),
    )
    with unittest.TestCase().assertRaisesRegex(ValueError, "missing perturbation"):
        task._validate_prediction_schema(missing_label, {"gene_b"}, "test")

    wrong_gene_order = ad.AnnData(
        X=np.ones((1, adata.n_vars)),
        obs=pd.DataFrame({"perturbation": ["gene_b"]}, index=["pred_0"]),
        var=pd.DataFrame(index=list(reversed(adata.var_names.tolist()))),
    )
    with unittest.TestCase().assertRaisesRegex(ValueError, "gene names and order"):
        task._validate_prediction_schema(wrong_gene_order, {"gene_b"}, "test")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Variable names are not unique")
        duplicate_genes = ad.AnnData(
            X=np.ones((1, adata.n_vars)),
            obs=pd.DataFrame({"perturbation": ["gene_b"]}, index=["pred_0"]),
            var=pd.DataFrame(index=["g1", "g1"]),
        )
    with unittest.TestCase().assertRaisesRegex(ValueError, "duplicate genes"):
        task._validate_prediction_schema(duplicate_genes, {"gene_b"}, "test")

    extra_perturbation = ad.AnnData(
        X=np.ones((2, adata.n_vars)),
        obs=pd.DataFrame(
            {"perturbation": ["gene_b", "gene_a"]},
            index=["pred_0", "pred_1"],
        ),
        var=adata.var.copy(),
    )
    with unittest.TestCase().assertRaisesRegex(ValueError, "gene_a"):
        task._validate_prediction_schema(extra_perturbation, {"gene_b"}, "test")


def test_in_distribution_sample_rejects_large_budget_gap_for_empty_pool(tmp_path):
    adata = _make_in_distribution_fixture_with_empty_pool()
    task = InDistribution(
        adata=adata,
        model_capacity=["gene_val"],
        model_wrapper=DummyModelWrapper(),
        config={
            "budget_plan": [50.0],
            "strategy": "random_sampling",
            "start_mode": "zero-shot",
            "dataset_cell_type": "target_ct",
            "use_support_data": False,
            "saved_dir": str(tmp_path),
            "model_name": "dummy-wrapper",
            "repeat": 1,
        },
    )

    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        "target_budget=50.0, actual_budget=0.0",
    ):
        task.sample(
            adata_pred=None,
            trained_condition=[],
            target_budget=50.0,
            strategy="random_sampling",
        )


def test_in_distribution_sample_deduplicates_trained_condition_for_source_budget(tmp_path):
    adata = _make_in_distribution_fixture()
    task = InDistribution(
        adata=adata,
        model_capacity=["gene_a", "gene_b", "gene_val"],
        model_wrapper=DummyModelWrapper(),
        config={
            "budget_plan": [50.0],
            "strategy": "random_sampling",
            "start_mode": "zero-shot",
            "dataset_cell_type": "target_ct",
            "use_support_data": False,
            "saved_dir": str(tmp_path),
            "model_name": "dummy-wrapper",
            "repeat": 1,
        },
    )

    source_budget, result_budget, selected = task.sample(
        adata_pred=None,
        trained_condition=["gene_a", "gene_a"],
        target_budget=50.0,
        strategy="random_sampling",
    )

    assert source_budget == 50.0
    assert result_budget == 50.0
    assert len(selected) == 1


def test_active_learning_sample_prefers_predicted_candidates_before_blind_spots(tmp_path):
    adata = _make_active_learning_fixture()
    task = ActiveLearning(
        adata=adata,
        model_capacity=["gene_a", "gene_b", "gene_c", "gene_d", "gene_val"],
        model_wrapper=DummyModelWrapper(),
        config={
            "budget_plan": [75.0],
            "strategy": "diversity_sampling",
            "start_mode": "zero-shot",
            "dataset_cell_type": "target_ct",
            "use_support_data": False,
            "saved_dir": str(tmp_path),
            "model_name": "dummy-wrapper",
            "repeat": 1,
        },
    )

    adata_pred = ad.AnnData(
        X=np.array([[0.0, 1.0], [2.0, 2.0]], dtype=float),
        obs=pd.DataFrame({"perturbation": ["gene_b", "gene_c"]}, index=["pred_b", "pred_c"]),
        var=adata.var.copy(),
    )

    source_budget, result_budget, selected = task.sample(
        adata_pred=adata_pred,
        trained_condition=["gene_a"],
        target_budget=75.0,
        strategy="diversity_sampling",
    )

    assert source_budget == 25.0
    assert result_budget == 75.0
    assert set(selected) == {"gene_a", "gene_b", "gene_c"}
    assert "gene_d" not in selected


def test_active_learning_rejects_legacy_active_diversity_strategy_name(tmp_path):
    adata = _make_active_learning_fixture()

    with unittest.TestCase().assertRaisesRegex(ValueError, "diversity_sampling"):
        ActiveLearning(
            adata=adata,
            model_capacity=["gene_a", "gene_b", "gene_c", "gene_d", "gene_val"],
            model_wrapper=DummyModelWrapper(),
            config={
                "budget_plan": [75.0],
                "strategy": "active_diversity_sampling",
                "start_mode": "zero-shot",
                "dataset_cell_type": "target_ct",
                "use_support_data": False,
                "saved_dir": str(tmp_path),
                "model_name": "dummy-wrapper",
                "repeat": 1,
            },
        )


def test_active_learning_rejects_ambiguous_pcc_strategy_name(tmp_path):
    adata = _make_active_learning_fixture()

    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        "diversity_sampling",
    ):
        ActiveLearning(
            adata=adata,
            model_capacity=["gene_a", "gene_b", "gene_c", "gene_d", "gene_val"],
            model_wrapper=DummyModelWrapper(),
            config={
                "budget_plan": [75.0],
                "strategy": "active_diversity_sampling_pcc",
                "start_mode": "zero-shot",
                "dataset_cell_type": "target_ct",
                "use_support_data": False,
                "saved_dir": str(tmp_path),
                "model_name": "dummy-wrapper",
                "repeat": 1,
            },
        )



def test_active_learning_pcc_pert_mean_sampling_uses_trained_perturbation_mean_reference(tmp_path):
    adata = _make_active_learning_correlation_reference_fixture()
    task = ActiveLearning(
        adata=adata,
        model_capacity=["gene_a", "gene_b", "gene_c", "gene_d", "gene_val"],
        model_wrapper=DummyModelWrapper(),
        config={
            "budget_plan": [75.0],
            "strategy": "diversity_sampling",
            "start_mode": "zero-shot",
            "dataset_cell_type": "target_ct",
            "use_support_data": False,
            "saved_dir": str(tmp_path),
            "model_name": "dummy-wrapper",
            "repeat": 1,
        },
    )

    adata_pred = ad.AnnData(
        X=np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 1.0]], dtype=float),
        obs=pd.DataFrame(
            {"perturbation": ["gene_b", "gene_d"]},
            index=["pred_b", "pred_d"],
        ),
        var=adata.var.copy(),
    )

    _source_budget, _result_budget, selected = task.sample(
        adata_pred=adata_pred,
        trained_condition=["gene_a", "gene_c"],
        target_budget=75.0,
        strategy="diversity_sampling",
    )

    assert selected == ["gene_a", "gene_c", "gene_d"]


def test_diversity_sampling_falls_back_to_random_when_predictions_are_uniform(tmp_path):
    # A constant-prediction model (e.g. CMean) gives identical predicted deltas
    # for every candidate, so all diversity scores tie. Selection must then be
    # random rather than a fixed pool-order pick.
    adata = _make_active_learning_fixture()
    task = ActiveLearning(
        adata=adata,
        model_capacity=["gene_a", "gene_b", "gene_c", "gene_d", "gene_val"],
        model_wrapper=DummyModelWrapper(),
        config={
            "budget_plan": [50.0],
            "strategy": "diversity_sampling",
            "start_mode": "zero-shot",
            "dataset_cell_type": "target_ct",
            "use_support_data": False,
            "saved_dir": str(tmp_path),
            "model_name": "dummy-wrapper",
            "repeat": 1,
        },
    )

    candidates = ["gene_a", "gene_b", "gene_c", "gene_d"]
    adata_pred = ad.AnnData(
        X=np.ones((len(candidates), adata.n_vars), dtype=float),
        obs=pd.DataFrame({"perturbation": candidates}, index=[f"pred_{p}" for p in candidates]),
        var=adata.var.copy(),
    )

    first_picks = set()
    for seed in range(10):
        np.random.seed(seed)
        _source, _result, selected = task.sample(
            adata_pred=adata_pred,
            trained_condition=[],
            target_budget=50.0,
            strategy="diversity_sampling",
        )
        assert set(selected).issubset(set(candidates))
        assert len(selected) == 2
        first_picks.add(selected[0])

    # Deterministic pool-order selection would always start with the same
    # perturbation; random tie-breaking yields more than one distinct first pick.
    assert len(first_picks) > 1


def test_representative_sampling_rejects_candidate_count_above_limit(tmp_path):
    adata = _make_active_learning_fixture()
    task = ActiveLearning(
        adata=adata,
        model_capacity=["gene_a", "gene_b", "gene_c", "gene_d", "gene_val"],
        model_wrapper=DummyModelWrapper(),
        config={
            "budget_plan": [50.0],
            "strategy": "representative_sampling",
            "start_mode": "zero-shot",
            "dataset_cell_type": "target_ct",
            "use_support_data": False,
            "saved_dir": str(tmp_path),
            "model_name": "dummy-wrapper",
            "repeat": 1,
            "max_representative_candidates": 1,
        },
    )
    prediction = ad.AnnData(
        X=np.array([[0.0, 1.0], [2.0, 2.0]], dtype=float),
        obs=pd.DataFrame(
            {"perturbation": ["gene_b", "gene_c"]},
            index=["pred_b", "pred_c"],
        ),
        var=adata.var.copy(),
    )

    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        "max_representative_candidates=1",
    ):
        task._select_representative_candidates(
            adata_pred=prediction,
            model_cand=["gene_b", "gene_c"],
            trained_condition=[],
            required_new=1,
        )


def test_active_learning_diversity_sampling_progress_bar_does_not_change_result(tmp_path):
    adata = _make_active_learning_fixture()
    task = ActiveLearning(
        adata=adata,
        model_capacity=["gene_a", "gene_b", "gene_c", "gene_d", "gene_val"],
        model_wrapper=DummyModelWrapper(),
        config={
            "budget_plan": [50.0],
            "strategy": "diversity_sampling",
            "start_mode": "zero-shot",
            "dataset_cell_type": "target_ct",
            "use_support_data": False,
            "saved_dir": str(tmp_path),
            "model_name": "dummy-wrapper",
            "repeat": 1,
        },
    )

    adata_pred = ad.AnnData(
        X=np.array([[1.5, 0.0], [0.0, 1.0], [3.0, 0.0]], dtype=float),
        obs=pd.DataFrame(
            {"perturbation": ["gene_b", "gene_c", "gene_d"]},
            index=["pred_b", "pred_c", "pred_d"],
        ),
        var=adata.var.copy(),
    )

    progress_calls = []

    def _fake_tqdm(iterable, **kwargs):
        items = list(iterable)
        progress_calls.append({"items": items, "kwargs": kwargs})
        return items

    # Same seed with and without the tqdm wrapper must give the same selection;
    # the progress bar only wraps iteration and must not alter the result.
    np.random.seed(0)
    with patch("alive.tasks.active_learning.tqdm", side_effect=_fake_tqdm):
        _source_budget, _result_budget, selected_wrapped = task.sample(
            adata_pred=adata_pred,
            trained_condition=["gene_a"],
            target_budget=50.0,
            strategy="diversity_sampling",
        )

    np.random.seed(0)
    _source_budget, _result_budget, selected_plain = task.sample(
        adata_pred=adata_pred,
        trained_condition=["gene_a"],
        target_budget=50.0,
        strategy="diversity_sampling",
    )

    assert selected_wrapped == selected_plain
    assert selected_wrapped[0] == "gene_a"
    assert len(selected_wrapped) == 2
    assert len(progress_calls) == 1
    assert progress_calls[0]["items"] == [0]
    assert "diversity_sampling" in progress_calls[0]["kwargs"]["desc"]


def test_active_learning_sample_falls_back_to_blind_spots_when_predictions_insufficient(tmp_path):
    adata = _make_active_learning_fixture()
    task = ActiveLearning(
        adata=adata,
        model_capacity=["gene_a", "gene_b", "gene_c", "gene_d", "gene_val"],
        model_wrapper=DummyModelWrapper(),
        config={
            "budget_plan": [100.0],
            "strategy": "diversity_sampling",
            "start_mode": "zero-shot",
            "dataset_cell_type": "target_ct",
            "use_support_data": False,
            "saved_dir": str(tmp_path),
            "model_name": "dummy-wrapper",
            "repeat": 1,
        },
    )

    adata_pred = ad.AnnData(
        X=np.array([[0.0, 1.0], [2.0, 2.0]], dtype=float),
        obs=pd.DataFrame({"perturbation": ["gene_b", "gene_c"]}, index=["pred_b", "pred_c"]),
        var=adata.var.copy(),
    )

    _source_budget, result_budget, selected = task.sample(
        adata_pred=adata_pred,
        trained_condition=["gene_a"],
        target_budget=100.0,
        strategy="diversity_sampling",
    )

    assert result_budget == 100.0
    assert set(selected) == {"gene_a", "gene_b", "gene_c", "gene_d"}


def test_active_learning_random_sampling_is_iterative_and_prefers_model_visible(tmp_path):
    adata = _make_active_learning_fixture()
    task = ActiveLearning(
        adata=adata,
        model_capacity=["gene_a", "gene_b", "gene_c", "gene_d", "gene_val"],
        model_wrapper=DummyModelWrapper(),
        config={
            "budget_plan": [50.0],
            "strategy": "random_sampling",
            "start_mode": "zero-shot",
            "dataset_cell_type": "target_ct",
            "use_support_data": False,
            "saved_dir": str(tmp_path),
            "model_name": "dummy-wrapper",
            "repeat": 1,
        },
    )

    adata_pred = ad.AnnData(
        X=np.array([[0.0, 1.0]], dtype=float),
        obs=pd.DataFrame({"perturbation": ["gene_b"]}, index=["pred_b"]),
        var=adata.var.copy(),
    )

    source_budget, result_budget, selected = task.sample(
        adata_pred=adata_pred,
        trained_condition=["gene_a"],
        target_budget=50.0,
        strategy="random_sampling",
    )

    assert source_budget == 25.0
    assert result_budget == 50.0
    assert set(selected) == {"gene_a", "gene_b"}


def test_active_learning_run_uses_test_only_runtime_evaluation(tmp_path):
    adata = _make_in_distribution_fixture()
    model_wrapper = DummyModelWrapper()
    atlas_dir = _write_runtime_metric_reference_files(
        tmp_path=tmp_path,
        dataset_cell_type="target_ct",
        hvg_genes=["g1"],
        de_dict={"gene_a": ["g1"], "gene_b": ["g2"], "gene_val": ["g1"]},
    )

    task = ActiveLearning(
        adata=adata,
        model_capacity=["gene_a", "gene_b", "gene_val"],
        model_wrapper=model_wrapper,
        config={
            "budget_plan": [50.0],
            "strategy": "random_sampling",
            "start_mode": "zero-shot",
            "dataset_cell_type": "target_ct",
            "use_support_data": False,
            "saved_dir": str(tmp_path),
            "atlas_dir": str(atlas_dir),
            "model_name": "dummy-wrapper",
            "repeat": 1,
            "predefined_condition": [{"50.0": ["gene_a"]}],
        },
    )

    run_result = task.run(repeat=0)

    assert (Path(tmp_path) / "repeat_0_50.0_test_pred_bulk.csv.gz").exists()
    assert (Path(tmp_path) / "repeat_0_condition_record.json").exists()
    assert not (Path(tmp_path) / "repeat_0_test_achievement_record.json").exists()
    assert not (Path(tmp_path) / "repeat_0_val_achievement_record.json").exists()

    runtime_rows = run_result["runtime_metric_rows"]
    assert len(runtime_rows) == 15
    assert {row["context"] for row in runtime_rows} == {"target_ct"}
    assert {row["set"] for row in runtime_rows} == {"test"}

    round_key = model_wrapper.fit_calls[0]
    assert model_wrapper.predict_calls == [(round_key, "test")]


class ActiveLearningDiversityImplementationTests(unittest.TestCase):
    def test_goal_types_use_canonical_task_names(self):
        self.assertEqual(InDistribution.goal_type, "InDistribution")
        self.assertEqual(OutOfDistribution.goal_type, "OutOfDistribution")
        self.assertEqual(ActiveLearning.goal_type, "ActiveLearning")

    def test_evaluate_exposes_only_supported_parameters(self):
        signature = inspect.signature(InDistribution.evaluate)

        self.assertEqual(
            list(signature.parameters),
            ["self", "adata_pred", "split", "trained_condition"],
        )
