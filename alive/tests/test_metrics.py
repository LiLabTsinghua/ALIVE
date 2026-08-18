import json
import unittest

import numpy as np
import pandas as pd

from alive.metrics import (
    GENE_SET_NAMES,
    METRIC_NAMES,
    build_empty_metric_summary,
    compute_runtime_metric_summary,
    load_runtime_gene_sets,
    metric_record_key,
)


def _make_bulk_metric_fixture():
    gene_names = ["g1", "g2"]
    pred_bulk = pd.DataFrame(
        [[2.0, 1.0], [1.0, 3.0]],
        index=["gene_a", "gene_b"],
        columns=gene_names,
    )
    true_bulk = pd.DataFrame(
        [[2.0, 1.0], [0.0, 2.0]],
        index=["gene_a", "gene_b"],
        columns=gene_names,
    )
    ref_control = np.array([0.0, 0.0], dtype=float)
    ref_pert_mean = np.array([1.0, 1.5], dtype=float)
    hvg_genes = ["g1"]
    de_dict = {"gene_a": ["g1"], "gene_b": ["g2"]}
    return pred_bulk, true_bulk, ref_control, ref_pert_mean, hvg_genes, de_dict


def test_build_empty_metric_summary_returns_expected_nested_keys():
    summary = build_empty_metric_summary()

    assert set(summary.keys()) == set(GENE_SET_NAMES)
    assert set(summary["ALL"].keys()) == set(METRIC_NAMES)
    assert np.isnan(summary["DE"]["MAE"])
    assert np.isnan(summary["ALL"]["Systema Bias"])


def test_metric_names_and_csv_keys_use_only_the_canonical_five_metrics():
    assert METRIC_NAMES == (
        "MSE",
        "MAE",
        "Correlation",
        "Systema Correlation",
        "Systema Bias",
    )
    assert [metric_record_key(metric_name) for metric_name in METRIC_NAMES] == [
        "mse",
        "mae",
        "correlation",
        "systema_correlation",
        "systema_bias",
    ]

    with unittest.TestCase().assertRaises(ValueError):
        metric_record_key("PCC-overall")


def test_compute_runtime_metric_summary_returns_expected_nested_structure():
    pred_bulk, true_bulk, ref_control, ref_pert_mean, hvg_genes, de_dict = _make_bulk_metric_fixture()

    summary = compute_runtime_metric_summary(
        pred_bulk=pred_bulk,
        true_bulk=true_bulk,
        ref_control=ref_control,
        ref_pert_mean=ref_pert_mean,
        hvg_genes=hvg_genes,
        de_dict=de_dict,
    )

    assert set(summary.keys()) == {"ALL", "HVG", "DE"}
    assert set(summary["ALL"].keys()) == set(METRIC_NAMES)
    assert summary["ALL"]["MSE"] == 0.5
    assert summary["ALL"]["MAE"] == 0.5
    assert summary["DE"]["MSE"] == 0.5
    assert summary["DE"]["MAE"] == 0.5
    assert summary["ALL"]["Systema Bias"] == 0.0


def test_compute_runtime_metric_summary_handles_zero_variance_vectors_without_crashing():
    pred_bulk = pd.DataFrame([[1.0, 1.0]], index=["gene_a"], columns=["g1", "g2"])
    true_bulk = pd.DataFrame([[1.0, 1.0]], index=["gene_a"], columns=["g1", "g2"])

    summary = compute_runtime_metric_summary(
        pred_bulk=pred_bulk,
        true_bulk=true_bulk,
        ref_control=np.array([0.0, 0.0], dtype=float),
        ref_pert_mean=np.array([1.0, 1.0], dtype=float),
        hvg_genes=["g1"],
        de_dict={"gene_a": ["g1", "g2"]},
    )

    assert summary["ALL"]["Correlation"] == 0.0
    assert summary["ALL"]["Systema Correlation"] == 0.0
    assert summary["ALL"]["Systema Bias"] == 0.0
    assert summary["ALL"]["MSE"] == 0.0
    assert summary["ALL"]["MAE"] == 0.0


def test_compute_runtime_metric_summary_returns_nan_without_common_perturbations():
    pred_bulk = pd.DataFrame([[1.0, 2.0]], index=["gene_a"], columns=["g1", "g2"])
    true_bulk = pd.DataFrame([[1.0, 2.0]], index=["gene_b"], columns=["g1", "g2"])

    summary = compute_runtime_metric_summary(
        pred_bulk=pred_bulk,
        true_bulk=true_bulk,
        ref_control=np.array([0.0, 0.0], dtype=float),
        ref_pert_mean=np.array([1.0, 1.0], dtype=float),
        hvg_genes=["g1"],
        de_dict={"gene_b": ["g2"]},
    )

    for metric_values in summary.values():
        assert all(np.isnan(value) for value in metric_values.values())


def test_compute_runtime_metric_summary_returns_nan_for_empty_gene_sets():
    pred_bulk = pd.DataFrame([[1.0, 2.0]], index=["gene_a"], columns=["g1", "g2"])
    true_bulk = pd.DataFrame([[1.0, 2.0]], index=["gene_a"], columns=["g1", "g2"])

    summary = compute_runtime_metric_summary(
        pred_bulk=pred_bulk,
        true_bulk=true_bulk,
        ref_control=np.array([0.0, 0.0], dtype=float),
        ref_pert_mean=np.array([1.0, 1.0], dtype=float),
        hvg_genes=[],
        de_dict={},
    )

    assert all(np.isnan(value) for value in summary["HVG"].values())
    assert all(np.isnan(value) for value in summary["DE"].values())
    assert summary["ALL"]["MSE"] == 0.0
    assert summary["ALL"]["MAE"] == 0.0


def test_load_runtime_gene_sets_reads_hvg_and_de_json(tmp_path):
    atlas_dir = tmp_path / "atlas"
    atlas_dir.mkdir()
    (atlas_dir / "target_ct_hvg.json").write_text(json.dumps(["g1", "g2"]), encoding="utf-8")
    (atlas_dir / "target_ct_de.json").write_text(
        json.dumps({"gene_a": ["g1"], "gene_b": ["g2"]}),
        encoding="utf-8",
    )

    hvg_genes, de_dict = load_runtime_gene_sets(
        atlas_dir=str(atlas_dir),
        dataset_cell_type="target_ct",
    )

    assert hvg_genes == ["g1", "g2"]
    assert de_dict == {"gene_a": ["g1"], "gene_b": ["g2"]}
