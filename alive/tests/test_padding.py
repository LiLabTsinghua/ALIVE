import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from alive.padding import clip_adata_x_nonnegative, pad_missing_perturbations


def _make_ctrl_adata(n_obs=3, dense=True):
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=float,
    )[:n_obs]
    if not dense:
        X = sp.csr_matrix(X)

    obs = pd.DataFrame(
        {
            "perturbation": ["control"] * n_obs,
            "dataset_cell_type": ["target_ct"] * n_obs,
            "split": ["control"] * n_obs,
        },
        index=[f"ctrl_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=["g1", "g2"])
    return ad.AnnData(X=X, obs=obs, var=var)


def test_padding_bulk_mode_adds_one_control_mean_row_per_missing_perturbation():
    adata_ctrl = _make_ctrl_adata()
    adata_pred = ad.AnnData(
        X=np.array([[10.0, 11.0]], dtype=float),
        obs=pd.DataFrame({"perturbation": ["gene_a"]}, index=["pred_0"]),
        var=adata_ctrl.var.copy(),
    )

    result = pad_missing_perturbations(
        adata_pred=adata_pred,
        perts_to_pred={"gene_a", "gene_b", "gene_c"},
        adata_ctrl=adata_ctrl,
        mode="bulk",
    )

    assert result.n_obs == 3
    assert set(result.obs["perturbation"]) == {"gene_a", "gene_b", "gene_c"}
    assert result.obs["is_padded"].sum() == 2


def test_padding_processes_missing_perturbations_in_stable_sorted_order():
    adata_ctrl = _make_ctrl_adata()
    adata_pred = ad.AnnData(
        X=np.array([[10.0, 11.0]], dtype=float),
        obs=pd.DataFrame({"perturbation": ["gene_a"]}, index=["pred_0"]),
        var=adata_ctrl.var.copy(),
    )

    result = pad_missing_perturbations(
        adata_pred=adata_pred,
        perts_to_pred={"gene_c", "gene_a", "gene_b"},
        adata_ctrl=adata_ctrl,
        mode="bulk",
    )

    assert result.obs["perturbation"].astype(str).tolist() == [
        "gene_a",
        "gene_b",
        "gene_c",
    ]


def test_padding_rejects_invalid_mode_even_when_nothing_is_missing():
    adata_ctrl = _make_ctrl_adata()
    adata_pred = ad.AnnData(
        X=np.array([[10.0, 11.0]], dtype=float),
        obs=pd.DataFrame({"perturbation": ["gene_a"]}, index=["pred_0"]),
        var=adata_ctrl.var.copy(),
    )

    with np.testing.assert_raises(ValueError):
        pad_missing_perturbations(
            adata_pred=adata_pred,
            perts_to_pred={"gene_a"},
            adata_ctrl=adata_ctrl,
            mode="invalid",
        )


def test_padding_single_cell_mode_adds_target_size_rows_per_missing_perturbation():
    adata_ctrl = _make_ctrl_adata()
    adata_pred = ad.AnnData(
        X=np.array([[10.0, 11.0]], dtype=float),
        obs=pd.DataFrame({"perturbation": ["gene_a"]}, index=["pred_0"]),
        var=adata_ctrl.var.copy(),
    )

    result = pad_missing_perturbations(
        adata_pred=adata_pred,
        perts_to_pred={"gene_a", "gene_b"},
        adata_ctrl=adata_ctrl,
        mode="single-cell",
        target_size=4,
    )

    assert result.n_obs == 5
    assert (result.obs["perturbation"] == "gene_b").sum() == 4
    assert result.obs["is_padded"].sum() == 4


def test_padding_bulk_mode_preserves_sparse_prediction_layout():
    adata_ctrl = _make_ctrl_adata(dense=False)
    adata_pred = ad.AnnData(
        X=sp.csr_matrix(np.array([[10.0, 11.0]], dtype=float)),
        obs=pd.DataFrame({"perturbation": ["gene_a"]}, index=["pred_0"]),
        var=adata_ctrl.var.copy(),
    )

    result = pad_missing_perturbations(
        adata_pred=adata_pred,
        perts_to_pred={"gene_a", "gene_b"},
        adata_ctrl=adata_ctrl,
        mode="bulk",
    )

    assert sp.issparse(result.X)


def test_clip_adata_x_nonnegative_clamps_dense_inputs():
    adata_pred = ad.AnnData(
        X=np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=float),
        obs=pd.DataFrame(index=["a", "b"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )

    clip_adata_x_nonnegative(adata_pred)

    assert np.array_equal(adata_pred.X, np.array([[0.0, 2.0], [3.0, 0.0]], dtype=float))


def test_clip_adata_x_nonnegative_clamps_sparse_inputs():
    adata_pred = ad.AnnData(
        X=sp.csr_matrix(np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=float)),
        obs=pd.DataFrame(index=["a", "b"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    )

    clip_adata_x_nonnegative(adata_pred)

    assert np.array_equal(adata_pred.X.toarray(), np.array([[0.0, 2.0], [3.0, 0.0]], dtype=float))
