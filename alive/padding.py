"""Prediction padding and expression post-processing utilities."""

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm


def pad_missing_perturbations(
    adata_pred,
    perts_to_pred,
    adata_ctrl,
    mode,
    target_size=300,
):
    """Pad missing perturbations with control-derived placeholders.

    Args:
        adata_pred: Predicted ``AnnData`` to extend.
        perts_to_pred: Perturbations that should be present after padding.
        adata_ctrl: Control cells used to synthesize placeholder profiles.
        mode: ``"bulk"`` (one control-mean cell per missing perturbation) or
            ``"single-cell"`` (``target_size`` sampled control cells each).
        target_size: Number of control cells sampled per perturbation in
            ``"single-cell"`` mode.

    Returns:
        anndata.AnnData: ``adata_pred`` with placeholders appended (or the
        original object when nothing is missing).

    Raises:
        ValueError: If ``mode`` is not ``"bulk"`` or ``"single-cell"``.
    """
    if mode not in {"bulk", "single-cell"}:
        raise ValueError(f"Unknown mode: {mode}")

    pred_pert_list = set(adata_pred.obs["perturbation"].astype(str))
    target_perts = {str(pert) for pert in perts_to_pred}
    missing_perts = sorted(target_perts - pred_pert_list)

    if len(target_perts) > 0:
        predictable_pct = 100 - (len(missing_perts) / len(target_perts) * 100)
    else:
        predictable_pct = 0.0

    if len(missing_perts) == 0:
        print("No missing perturbations found. Predictable=100.00%. Padding skipped.")
        return adata_pred

    print(f"Found {len(missing_perts)} missing perts. Predictable={predictable_pct:.2f}%.")

    is_pred_dense = isinstance(adata_pred.X, np.ndarray)
    target_dtype = adata_pred.X.dtype

    all_padded_X = []
    all_padded_obs = []

    if mode == "bulk":
        print("Using 'control_mean' padding mode (1 cell per missing pert).")
        if sp.issparse(adata_ctrl.X):
            ctrl_mean_vec = np.array(adata_ctrl.X.mean(axis=0)).reshape(1, -1)
        else:
            ctrl_mean_vec = adata_ctrl.X.mean(axis=0, keepdims=True)

        ctrl_mean_vec = ctrl_mean_vec.astype(target_dtype)
        if not is_pred_dense:
            ctrl_mean_vec = sp.csr_matrix(ctrl_mean_vec)

        obs_template = adata_ctrl.obs.iloc[[0]].copy()

        for pert in tqdm(missing_perts, desc="Padding Mean"):
            all_padded_X.append(ctrl_mean_vec)

            tmp_obs = obs_template.copy()
            tmp_obs["perturbation"] = pert
            tmp_obs["is_padded"] = True
            tmp_obs.index = [f"pad_mean_{pert}"]
            all_padded_obs.append(tmp_obs)

    elif mode == "single-cell":
        ctrl_X = adata_ctrl.X
        ctrl_obs_template = adata_ctrl.obs.copy()

        for pert in tqdm(missing_perts, desc="Padding Cells"):
            idx = np.random.choice(adata_ctrl.n_obs, size=target_size, replace=True)
            chunk_X = ctrl_X[idx]

            if is_pred_dense and sp.issparse(chunk_X):
                chunk_X = chunk_X.toarray()

            all_padded_X.append(chunk_X.astype(target_dtype))

            tmp_obs = ctrl_obs_template.iloc[idx].copy()
            tmp_obs["perturbation"] = pert
            tmp_obs["is_padded"] = True
            tmp_obs.index = [f"pad_{pert}_{i}" for i in range(target_size)]
            all_padded_obs.append(tmp_obs)

    else:
        raise ValueError(f"Unexpected validated mode: {mode}")

    if is_pred_dense:
        new_X_padded = np.vstack(all_padded_X)
    else:
        new_X_padded = sp.vstack(all_padded_X)

    new_obs_padded = pd.concat(all_padded_obs)
    adata_padding_all = ad.AnnData(
        X=new_X_padded,
        obs=new_obs_padded,
        var=adata_ctrl.var.copy(),
    )

    if "is_padded" not in adata_pred.obs.columns:
        adata_pred.obs["is_padded"] = False

    final_adata = ad.concat([adata_pred, adata_padding_all], merge="same", uns_merge=None)
    final_adata.obs_names_make_unique()

    print(f"Padding complete. Total cells: {final_adata.n_obs}")
    return final_adata


def clip_adata_x_nonnegative(adata_pred):
    """Clamp ``adata_pred.X`` to be non-negative in place.

    Args:
        adata_pred: ``AnnData`` whose expression matrix is clipped at zero.
    """
    if sp.issparse(adata_pred.X):
        adata_pred.X.data = np.clip(adata_pred.X.data, a_min=0, a_max=None)
    else:
        adata_pred.X = np.clip(adata_pred.X, a_min=0, a_max=None)
