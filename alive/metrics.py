"""Canonical ALIVE runtime metric definitions and computation helpers."""

import json
from pathlib import Path

import numpy as np


METRIC_NAMES = (
    "MSE",
    "MAE",
    "Correlation",
    "Systema Correlation",
    "Systema Bias",
)
GENE_SET_NAMES = ("ALL", "HVG", "DE")

# These are the only metrics emitted by runtime and offline evaluation. The
# human-readable names are used in console summaries; the values below are the
# canonical snake_case names written to long-format CSV files.
METRIC_RECORD_KEYS = {
    "MSE": "mse",
    "MAE": "mae",
    "Correlation": "correlation",
    "Systema Correlation": "systema_correlation",
    "Systema Bias": "systema_bias",
}


def metric_record_key(metric_name):
    """Return the CSV record key for a metric name.

    Args:
        metric_name: Human-readable canonical metric name.

    Returns:
        str: Snake-case metric key written to runtime CSV files.

    Raises:
        ValueError: If ``metric_name`` is not one of the canonical five metrics.
    """
    if metric_name not in METRIC_RECORD_KEYS:
        raise ValueError(f"Unsupported runtime metric name: {metric_name}")
    return METRIC_RECORD_KEYS[metric_name]


def pairwise_pcc_distance(candidate_mat, reference_mat):
    """Compute pairwise ``1 - PCC`` distances between candidate and reference rows.

    Zero-variance rows are treated as zero-correlation rows so the distance stays
    finite. Used by the active sampling strategies.

    Args:
        candidate_mat: ``(n_candidates, n_genes)`` matrix.
        reference_mat: ``(n_references, n_genes)`` matrix.

    Returns:
        numpy.ndarray: ``(n_candidates, n_references)`` distance matrix.
    """
    candidate_centered = candidate_mat - candidate_mat.mean(axis=1, keepdims=True)
    reference_centered = reference_mat - reference_mat.mean(axis=1, keepdims=True)
    candidate_norms = np.linalg.norm(candidate_centered, axis=1)
    reference_norms = np.linalg.norm(reference_centered, axis=1)

    dot = candidate_centered @ reference_centered.T
    denom = candidate_norms[:, None] * reference_norms[None, :]
    valid = (candidate_norms[:, None] > 0.0) & (reference_norms[None, :] > 0.0)
    corr = np.where(valid, np.divide(dot, np.where(denom == 0.0, 1.0, denom)), 0.0)
    return 1.0 - np.clip(corr, -1.0, 1.0)


def build_empty_metric_summary():
    """Return an undefined metric summary for an empty evaluation.

    ``NaN`` distinguishes missing evidence from a genuine zero error. Callers
    keep the complete metric schema so runtime and rebuilt CSV files remain
    structurally consistent even when no perturbation can be evaluated.

    Returns:
        dict: Canonical gene-set and metric mapping filled with ``NaN``.
    """
    return {
        gene_set: {metric_name: float("nan") for metric_name in METRIC_NAMES}
        for gene_set in GENE_SET_NAMES
    }


def load_runtime_gene_sets(atlas_dir, dataset_cell_type):
    """Load the HVG list and per-perturbation DE gene sets for a context.

    Args:
        atlas_dir: Directory holding ``{dataset_cell_type}_hvg.json`` and
            ``{dataset_cell_type}_de.json``.
        dataset_cell_type: Target context identifier.

    Returns:
        tuple: ``(hvg_genes, de_dict)`` with stringified gene names.

    Raises:
        ValueError: If ``atlas_dir`` is ``None``.
        TypeError: If the HVG file is not a list or the DE file is not an object.
    """
    if atlas_dir is None:
        raise ValueError("Runtime evaluation requires config['atlas_dir'].")

    atlas_root = Path(atlas_dir)
    hvg_path = atlas_root / f"{dataset_cell_type}_hvg.json"
    de_path = atlas_root / f"{dataset_cell_type}_de.json"

    with open(hvg_path, "r", encoding="utf-8") as f:
        hvg_genes = json.load(f)

    with open(de_path, "r", encoding="utf-8") as f:
        de_dict = json.load(f)

    if not isinstance(hvg_genes, list):
        raise TypeError(f"{hvg_path} must contain a JSON list.")

    if not isinstance(de_dict, dict):
        raise TypeError(f"{de_path} must contain a JSON object.")

    normalized_de_dict = {}
    for pert, genes in de_dict.items():
        if not isinstance(genes, list):
            raise TypeError(f"DE entry for perturbation {pert!r} must be a JSON list.")
        normalized_de_dict[str(pert)] = [str(gene) for gene in genes]

    return [str(gene) for gene in hvg_genes], normalized_de_dict


def _safe_pearsonr(a, b):
    """Compute Pearson correlation with stable degenerate-input behavior.

    Args:
        a: First numeric vector.
        b: Second numeric vector.

    Returns:
        float: Pearson correlation, or ``0.0`` for empty or zero-variance input.

    Raises:
        ValueError: If the input vectors have different nonzero lengths.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()

    if a.size == 0 or b.size == 0:
        return 0.0

    if a.size != b.size:
        raise ValueError(f"Pearson inputs must have the same length. Got {a.size} and {b.size}.")

    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0

    corr = np.corrcoef(a, b)[0, 1]
    if np.isnan(corr):
        return 0.0

    return float(corr)


def _mean_or_nan(values):
    """Return the mean of ``values``, or ``NaN`` when no evidence exists.

    Args:
        values: Numeric values to average.

    Returns:
        float: Arithmetic mean, or ``NaN`` for an empty collection.
    """
    if not values:
        return float("nan")
    return float(np.mean(values))


def _validate_bulk_frames(pred_bulk, true_bulk):
    """Ensure prediction and truth bulk tables share an identical gene order.

    Args:
        pred_bulk: Predicted pseudobulk table.
        true_bulk: Ground-truth pseudobulk table.

    Raises:
        ValueError: If the gene columns or their order differ.
    """
    if pred_bulk.columns.tolist() != true_bulk.columns.tolist():
        raise ValueError("Prediction and truth bulk tables must have identical gene order.")


def compute_runtime_metric_summary(
    pred_bulk,
    true_bulk,
    ref_control,
    ref_pert_mean,
    hvg_genes,
    de_dict,
):
    """Compute the five canonical metrics for one round's predictions.

    ``Correlation`` compares predicted and true deltas from control. ``Systema
    Correlation`` compares their residuals from the mean perturbation profile.
    ``Systema Bias`` is the difference between predicted and true correlations
    with that mean perturbation direction. MSE and MAE use the expression
    profiles directly.

    Args:
        pred_bulk: Predicted pseudobulk, perturbations x genes.
        true_bulk: Ground-truth pseudobulk with the same gene order.
        ref_control: Control reference profile, length ``n_genes``.
        ref_pert_mean: Mean non-control perturbation profile, length ``n_genes``.
        hvg_genes: Highly variable gene names defining the HVG set.
        de_dict: Mapping of perturbation to its differentially expressed genes.

    Returns:
        dict: ``{gene_set: {metric_name: value}}`` over ALL/HVG/DE.

    Raises:
        ValueError: If gene order or reference lengths are inconsistent.
    """
    _validate_bulk_frames(pred_bulk, true_bulk)

    summary = build_empty_metric_summary()
    common_perts = sorted((set(pred_bulk.index) & set(true_bulk.index)) - {"control"})
    if not common_perts:
        return summary

    gene_names = pred_bulk.columns.tolist()
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    all_idx = list(range(len(gene_names)))
    hvg_idx = [gene_to_idx[gene] for gene in hvg_genes if gene in gene_to_idx]

    collectors = {
        gene_set: {metric_name: [] for metric_name in METRIC_NAMES}
        for gene_set in GENE_SET_NAMES
    }

    ref_control = np.asarray(ref_control, dtype=float)
    ref_pert_mean = np.asarray(ref_pert_mean, dtype=float)
    if ref_control.shape[0] != len(gene_names):
        raise ValueError("ref_control length must match the bulk gene dimension.")
    if ref_pert_mean.shape[0] != len(gene_names):
        raise ValueError("ref_pert_mean length must match the bulk gene dimension.")

    for pert in common_perts:
        true_vec = true_bulk.loc[pert].to_numpy(dtype=float)
        pred_vec = pred_bulk.loc[pert].to_numpy(dtype=float)

        de_idx = [gene_to_idx[gene] for gene in de_dict.get(pert, []) if gene in gene_to_idx]

        for gene_set in GENE_SET_NAMES:
            if gene_set == "ALL":
                gene_idx = all_idx
            elif gene_set == "HVG":
                gene_idx = hvg_idx
            elif gene_set == "DE":
                gene_idx = de_idx
            else:
                raise ValueError(f"Unsupported gene set: {gene_set}")

            if not gene_idx:
                continue

            true_sub = true_vec[gene_idx]
            pred_sub = pred_vec[gene_idx]
            ctrl_sub = ref_control[gene_idx]
            pert_mean_sub = ref_pert_mean[gene_idx]

            collectors[gene_set]["Correlation"].append(
                _safe_pearsonr(pred_sub - ctrl_sub, true_sub - ctrl_sub)
            )
            pred_bias = _safe_pearsonr(pred_sub - ctrl_sub, pert_mean_sub - ctrl_sub)
            true_bias = _safe_pearsonr(true_sub - ctrl_sub, pert_mean_sub - ctrl_sub)
            collectors[gene_set]["Systema Bias"].append(pred_bias - true_bias)
            collectors[gene_set]["Systema Correlation"].append(
                _safe_pearsonr(pred_sub - pert_mean_sub, true_sub - pert_mean_sub)
            )
            collectors[gene_set]["MSE"].append(float(np.mean(np.square(pred_sub - true_sub))))
            collectors[gene_set]["MAE"].append(float(np.mean(np.abs(pred_sub - true_sub))))

    for gene_set in GENE_SET_NAMES:
        for metric_name in METRIC_NAMES:
            summary[gene_set][metric_name] = _mean_or_nan(collectors[gene_set][metric_name])

    return summary
