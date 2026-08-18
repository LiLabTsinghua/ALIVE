"""Cumulative active-learning task and acquisition strategies."""

import numpy as np
from tqdm import tqdm

from ..base import TASK_ALLOWED_STRATEGIES
from ..metrics import pairwise_pcc_distance
from .in_distribution import InDistribution


class ActiveLearning(InDistribution):
    """Iterative active-learning benchmark.

    Unlike :class:`InDistribution`, budgets are cumulative across rounds: each
    round keeps the previously selected perturbations and acquires new ones,
    preferring perturbations the model can currently predict.
    """

    goal_type = "ActiveLearning"

    # ``random_sampling`` is cumulative random acquisition with model-visible
    # candidates selected before blind spots, ensuring acquired perturbations
    # can contribute to the next model-training round whenever possible;
    # ``diversity_sampling`` is the farthest-first (1 - PCC) diversity strategy;
    # Strategies split into two families. The ``coverage-based`` family spreads
    # the budget across the predicted delta space via set-level selection:
    # ``random_sampling`` (model-visible-first random), ``diversity_sampling``
    # (farthest-first),
    # ``representative_sampling`` (facility location). The ``score-based`` family
    # ranks candidates by an independent per-candidate score and takes a top-k
    # slice, concentrating the budget in one region: ``magnitude_sampling``
    # (largest delta norm), ``outlier_sampling`` / ``typical_sampling`` (farthest
    # from / nearest to the candidate centroid).
    allowed_strategies = list(TASK_ALLOWED_STRATEGIES["active_learning"])

    def _validate_config(self):
        """Validate shared settings and the representative-sampling size limit.

        Raises:
            TypeError: If the representative candidate limit is not an integer.
            ValueError: If shared config is invalid or the limit is not positive.
        """
        super()._validate_config()
        max_candidates = self.config.get("max_representative_candidates", 5000)
        if isinstance(max_candidates, bool) or not isinstance(max_candidates, int):
            raise TypeError("max_representative_candidates must be a positive integer.")
        if max_candidates <= 0:
            raise ValueError("max_representative_candidates must be positive.")
        self.max_representative_candidates = max_candidates

    def _budget_count(self, budget):
        """Convert a budget percentage to a perturbation count over the pool.

        Args:
            budget: Budget percentage.

        Returns:
            int: Number of perturbations represented by the budget.
        """
        return int((budget / 100.0) * self.total_pool_count)

    def _compute_candidate_delta_table(self, adata_pred, model_cand, reference_vec):
        """Return predicted perturbation deltas in the original candidate order.

        Keeping the requested order stable makes the greedy selection path
        deterministic when multiple candidates share the same score.

        Args:
            adata_pred: Predicted ``AnnData``.
            model_cand: Model-visible perturbations in preferred order.
            reference_vec: Reference expression vector subtracted from predictions.

        Returns:
            pandas.DataFrame: Predicted delta table in candidate order.
        """
        pred_delta = self._get_bulk_feat_delta(adata_pred, model_cand, reference_vec)
        pred_delta = pred_delta.reindex(model_cand).dropna(how="all")
        return pred_delta

    def _get_reference_delta_vectors(self, trained_condition, reference_vec):
        """Materialize covered perturbations as dense delta vectors.

        These vectors form the current covered set used when scoring
        model-visible candidates.

        Args:
            trained_condition: Previously selected perturbations.
            reference_vec: Reference expression vector used for deltas.

        Returns:
            numpy.ndarray: Dense trained-perturbation delta matrix.
        """
        trained_effective = [pert for pert in trained_condition if pert in self.pool]
        if not trained_effective:
            return np.empty((0, self.adata.n_vars), dtype=float)

        ref_delta = self._get_bulk_feat_delta(self.adata_real, trained_effective, reference_vec)
        if ref_delta.empty:
            return np.empty((0, self.adata.n_vars), dtype=float)

        return ref_delta.to_numpy(dtype=float, copy=False)

    def _get_trained_perturbation_mean_reference(self, trained_condition):
        """Compute the mean pseudobulk profile across already trained perturbations.

        When active learning is still in zero-shot mode and no target-context
        perturbation has been selected yet, the strategy falls back to the
        control reference so the first PCC round remains well-defined.

        Args:
            trained_condition: Previously selected perturbations.

        Returns:
            numpy.ndarray: Mean trained-perturbation profile or control fallback.
        """
        trained_effective = [pert for pert in trained_condition if pert in self.pool]
        if not trained_effective:
            return np.asarray(self.ref_control, dtype=float)

        trained_bulk = self._get_bulk_feat(self.adata_real, trained_effective)
        if trained_bulk.empty:
            return np.asarray(self.ref_control, dtype=float)

        return trained_bulk.mean(axis=0).to_numpy(dtype=float, copy=False)

    @staticmethod
    def _center_rows(mat):
        """Return ``mat`` with each row mean-centered.

        Args:
            mat: Two-dimensional numeric matrix.

        Returns:
            numpy.ndarray: Row-centered matrix.
        """
        return mat - mat.mean(axis=1, keepdims=True)

    @staticmethod
    def _row_norms(mat):
        """Return the L2 norm of each row of ``mat``.

        Args:
            mat: Two-dimensional numeric matrix.

        Returns:
            numpy.ndarray: One norm per matrix row.
        """
        return np.linalg.norm(mat, axis=1)

    def _pairwise_pcc_distance_to_vector(self, candidate_centered, candidate_norms, chosen_vec):
        """Update ``1 - PCC`` distances against one newly selected perturbation.

        The active loop calls this after each greedy choice to update the running
        nearest-distance table in one vectorized pass.

        Args:
            candidate_centered: Row-centered candidate delta matrix.
            candidate_norms: Precomputed L2 norm for each candidate row.
            chosen_vec: Newly selected candidate delta vector.

        Returns:
            numpy.ndarray: ``1 - PCC`` distance from every candidate to the choice.
        """
        chosen_centered = chosen_vec - chosen_vec.mean()
        chosen_norm = np.linalg.norm(chosen_centered)
        corr = np.zeros(candidate_centered.shape[0], dtype=float)
        valid = candidate_norms > 0.0
        if chosen_norm > 0.0 and np.any(valid):
            corr[valid] = (
                candidate_centered[valid] @ chosen_centered
            ) / (candidate_norms[valid] * chosen_norm)
        corr = np.clip(corr, -1.0, 1.0)
        return 1.0 - corr

    def _select_diverse_candidates(
        self,
        adata_pred,
        model_cand,
        trained_condition,
        required_new,
    ):
        """Greedily pick diverse model-visible candidates by farthest ``1 - PCC``.

        Args:
            adata_pred: Predicted ``AnnData`` for model-visible candidates.
            model_cand: Candidate perturbations the model can predict.
            trained_condition: Perturbations already selected.
            required_new: Number of new perturbations to acquire.

        Returns:
            list: Selected candidate perturbations (up to ``required_new``).
        """
        if required_new <= 0 or not model_cand:
            return []

        reference_vec = self._get_trained_perturbation_mean_reference(trained_condition)
        pred_delta = self._compute_candidate_delta_table(adata_pred, model_cand, reference_vec)
        if pred_delta.empty:
            return []

        candidate_names = pred_delta.index.astype(str).tolist()
        candidate_mat = pred_delta.to_numpy(dtype=float, copy=False)
        covered = self._get_reference_delta_vectors(trained_condition, reference_vec)
        selected = []
        active_mask = np.ones(len(candidate_names), dtype=bool)

        candidate_centered = self._center_rows(candidate_mat)
        candidate_centered_norms = self._row_norms(candidate_centered)

        if covered.size == 0:
            nearest_dist = np.einsum("ij,ij->i", candidate_mat, candidate_mat).copy()
        else:
            nearest_dist = pairwise_pcc_distance(candidate_mat, covered).min(axis=1)

        max_steps = min(required_new, len(candidate_names))
        progress = tqdm(
            range(max_steps),
            desc="Greedy selection (diversity_sampling)",
            leave=False,
        )
        for _ in progress:
            if not active_mask.any():
                break

            scores = np.where(active_mask, nearest_dist, -np.inf)
            # Random tie-break: a single best candidate is taken deterministically,
            # but when several tie (e.g. a constant-prediction baseline gives no
            # diversity signal) the choice is made uniformly at random rather than
            # always favoring pool order, so selection degenerates to random
            # sampling instead of a fixed order.
            tied = np.flatnonzero(scores == scores.max())
            best_idx = int(tied[0]) if tied.size == 1 else int(np.random.choice(tied))
            chosen = candidate_names[best_idx]
            selected.append(chosen)

            active_mask[best_idx] = False
            chosen_vec = candidate_mat[best_idx]
            new_dist = self._pairwise_pcc_distance_to_vector(
                candidate_centered=candidate_centered,
                candidate_norms=candidate_centered_norms,
                chosen_vec=chosen_vec,
            )
            nearest_dist = np.minimum(nearest_dist, new_dist)

        return selected

    def _select_representative_candidates(
        self,
        adata_pred,
        model_cand,
        trained_condition,
        required_new,
    ):
        """Greedily pick representative model-visible candidates by facility location.

        Mirror of :meth:`_select_diverse_candidates`: where farthest-first
        maximizes the distance to the selected set (favoring outliers), this
        greedily minimizes the total ``1 - PCC`` distance from every candidate to
        its nearest selected medoid, so the chosen set sits in dense, typical
        regions of the predicted delta space.

        Args:
            adata_pred: Predicted ``AnnData`` for model-visible candidates.
            model_cand: Candidate perturbations the model can predict.
            trained_condition: Perturbations already selected.
            required_new: Number of new perturbations to acquire.

        Returns:
            list: Selected candidate perturbations (up to ``required_new``).

        Raises:
            ValueError: If the candidate count exceeds the configured limit.
        """
        if required_new <= 0 or not model_cand:
            return []

        reference_vec = self._get_trained_perturbation_mean_reference(trained_condition)
        pred_delta = self._compute_candidate_delta_table(adata_pred, model_cand, reference_vec)
        if pred_delta.empty:
            return []

        candidate_names = pred_delta.index.astype(str).tolist()
        n_candidates = len(candidate_names)
        if n_candidates > self.max_representative_candidates:
            raise ValueError(
                "representative_sampling candidate count exceeds "
                f"max_representative_candidates={self.max_representative_candidates}. "
                f"Got {n_candidates} candidates. Increase the limit explicitly "
                "or use another sampling strategy."
            )

        candidate_mat = pred_delta.to_numpy(dtype=float, copy=False)
        # Full pairwise distance among candidates; entry ``(i, j)`` is the cost of
        # serving candidate ``i`` by medoid ``j``.
        dist_matrix = pairwise_pcc_distance(candidate_mat, candidate_mat)

        nearest_dist = np.full(n_candidates, np.inf)
        active_mask = np.ones(n_candidates, dtype=bool)
        selected = []

        max_steps = min(required_new, n_candidates)
        progress = tqdm(
            range(max_steps),
            desc="Greedy selection (representative_sampling)",
            leave=False,
        )
        for _ in progress:
            if not active_mask.any():
                break

            # Total nearest-medoid distance that would remain after adding each
            # candidate. The first pick (all distances infinite) reduces to the
            # global medoid; later picks lower the residual coverage cost.
            post_total = np.minimum(nearest_dist[:, None], dist_matrix).sum(axis=0)
            post_total = np.where(active_mask, post_total, np.inf)

            # Random tie-break mirrors diversity: keep selection from collapsing to
            # pool order when several candidates yield the same residual cost.
            tied = np.flatnonzero(post_total == post_total.min())
            best_idx = int(tied[0]) if tied.size == 1 else int(np.random.choice(tied))
            selected.append(candidate_names[best_idx])

            active_mask[best_idx] = False
            nearest_dist = np.minimum(nearest_dist, dist_matrix[:, best_idx])

        return selected

    def _select_magnitude_candidates(
        self,
        adata_pred,
        model_cand,
        trained_condition,
        required_new,
    ):
        """Pick model-visible candidates with the strongest predicted effect.

        Orthogonal to the geometry-based strategies: candidates are ranked by the
        L2 norm of their predicted delta (largest first), preferring the
        perturbations the model expects to move expression the most.

        Args:
            adata_pred: Predicted ``AnnData`` for model-visible candidates.
            model_cand: Candidate perturbations the model can predict.
            trained_condition: Perturbations already selected.
            required_new: Number of new perturbations to acquire.

        Returns:
            list: Selected candidate perturbations (up to ``required_new``).
        """
        if required_new <= 0 or not model_cand:
            return []

        reference_vec = self._get_trained_perturbation_mean_reference(trained_condition)
        pred_delta = self._compute_candidate_delta_table(adata_pred, model_cand, reference_vec)
        if pred_delta.empty:
            return []

        candidate_names = pred_delta.index.astype(str).tolist()
        magnitudes = self._row_norms(pred_delta.to_numpy(dtype=float, copy=False))
        # Sort by descending magnitude; ``kind="stable"`` keeps the original
        # candidate order as a deterministic tie-break.
        order = np.argsort(-magnitudes, kind="stable")
        take = min(required_new, len(candidate_names))
        return [candidate_names[i] for i in order[:take]]

    def _select_by_centroid_distance(
        self,
        adata_pred,
        model_cand,
        trained_condition,
        required_new,
        farthest,
    ):
        """Pick model-visible candidates by ``1 - PCC`` distance to the candidate centroid.

        Shared top-k scorer for the two region-focused twins of the greedy
        geometry strategies: it ranks candidates by how far their predicted delta
        sits from the mean predicted delta (the typical perturbation), then takes
        a top-k slice without the farthest-first / facility-location repulsion, so
        the budget concentrates in one region instead of spreading out.

        Args:
            adata_pred: Predicted ``AnnData`` for model-visible candidates.
            model_cand: Candidate perturbations the model can predict.
            trained_condition: Perturbations already selected.
            required_new: Number of new perturbations to acquire.
            farthest: If ``True`` keep the most atypical (largest distance)
                candidates (``outlier_sampling``); if ``False`` keep the most
                typical (smallest distance) candidates (``typical_sampling``).

        Returns:
            list: Selected candidate perturbations (up to ``required_new``).
        """
        if required_new <= 0 or not model_cand:
            return []

        reference_vec = self._get_trained_perturbation_mean_reference(trained_condition)
        pred_delta = self._compute_candidate_delta_table(adata_pred, model_cand, reference_vec)
        if pred_delta.empty:
            return []

        candidate_names = pred_delta.index.astype(str).tolist()
        candidate_mat = pred_delta.to_numpy(dtype=float, copy=False)
        centroid = candidate_mat.mean(axis=0, keepdims=True)
        # Same 1 - PCC metric as the greedy geometry strategies, measured against
        # the single centroid row.
        dist = pairwise_pcc_distance(candidate_mat, centroid).ravel()

        # ``kind="stable"`` keeps the original candidate order as a deterministic
        # tie-break (mirrors ``_select_magnitude_candidates``).
        order = np.argsort(-dist if farthest else dist, kind="stable")
        take = min(required_new, len(candidate_names))
        return [candidate_names[i] for i in order[:take]]

    def _sample_from_model_visible(
        self,
        adata_pred,
        strategy,
        model_cand,
        trained_condition,
        required_new,
    ):
        """Acquire new perturbations from the model-visible subset first.

        Active diversity strategies always prioritize perturbations that already
        have model predictions. Blind spots are filled later by :meth:`sample`.

        Args:
            adata_pred: Predicted ``AnnData`` for model-visible candidates.
            strategy: Sampling strategy name.
            model_cand: Model-visible candidate perturbations.
            trained_condition: Perturbations already selected.
            required_new: Number of new perturbations to acquire.

        Returns:
            list: Newly selected model-visible perturbations.

        Raises:
            ValueError: If ``strategy`` is unsupported.
        """
        if required_new <= 0 or not model_cand:
            return []

        if strategy == "random_sampling":
            picked = np.random.choice(
                model_cand,
                size=min(required_new, len(model_cand)),
                replace=False,
            )
            return picked.tolist()

        if strategy == "diversity_sampling":
            return self._select_diverse_candidates(
                adata_pred=adata_pred,
                model_cand=model_cand,
                trained_condition=trained_condition,
                required_new=required_new,
            )

        if strategy == "representative_sampling":
            return self._select_representative_candidates(
                adata_pred=adata_pred,
                model_cand=model_cand,
                trained_condition=trained_condition,
                required_new=required_new,
            )

        if strategy == "magnitude_sampling":
            return self._select_magnitude_candidates(
                adata_pred=adata_pred,
                model_cand=model_cand,
                trained_condition=trained_condition,
                required_new=required_new,
            )

        if strategy in {"outlier_sampling", "typical_sampling"}:
            return self._select_by_centroid_distance(
                adata_pred=adata_pred,
                model_cand=model_cand,
                trained_condition=trained_condition,
                required_new=required_new,
                farthest=(strategy == "outlier_sampling"),
            )

        raise ValueError(f"Unsupported sampling strategy: {strategy}")

    def init_random_sample(self, target_budget):
        """Build the first-round cold start when the task starts from random data.

        Args:
            target_budget: Requested budget percentage.

        Returns:
            tuple: ``(source_budget, result_budget, selected)``.

        Raises:
            ValueError: If the actual budget misses the target beyond tolerance.
        """
        source_budget = 0.0
        target_count = self._budget_count(target_budget)

        print("❄️ [Cold Start] with random sampling.")
        if target_count == 0:
            self._validate_budget_gap(target_budget, 0.0)
            return source_budget, 0.0, []

        selected = np.random.choice(self.pool, size=target_count, replace=False).tolist()
        result_budget = self._budget_percent_from_count(len(selected))
        self._validate_budget_gap(target_budget, result_budget)
        return source_budget, result_budget, selected

    def sample(self, adata_pred, trained_condition, target_budget, strategy):
        """Expand the cumulative active set to the requested budget.

        First acquires from model-visible candidates using the selected backend,
        then backfills any remaining quota from blind spots so the active task
        preserves its cumulative-budget semantics.

        Args:
            adata_pred: Predicted ``AnnData`` used for acquisition.
            trained_condition: Perturbations already selected.
            target_budget: Requested cumulative budget percentage.
            strategy: Active-learning sampling strategy.

        Returns:
            tuple: ``(source_budget, result_budget, selected)``.

        Raises:
            ValueError: If predictions, strategy, or resulting budget are invalid.
        """
        target_count = self._budget_count(target_budget)
        current_trained_effective = self._deduplicate_preserve_order(
            [p for p in trained_condition if p in self.pool]
        )
        current_count = len(current_trained_effective)
        source_budget = self._budget_percent_from_count(current_count)
        required_new = max(target_count - current_count, 0)

        if required_new == 0:
            self._validate_budget_gap(target_budget, source_budget)
            print("")
            print(f"   - Source Budget : {source_budget}%")
            print(f"   - Target Budget : {target_budget}%")
            print(f"   - Result Budget : {source_budget}%")
            return source_budget, source_budget, current_trained_effective

        model_cand, blind_spot = self._get_candidates(
            adata_pred,
            current_trained_effective,
        )
        new_from_model = self._sample_from_model_visible(
            adata_pred=adata_pred,
            strategy=strategy,
            model_cand=model_cand,
            trained_condition=current_trained_effective,
            required_new=required_new,
        )

        remaining_quota = required_new - len(new_from_model)
        blind_fill = []
        if remaining_quota > 0 and blind_spot:
            blind_fill = np.random.choice(
                blind_spot,
                size=min(remaining_quota, len(blind_spot)),
                replace=False,
            ).tolist()

        selected = current_trained_effective + new_from_model + blind_fill
        result_budget = self._budget_percent_from_count(len(selected))
        self._validate_budget_gap(target_budget, result_budget)

        print("")
        print(f"   - Source Budget : {source_budget}%")
        print(f"   - Target Budget : {target_budget}%")
        print(f"   - Result Budget : {result_budget}%")

        return source_budget, result_budget, selected
