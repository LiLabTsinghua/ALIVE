"""In-distribution data-scaling benchmark task."""

import numpy as np

from ..base import ALIVE
from ..split import build_round_split_labels


class InDistribution(ALIVE):
    """In-distribution data-scaling benchmark.

    Each round holds out a random subset of target-context perturbations for
    training and evaluates the model on the remaining (in-distribution) pool.
    Rounds are independent: the training set is re-sampled at every budget
    rather than accumulated.
    """

    goal_type = "InDistribution"

    def _init_target_views(self):
        """Use the full target-context pool as the adaptive evaluation pool."""
        self.pool = self._build_shared_target_views()
        self.total_pool_count = len(self.pool)

        print(f"   - Total Pool Perturbations: {self.total_pool_count}")

    def _build_round_split_labels(self, trained_condition):
        """Label the round split with selected perturbations promoted to ``train``.

        Args:
            trained_condition: Perturbations selected for training this round.

        Returns:
            pandas.Series: Split labels aligned to ``self.adata_model.obs``.
        """
        return build_round_split_labels(
            adata_model=self.adata_model,
            dataset_cell_type=self.config["dataset_cell_type"],
            trained_condition=trained_condition,
        )

    def _resolve_eval_targets(self, split, trained_condition):
        """Return the perturbations and reference AnnData for a split evaluation.

        Args:
            split: Evaluation split name. Only ``"test"`` is supported.
            trained_condition: Perturbations already in training this round.

        Returns:
            tuple: ``(perts_to_pred, adata_target)``.

        Raises:
            ValueError: If ``split`` is not ``"test"``.
        """
        if split == "test":
            perts_to_pred = set(self.pool) - set(trained_condition)
            adata_target = self.adata_real[
                self.adata_real.obs["perturbation"].isin(perts_to_pred)
            ]
            return perts_to_pred, adata_target

        raise ValueError(f"Unknown split: {split}. Available splits: ['test']")

    def _budget_percent_from_count(self, count):
        """Convert a selected perturbation count to a rounded pool percentage.

        Args:
            count: Number of selected perturbations.

        Returns:
            float: Selected fraction of the pool as a percentage.
        """
        if self.total_pool_count == 0:
            return 0.0
        return round((count / self.total_pool_count) * 100, 1)

    def init_random_sample(self, target_budget):
        """Build the first-round cold start by random sampling.

        Args:
            target_budget: Requested budget percentage.

        Returns:
            tuple: ``(source_budget, result_budget, selected)``.

        Raises:
            ValueError: If the actual budget misses the target beyond tolerance.
        """
        source_budget = 0
        target_count = int((target_budget / 100.0) * self.total_pool_count)

        print("❄️ [Cold Start] with random sampling.")
        selected = np.random.choice(self.pool, size=target_count, replace=False).tolist()
        result_budget = self._budget_percent_from_count(len(selected))
        self._validate_budget_gap(target_budget, result_budget)

        return source_budget, result_budget, selected

    def sample(self, adata_pred, trained_condition, target_budget, strategy):
        """Re-sample a fresh random training set for the next round.

        Args:
            adata_pred: Unused; kept for interface parity with active sampling.
            trained_condition: Current round's trained perturbations.
            target_budget: Target budget percentage for the next round.
            strategy: Must be ``"random_sampling"``.

        Returns:
            tuple: ``(source_budget, result_budget, selected)``.

        Raises:
            ValueError: If ``strategy`` is unsupported or the actual budget
                misses the target beyond tolerance.
        """
        if strategy != "random_sampling":
            raise ValueError(f"Unsupported sampling strategy: {strategy}")

        # Keep the random-sampling API stable when the evaluation pool is empty.
        if self.total_pool_count == 0:
            self._validate_budget_gap(target_budget, 0.0)
            print("")
            print("   - Source Budget : 0.0%")
            print(f"   - Target Budget : {target_budget}%")
            print("   - Result Budget : 0.0%")
            return 0.0, 0.0, []

        target_count = int((target_budget / 100.0) * self.total_pool_count)
        current_trained_effective = self._deduplicate_preserve_order(
            p for p in trained_condition if p in self.pool
        )
        current_count = len(current_trained_effective)
        source_budget = round((current_count / self.total_pool_count) * 100, 1)

        selected = np.random.choice(self.pool, size=target_count, replace=False).tolist()
        result_budget = self._budget_percent_from_count(len(selected))
        self._validate_budget_gap(target_budget, result_budget)

        print("")
        print(f"   - Source Budget : {source_budget}%")
        print(f"   - Target Budget : {target_budget}%")
        print(f"   - Result Budget : {result_budget}%")

        return source_budget, result_budget, selected
