"""Out-of-distribution data-scaling benchmark task."""

from ..split import (
    build_cluster_round_split_labels,
    load_fixed_conditions_from_config,
)
from .in_distribution import InDistribution


class OutOfDistribution(InDistribution):
    """Out-of-distribution data-scaling benchmark with a fixed cluster holdout.

    A fixed ``cluster_test`` set is carved out of the target-context pool and
    held out across every repeat and round, forming the out-of-distribution
    evaluation. The remaining pool stays adaptive: selected perturbations become
    ``train`` and the unselected remainder is evaluated as in-distribution
    ``test``.
    """

    goal_type = "OutOfDistribution"

    def _validate_config(self):
        """Validate the base config plus the required fixed cluster-holdout file.

        Raises:
            ValueError: If ``cluster_condition_path`` is missing/empty, an inline
                ``cluster_condition`` is given, or the holdout includes ``control``.
            TypeError: If the cluster condition file is not a JSON list.
        """
        super()._validate_config()

        if "cluster_condition" in self.config:
            raise ValueError(
                "OutOfDistribution accepts only 'cluster_condition_path'. "
                "Inline 'cluster_condition' is not supported."
            )

        cluster_condition_path = self.config.get("cluster_condition_path")
        if cluster_condition_path is None:
            raise ValueError("OutOfDistribution requires 'cluster_condition_path'.")

        cluster_condition = load_fixed_conditions_from_config(self.config, "cluster_condition")
        if not isinstance(cluster_condition, list):
            raise TypeError("cluster_condition_path must point to a JSON list.")

        self.cluster_test = self._deduplicate_preserve_order(
            [str(pert) for pert in cluster_condition]
        )
        if len(self.cluster_test) == 0:
            raise ValueError("cluster_condition_path cannot point to an empty list.")
        if "control" in self.cluster_test:
            raise ValueError("cluster_condition_path cannot include 'control'.")

    def _init_target_views(self):
        """Carve the fixed cluster holdout out of the adaptive target pool.

        Raises:
            ValueError: If a fixed holdout perturbation is absent from the target pool.
        """
        full_pool = self._build_shared_target_views()
        missing_cluster_test = [pert for pert in self.cluster_test if pert not in full_pool]
        if missing_cluster_test:
            raise ValueError(
                "cluster_condition_path contains perturbations missing from the target pool: "
                f"{missing_cluster_test}"
            )

        self.pool = [pert for pert in full_pool if pert not in set(self.cluster_test)]
        self.total_pool_count = len(self.pool)

        print(f"   - Fixed Cluster Test Perturbations: {len(self.cluster_test)}")
        print(f"   - Adaptive Pool Perturbations: {self.total_pool_count}")

    def _evaluation_splits(self):
        """Return out-of-distribution and in-distribution evaluation splits.

        Returns:
            list[str]: ``cluster_test`` followed by ``test``.
        """
        return ["cluster_test", "test"]

    def _build_round_split_labels(self, trained_condition):
        """Label the round split with a fixed ``cluster_test`` holdout.

        Args:
            trained_condition: Adaptive perturbations selected for training.

        Returns:
            pandas.Series: Split labels aligned to ``self.adata_model.obs``.
        """
        return build_cluster_round_split_labels(
            adata_model=self.adata_model,
            dataset_cell_type=self.config["dataset_cell_type"],
            trained_condition=trained_condition,
            cluster_test_condition=self.cluster_test,
        )

    def _resolve_eval_targets(self, split, trained_condition):
        """Return ``(perts_to_pred, adata_target)`` for a cluster-task split.

        Args:
            split: ``cluster_test`` or ``test``.
            trained_condition: Adaptive perturbations already selected.

        Returns:
            tuple: Requested perturbations and their target ``AnnData``.

        Raises:
            ValueError: If ``split`` is neither ``"cluster_test"`` nor ``"test"``.
        """
        if split == "cluster_test":
            perts_to_pred = set(self.cluster_test)
            adata_target = self.adata_real[
                self.adata_real.obs["perturbation"].isin(perts_to_pred)
            ]
            return perts_to_pred, adata_target

        if split == "test":
            perts_to_pred = set(self.pool) - set(trained_condition)
            adata_target = self.adata_real[
                self.adata_real.obs["perturbation"].isin(perts_to_pred)
            ]
            return perts_to_pred, adata_target

        raise ValueError(
            f"Unknown split: {split}. Available splits: ['cluster_test', 'test']"
        )

    def _prepare_round_trained_condition(self, trained_condition):
        """Deduplicate the trained set and require it to stay within the adaptive pool.

        Args:
            trained_condition: Perturbations configured for the current round.

        Returns:
            list: Deduplicated string perturbation labels.

        Raises:
            ValueError: If any perturbation falls outside the adaptive pool.
        """
        normalized = self._deduplicate_preserve_order([str(pert) for pert in trained_condition])
        invalid = [pert for pert in normalized if pert not in self.pool]
        if invalid:
            raise ValueError(
                "Predefined trained_condition must stay within the adaptive pool. "
                f"Invalid perturbations: {invalid}"
            )
        return normalized

    def _finalize_condition_record(self, condition_record, result_budget, new_trained_condition):
        """Leave the condition record unchanged after the final round.

        Args:
            condition_record: Existing budget-to-condition mapping.
            result_budget: Unused final result budget.
            new_trained_condition: Unused final selected perturbations.
        """
        # OutOfDistribution intentionally writes no trailing condition entry.
        return
