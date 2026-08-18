"""Shared benchmark lifecycle, validation, evaluation, and data utilities."""

from abc import ABC, abstractmethod

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .metrics import (
    build_empty_metric_summary,
    compute_runtime_metric_summary,
    load_runtime_gene_sets,
    metric_record_key,
)
from .padding import clip_adata_x_nonnegative, pad_missing_perturbations
from .split import (
    get_round_predefined_condition,
    load_predefined_conditions_from_config,
    validate_predefined_conditions,
    write_condition_record,
)

_THICK_LINE = "═" * 80

TASK_ALLOWED_STRATEGIES = {
    "in_distribution": ("random_sampling",),
    "out_of_distribution": ("random_sampling",),
    "active_learning": (
        "random_sampling",
        "diversity_sampling",
        "representative_sampling",
        "magnitude_sampling",
        "outlier_sampling",
        "typical_sampling",
    ),
}


class ALIVE(ABC):
    """Shared skeleton for ALIVE perturbation benchmarks.

    A benchmark runs a budget schedule: at each budget it decides which
    perturbations are available for training, fits the wrapped model, predicts
    the evaluation split(s), and records runtime metrics. The whole round loop
    lives in :meth:`run`; concrete tasks customize behavior through a small set
    of hooks (which split(s) to evaluate, how to label the round split, how to
    select the next training set, and so on).

    Concrete implementations are ``InDistribution`` for random per-round
    holdouts, ``OutOfDistribution`` for fixed cluster holdouts, and
    ``ActiveLearning`` for cumulative active acquisition.
    """

    # Console-only labels; not written to any output file.
    goal_type = None
    goal = (
        "lower MSE/MAE, higher Correlation/Systema Correlation, "
        "and Systema Bias closer to zero are better"
    )

    # Validation contracts. Subclasses widen ``allowed_strategies`` as needed.
    allowed_strategies = list(TASK_ALLOWED_STRATEGIES["in_distribution"])
    allowed_start_modes = ["random", "zero-shot"]

    def __init__(self, adata, model_capacity, model_wrapper, config):
        """Initialize a benchmark task.

        Args:
            adata: Loaded atlas ``AnnData``. Kept by reference to avoid
                duplicating the full matrix; task subsets opt into ``copy()``.
            model_capacity: Perturbations the wrapped model can predict.
            model_wrapper: Method adapter exposing ``fit`` and ``predict``.
            config: Resolved runtime config (budget plan, strategy, paths, ...).
        """
        self.adata = adata
        self.model_capacity = model_capacity
        self.model_wrapper = model_wrapper
        self.config = config

        self.budget_plan = config["budget_plan"]
        self.strategy = config["strategy"]
        self.start_mode = config["start_mode"]
        self.max_budget_gap = config.get("max_budget_gap", 5.0)
        self.predefined_condition = load_predefined_conditions_from_config(
            config, self.budget_plan
        )

        self.pool = None

        self._print_welcome_banner()
        self._validate()
        self._print_init_summary()
        self._initialize_adata_views()

    # ------------------------------------------------------------------
    # Banner and console summaries
    # ------------------------------------------------------------------
    def _print_welcome_banner(self):
        """Print the ALIVE banner shown when a task is instantiated."""
        logo = r"""
     _    _     ___ __     __ _____
    / \  | |   |_ _|\ \   / /| ____|
   / _ \ | |    | |  \ \ / / |  _|
  / ___ \| |___ | |   \ V /  | |___
 /_/   \_\_____|___|   \_/   |_____|
        """

        width = 76
        top_border = "┏" + "━" * width + "┓"
        bottom_border = "┗" + "━" * width + "┛"
        thick_line = "┣" + "━" * width + "┫"

        cyan = "\033[36m"
        yellow = "\033[1;33m"
        white = "\033[1;37m"
        italic = "\033[3m"
        reset = "\033[0m"

        print(f"{cyan}{logo}{reset}")
        print(f"{yellow}{top_border}{reset}")

        title_text = " AIVC still ALIVE"
        print(f"{yellow}┃{reset}{white}{title_text.center(width)}{reset}{yellow}┃{reset}")
        print(f"{yellow}{thick_line}{reset}")

        mission_1 = "Empowering Biological Discovery with Lab-in-the-Loop AI Workflows."
        mission_2 = "ALIVE systematically evaluates how experimental feedback guides the"
        mission_3 = "exploration of high-dimensional perturbation landscapes."

        print(f"{yellow}┃{reset} {italic}{mission_1.center(width - 2)}{reset} {yellow}┃{reset}")
        print(f"{yellow}┃{reset} {' '.center(width - 2)} {yellow}┃{reset}")
        print(f"{yellow}┃{reset} {mission_2.center(width - 2)} {yellow}┃{reset}")
        print(f"{yellow}┃{reset} {mission_3.center(width - 2)} {yellow}┃{reset}")
        print(f"{yellow}{bottom_border}{reset}")

    def _print_init_summary(self):
        """Print the resolved task, model, and schedule settings at startup."""
        print("⚙️ Framework Initialized:")
        print(f"   - Type: {self.goal_type}")
        print(f"   - Goal: {self.goal}")
        print(f"🚀 Model: {self.config['model_name']}")
        print(f"   - Model Capacity: {len(self.model_capacity)} perturbations")
        print(f"🚀 AL Strategy: {self.strategy}")
        print(f"   - Budget Plan: {self.budget_plan}")
        print(f"   - Start Mode: {self.start_mode}")
        schedule = "enabled" if self.predefined_condition is not None else "disabled"
        print(f"   - Predefined Schedule: {schedule}")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate(self):
        """Validate config and AnnData before any views are built."""
        self._validate_config()
        self._validate_adata()

    def _validate_config(self):
        """Validate strategy, start mode, and budget plan against the contracts.

        Raises:
            TypeError: If a configured value has an invalid type.
            ValueError: If a configured value violates its allowed range or set.
        """
        self._validate_choice("strategy", self.strategy, self.allowed_strategies)
        self._validate_choice("start_mode", self.start_mode, self.allowed_start_modes)
        self._validate_budget_plan(self.budget_plan)
        self._validate_max_budget_gap(self.max_budget_gap)
        if not isinstance(self.config["use_support_data"], bool):
            raise TypeError("config['use_support_data'] must be a boolean.")

    def _validate_adata(self):
        """Validate the raw atlas schema used to construct round-specific splits.

        The atlas stores candidate perturbations as ``pool``, controls as
        ``control``, and fixed validation rows as ``val``. ALIVE derives the
        model-facing ``train`` and ``test`` labels independently for every round.

        Raises:
            KeyError: If a required ``obs`` column is missing.
            ValueError: If split labels are invalid or disagree with control
                perturbation labels.
        """
        required_columns = ["dataset_cell_type", "split", "perturbation"]
        missing_cols = [c for c in required_columns if c not in self.adata.obs.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns in adata.obs: {missing_cols}")

        allowed_split_values = {"pool", "control", "val"}
        split_series = self.adata.obs["split"]
        split_values = set(split_series.dropna().astype(str).unique())
        invalid_split_values = sorted(split_values - allowed_split_values)
        if split_series.isna().any():
            invalid_split_values.append("<missing>")
        if invalid_split_values:
            raise ValueError(
                "adata.obs['split'] contains unsupported values. "
                f"Allowed: {sorted(allowed_split_values)}. "
                f"Invalid: {invalid_split_values}."
            )

        perturbation_series = self.adata.obs["perturbation"]
        if "control" not in perturbation_series.values:
            raise ValueError("'control' must exist in adata.obs['perturbation']")

        split_control_mask = split_series.astype(str) == "control"
        perturbation_control_mask = perturbation_series.astype(str) == "control"
        inconsistent_control_mask = split_control_mask != perturbation_control_mask
        if inconsistent_control_mask.any():
            inconsistent_count = int(inconsistent_control_mask.sum())
            raise ValueError(
                "adata.obs['split'] must be 'control' exactly when "
                "adata.obs['perturbation'] is 'control'. "
                f"Found {inconsistent_count} inconsistent rows."
            )

    @staticmethod
    def _validate_budget_plan(budget_plan):
        """Validate the shared budget-plan contract used by every task.

        The plan must be a non-empty, strictly increasing list of percentages in
        the closed range [0, 100].

        Args:
            budget_plan: Ordered budget percentages to validate.

        Raises:
            TypeError: If the plan or a value has the wrong type.
            ValueError: If the plan is empty, out of range, or not increasing.
        """
        if not isinstance(budget_plan, list):
            raise TypeError("budget_plan must be a list.")
        if len(budget_plan) == 0:
            raise ValueError("budget_plan cannot be empty.")

        for budget in budget_plan:
            if isinstance(budget, bool) or not isinstance(budget, (int, float)):
                raise TypeError(f"Budget value must be numeric. Got {type(budget)}")
            if not (0 <= budget <= 100):
                raise ValueError(
                    f"Budget value {budget} out of range. Must be between 0 and 100."
                )

        for i in range(1, len(budget_plan)):
            if budget_plan[i] <= budget_plan[i - 1]:
                raise ValueError(
                    f"budget_plan must be strictly increasing. Got: {budget_plan}"
                )

    @staticmethod
    def _validate_choice(name, value, allowed):
        """Validate that ``value`` is one of ``allowed`` for config field ``name``.

        Args:
            name: Configuration field name used in error messages.
            value: Configured value to validate.
            allowed: Collection of accepted values.

        Raises:
            ValueError: If ``value`` is not in ``allowed``.
        """
        if value not in allowed:
            raise ValueError(f"Invalid {name}: {value}. Allowed: {allowed}")

    @staticmethod
    def _validate_max_budget_gap(max_budget_gap):
        """Validate the allowed absolute gap between target and actual budgets.

        Args:
            max_budget_gap: Maximum allowed absolute percentage-point gap.

        Raises:
            TypeError: If ``max_budget_gap`` is not numeric or is a boolean.
            ValueError: If ``max_budget_gap`` is negative.
        """
        if isinstance(max_budget_gap, bool) or not isinstance(max_budget_gap, (int, float)):
            raise TypeError("max_budget_gap must be a non-negative number.")
        if max_budget_gap < 0:
            raise ValueError("max_budget_gap must be non-negative.")

    def _validate_budget_gap(self, target_budget, actual_budget):
        """Raise when discrete sampling misses the target budget by too much.

        Args:
            target_budget: Requested budget percentage.
            actual_budget: Percentage represented by the selected perturbations.

        Raises:
            ValueError: If the absolute percentage-point gap exceeds
                ``config['max_budget_gap']``.
        """
        budget_gap = abs(float(target_budget) - float(actual_budget))
        if budget_gap > self.max_budget_gap:
            raise ValueError(
                "Actual sampling budget differs from the target by more than "
                f"max_budget_gap={self.max_budget_gap:.1f} percentage points. "
                f"target_budget={float(target_budget):.1f}, "
                f"actual_budget={float(actual_budget):.1f}, "
                f"gap={budget_gap:.1f}, pool_size={self.total_pool_count}."
            )

    # ------------------------------------------------------------------
    # AnnData view initialization (shared across tasks)
    # ------------------------------------------------------------------
    def _initialize_adata_views(self):
        """Build the target, model, and reference views from the atlas."""
        print("Initializing AnnData views...")
        self._init_target_views()
        self._init_model_view()
        self._init_reference_profiles()

    def _get_target_mask(self):
        """Return a boolean mask selecting the target-context rows of the atlas.

        Returns:
            pandas.Series: Boolean mask aligned to ``self.adata.obs``.
        """
        return self.adata.obs["dataset_cell_type"] == self.config["dataset_cell_type"]

    def _build_shared_target_views(self):
        """Build the target-context evaluation subset shared by tasks.

        Sets ``self.adata_real`` / ``self.adata_ctrl`` (evaluation rows and their
        controls). Validation rows remain in ``self.adata_model`` and receive
        per-round ``val`` labels for model training and early stopping. Tasks
        call this and derive their own ``self.pool`` from the returned full pool.

        Returns:
            list: Target-context perturbations excluding ``control``.

        Raises:
            ValueError: If the evaluation subset has no control cells.
        """
        target_mask = self._get_target_mask()
        real_mask = target_mask & (self.adata.obs["split"] != "val")

        self.adata_real = self.adata[real_mask]
        self.adata_ctrl = self.adata_real[self.adata_real.obs["perturbation"] == "control"]
        if self.adata_ctrl.n_obs == 0:
            raise ValueError("Control cells were not found in the evaluation subset.")

        full_pool = self.adata_real.obs["perturbation"].unique().tolist()
        full_pool.remove("control")
        return full_pool

    def _init_model_view(self):
        """Prepare the model-facing training universe with optional support context."""
        support_mask = self.adata.obs["perturbation"].isin(self.model_capacity) | (
            self.adata.obs["perturbation"] == "control"
        )

        if self.config["use_support_data"]:
            self.adata_model = self.adata[support_mask].copy()
            context_count = self.adata_model.obs["dataset_cell_type"].nunique()
            print(f"Using support data across {context_count} contexts.")
        else:
            target_mask = self._get_target_mask()
            self.adata_model = self.adata[support_mask & target_mask].copy()
            print("Using target context only.")

    def _init_reference_profiles(self):
        """Compute reference profiles used by runtime evaluation metrics.

        Raises:
            ValueError: If the target context has no non-control perturbation.
        """
        target_mask = self._get_target_mask()
        target_non_control_mask = target_mask & (self.adata.obs["perturbation"] != "control")
        adata_target_non_control = self.adata[target_non_control_mask]
        if adata_target_non_control.n_obs == 0:
            raise ValueError("Target context must contain at least one non-control perturbation.")

        self.ref_control = self._get_mean_feat(self.adata_ctrl)
        target_non_control_perts = (
            adata_target_non_control.obs["perturbation"].astype(str).unique().tolist()
        )
        target_bulk = self._get_bulk_feat(adata_target_non_control, target_non_control_perts)
        self.ref_pert_mean = target_bulk.mean(axis=0).to_numpy(dtype=float, copy=False)
        self.hvg_genes = None
        self.de_dict = None

    def _ensure_runtime_metric_gene_sets_loaded(self):
        """Lazily load the HVG and DE gene sets on first evaluation."""
        if self.hvg_genes is not None and self.de_dict is not None:
            return
        self.hvg_genes, self.de_dict = load_runtime_gene_sets(
            atlas_dir=self.config.get("atlas_dir"),
            dataset_cell_type=self.config["dataset_cell_type"],
        )

    # ------------------------------------------------------------------
    # Round loop (the one shared orchestration)
    # ------------------------------------------------------------------
    def run(self, repeat=None):
        """Run all budget rounds for one repeat and persist runtime metrics.

        The loop is identical across tasks; per-task behavior comes from the
        hooks (``_evaluation_splits``, ``_build_round_split_labels``,
        ``_resolve_eval_targets``, ``sample``, ``init_random_sample``,
        ``_finalize_condition_record``).

        Args:
            repeat: Zero-based repeat index. Required.

        Returns:
            dict: ``{"runtime_metric_rows": [...]}`` for the run aggregator.

        Raises:
            ValueError: If ``repeat`` is not provided.
        """
        if repeat is None:
            raise ValueError("Please specify the repeat number.")

        predefined_condition = self.predefined_condition
        if predefined_condition is not None:
            validate_predefined_conditions(
                predefined_condition=predefined_condition,
                repeat_count=self.config["repeat"],
                budget_plan=self.budget_plan,
            )
            print("Using predefined condition schedule.")

        eval_splits = self._evaluation_splits()
        condition_record = {}
        runtime_metric_rows = []
        trained_condition = []

        print(f"\033[1;33m{_THICK_LINE}\033[0m")
        print(f"⚙️ Framework (Repeat {repeat}):")
        print(f"\033[1;33m{_THICK_LINE}\033[0m")

        source_budget = 0.0
        result_budget = 0.0
        new_trained_condition = []
        for i, target_budget in enumerate(self.budget_plan):
            print(f"Round {i + 1} start.")

            if predefined_condition is not None:
                trained_condition = get_round_predefined_condition(
                    predefined_condition=predefined_condition,
                    repeat=repeat,
                    round_index=i,
                )
            trained_condition = self._prepare_round_trained_condition(trained_condition)

            split_key = f"{i}_cond"
            self.adata_model.obs[split_key] = self._build_round_split_labels(trained_condition)
            print(f"Model trained on {trained_condition}")
            print(self.adata_model.obs[split_key].value_counts())

            predictions, source_budget, result_budget, new_trained_condition = (
                self._run_round_model_cycle(
                    split_key=split_key,
                    trained_condition=trained_condition,
                    target_budget=target_budget,
                    is_first_round=(i == 0),
                    eval_splits=eval_splits,
                )
            )

            for split in eval_splits:
                pred_bulk, metric_summary = self.evaluate(
                    predictions[split],
                    trained_condition=trained_condition,
                    split=split,
                )
                self._write_round_prediction_output(repeat, source_budget, split, pred_bulk)
                runtime_metric_rows.extend(
                    self._metric_summary_to_rows(
                        repeat=repeat,
                        x=source_budget,
                        split_name=split,
                        metric_summary=metric_summary,
                    )
                )

            condition_record[source_budget] = trained_condition

            trained_condition = new_trained_condition
            print(f"Round {i + 1} end.")
            print(f"\033[1;33m{_THICK_LINE}\033[0m")

        self._finalize_condition_record(condition_record, result_budget, new_trained_condition)
        write_condition_record(
            saved_dir=self.config["saved_dir"],
            repeat=repeat,
            condition_record=condition_record,
        )
        return {"runtime_metric_rows": runtime_metric_rows}

    def _zero_shot_baseline_prediction(self):
        """Return an empty prediction so evaluation pads every target to control.

        A random cold start fits no model, but the round is still evaluated
        against a zero-shot baseline: with no predicted cells, the padding step
        fills every target perturbation with the control mean. This yields the
        control-baseline metrics (Correlation against control is approximately
        zero; MSE/MAE reflect the true control gap) instead of an empty summary.

        Returns:
            anndata.AnnData: A zero-cell prediction over the atlas gene space.
        """
        empty_obs = pd.DataFrame(
            {"perturbation": pd.Series([], dtype=object)},
            index=pd.Index([], dtype=str),
        )
        return ad.AnnData(
            X=np.empty((0, self.adata_ctrl.n_vars), dtype=np.float32),
            obs=empty_obs,
            var=self.adata_ctrl.var.copy(),
        )

    def _run_round_model_cycle(
        self, split_key, trained_condition, target_budget, is_first_round, eval_splits
    ):
        """Run one round's fit/predict/select cycle.

        A random cold start (first round, ``start_mode == "random"``) fits no
        model: every evaluation split is scored against a zero-shot control
        baseline (via padding) and the next training set is chosen randomly. All
        other rounds (including a zero-shot first round) fit, predict every
        evaluation split, then select the next training set from the acquisition
        split's prediction.

        Args:
            split_key: Per-round column in ``adata_model.obs`` containing split labels.
            trained_condition: Perturbations available for training this round.
            target_budget: Requested budget percentage for the next selection.
            is_first_round: Whether this is the first budget round.
            eval_splits: Ordered split names to predict and evaluate.

        Returns:
            tuple: ``(predictions, source_budget, result_budget, new_trained)``
            where ``predictions`` maps each evaluation split to its predicted
            ``AnnData`` (a zero-shot baseline on a random cold start).
        """
        if is_first_round and self.start_mode == "random":
            predictions = {split: self._zero_shot_baseline_prediction() for split in eval_splits}
            source_budget, result_budget, new_trained = self.init_random_sample(target_budget)
            return predictions, source_budget, result_budget, new_trained

        self.model_wrapper.fit(self.adata_model, split_key=split_key)
        predictions = {
            split: self.model_wrapper.predict(self.adata_model, split_key=split_key, split=split)
            for split in eval_splits
        }
        for split, adata_pred in predictions.items():
            perts_to_pred, _ = self._resolve_eval_targets(split, trained_condition)
            self._validate_prediction_schema(adata_pred, perts_to_pred, split)

        source_budget, result_budget, new_trained = self.sample(
            predictions[self._acquisition_split()],
            trained_condition,
            target_budget,
            strategy=self.strategy,
        )
        return predictions, source_budget, result_budget, new_trained

    # ------------------------------------------------------------------
    # Evaluation (shared across tasks)
    # ------------------------------------------------------------------
    def evaluate(self, adata_pred, split, trained_condition=None):
        """Evaluate one split's prediction and return its bulk table and metrics.

        Args:
            adata_pred: Predicted ``AnnData`` for the split, or ``None``.
            split: Evaluation split name (e.g. ``"test"`` / ``"cluster_test"``).
            trained_condition: Perturbations already in training this round.

        Returns:
            tuple: ``(pred_bulk, metric_summary)``. ``pred_bulk`` is ``None`` when
            no prediction is available.
        """
        if adata_pred is None:
            metric_summary = build_empty_metric_summary()
            print(f"---------------computing {split} metrics-----------------")
            print(f"No {split} prediction available. Metrics are undefined (NaN).")
            self._print_split_summary(split, metric_summary)
            return None, metric_summary

        print(f"---------------computing {split} metrics-----------------")
        perts_to_pred, adata_target = self._resolve_eval_targets(split, trained_condition)
        self._validate_prediction_schema(adata_pred, perts_to_pred, split)
        self._ensure_runtime_metric_gene_sets_loaded()
        mode = self._infer_prediction_mode(adata_pred)

        adata_pred = pad_missing_perturbations(
            adata_pred,
            perts_to_pred,
            self.adata_ctrl,
            mode=mode,
        )
        clip_adata_x_nonnegative(adata_pred)
        pred_bulk = self._get_bulk_feat(
            adata_pred,
            adata_pred.obs["perturbation"].astype(str).unique().tolist(),
        )
        true_bulk = self._get_bulk_feat(adata_target, sorted(perts_to_pred))

        print("Computing population-average metrics.")
        metric_summary = compute_runtime_metric_summary(
            pred_bulk=pred_bulk,
            true_bulk=true_bulk,
            ref_control=self.ref_control,
            ref_pert_mean=self.ref_pert_mean,
            hvg_genes=self.hvg_genes,
            de_dict=self.de_dict,
        )
        self._print_split_summary(split, metric_summary)
        return pred_bulk, metric_summary

    def _validate_prediction_schema(self, adata_pred, perts_to_pred, split):
        """Validate one model prediction before acquisition or evaluation.

        Missing target perturbations remain legal because the evaluation path
        pads them with control profiles. Extra perturbations are rejected so a
        stale or incorrectly scoped prediction cannot affect acquisition or
        reported metrics.

        Args:
            adata_pred: Predicted ``AnnData``, or ``None`` when unavailable.
            perts_to_pred: Perturbations requested for this split.
            split: Split label used in error messages.

        Raises:
            TypeError: If the prediction is not an ``AnnData`` or perturbation
                labels are not strings.
            KeyError: If ``obs['perturbation']`` is missing.
            ValueError: If perturbation labels are missing, genes are duplicated
                or incompatible, or unrequested perturbations are present.
        """
        if adata_pred is None:
            return
        if not isinstance(adata_pred, ad.AnnData):
            raise TypeError(
                f"Prediction for split '{split}' must be an AnnData or None. "
                f"Got {type(adata_pred).__name__}."
            )
        if "perturbation" not in adata_pred.obs.columns:
            raise KeyError(
                f"Prediction for split '{split}' must contain obs['perturbation']."
            )

        perturbations = adata_pred.obs["perturbation"]
        if perturbations.isna().any():
            raise ValueError(
                f"Prediction for split '{split}' contains missing perturbation labels."
            )
        non_string_labels = [
            label for label in perturbations.unique().tolist() if not isinstance(label, str)
        ]
        if non_string_labels:
            raise TypeError(
                f"Prediction for split '{split}' has non-string perturbation labels: "
                f"{non_string_labels}."
            )

        if not adata_pred.var_names.is_unique:
            duplicate_genes = adata_pred.var_names[
                adata_pred.var_names.duplicated()
            ].unique().tolist()
            raise ValueError(
                f"Prediction for split '{split}' contains duplicate genes: "
                f"{duplicate_genes}."
            )

        predicted_genes = adata_pred.var_names.tolist()
        expected_genes = self.adata_ctrl.var_names.tolist()
        if predicted_genes != expected_genes:
            raise ValueError(
                f"Prediction genes for split '{split}' must exactly match the atlas "
                "gene names and order."
            )

        requested_perturbations = {str(pert) for pert in perts_to_pred}
        predicted_perturbations = set(perturbations.tolist())
        extra_perturbations = sorted(
            predicted_perturbations - requested_perturbations
        )
        if extra_perturbations:
            raise ValueError(
                f"Prediction for split '{split}' contains unrequested perturbations: "
                f"{extra_perturbations}."
            )

    @staticmethod
    def _infer_prediction_mode(adata_pred):
        """Detect whether predictions are aggregated to one cell per perturbation.

        Args:
            adata_pred: Predicted ``AnnData`` with ``obs['perturbation']``.

        Returns:
            str: ``"bulk"`` for one row per perturbation, otherwise ``"single-cell"``.
        """
        perturb_counts = adata_pred.obs["perturbation"].value_counts()
        if (perturb_counts == 1).all():
            print("Detected one predicted cell per perturbation. Using bulk evaluation mode.")
            return "bulk"
        print("Detected multi-cell perturbation predictions. Using single-cell evaluation mode.")
        return "single-cell"

    @staticmethod
    def _format_metric_summary_df(metric_summary):
        """Format a metric summary dict as a gene-set by metric table.

        Args:
            metric_summary: Nested mapping of gene sets to metric values.

        Returns:
            pandas.DataFrame: Gene-set by metric summary table.
        """
        return pd.DataFrame.from_dict(metric_summary, orient="index")[
            list(metric_summary["ALL"].keys())
        ]

    def _print_split_summary(self, split_name, metric_summary):
        """Print one split's metric summary table to the console.

        Args:
            split_name: Evaluation split label.
            metric_summary: Nested mapping of gene sets to metric values.
        """
        summary_df = self._format_metric_summary_df(metric_summary)
        print(f"[{split_name}] Summary Statistics:")
        print(summary_df.to_string(float_format=lambda x: f"{x:.6f}"))

    def _metric_summary_to_rows(self, repeat, x, split_name, metric_summary):
        """Flatten a metric summary into long-format runtime rows for one split.

        Args:
            repeat: Zero-based repeat index.
            x: Source budget percentage recorded on the x-axis.
            split_name: Evaluation split label.
            metric_summary: Nested mapping of gene sets to metric values.

        Returns:
            list[dict]: Long-format runtime metric rows.
        """
        rows = []
        for gene_set_name, metric_values in metric_summary.items():
            gene_name = gene_set_name.lower()
            for metric_name, value in metric_values.items():
                rows.append(
                    {
                        "method": self.config["model_name"],
                        "context": self.config["dataset_cell_type"],
                        "repeat": int(repeat),
                        "x": float(x),
                        "set": str(split_name),
                        "metric_name": metric_record_key(metric_name),
                        "gene": gene_name,
                        "value": float(value),
                    }
                )
        return rows

    def _write_round_prediction_output(self, repeat, source_budget, split_name, pred_bulk):
        """Persist one round's prediction bulk table for a split.

        Args:
            repeat: Zero-based repeat index.
            source_budget: Budget percentage used to train the prediction model.
            split_name: Evaluation split label used in the filename.
            pred_bulk: Per-perturbation prediction table, or ``None``.
        """
        if pred_bulk is None:
            return
        pred_bulk.astype("float32").to_csv(
            f'{self.config["saved_dir"]}/repeat_{repeat}_{source_budget:.1f}_{split_name}_pred_bulk.csv.gz',
            index=True,
            compression="gzip",
        )

    # ------------------------------------------------------------------
    # Hooks with shared defaults (in-distribution behavior)
    # ------------------------------------------------------------------
    def _evaluation_splits(self):
        """Return the ordered split names to predict and evaluate each round.

        Returns:
            list[str]: Evaluation split names.
        """
        return ["test"]

    def _acquisition_split(self):
        """Return the split whose prediction feeds the next-round acquisition.

        Returns:
            str: Acquisition split name.
        """
        return "test"

    def _prepare_round_trained_condition(self, trained_condition):
        """Normalize the round's trained set before building split labels.

        Args:
            trained_condition: Perturbations configured for the current round.

        Returns:
            list: Normalized perturbation list.
        """
        return trained_condition

    def _finalize_condition_record(self, condition_record, result_budget, new_trained_condition):
        """Append the final selected set after the last round.

        In-distribution and active-learning record the terminal selection keyed
        by the last round's result budget. Out-of-distribution overrides this to
        a no-op to preserve its original behavior.

        Args:
            condition_record: Mutable mapping of budgets to selected perturbations.
            result_budget: Actual percentage of the final selected pool.
            new_trained_condition: Final selected perturbations.
        """
        condition_record[result_budget] = new_trained_condition

    # ------------------------------------------------------------------
    # Task-specific hooks (must be implemented by concrete tasks)
    # ------------------------------------------------------------------
    @abstractmethod
    def _init_target_views(self):
        """Prepare target-context validation/evaluation subsets and ``self.pool``."""

    @abstractmethod
    def _build_round_split_labels(self, trained_condition):
        """Return the per-round train/val/test split labels for ``adata_model``.

        Args:
            trained_condition: Perturbations available for training this round.

        Returns:
            pandas.Series: Split labels aligned to ``self.adata_model.obs``.
        """

    @abstractmethod
    def _resolve_eval_targets(self, split, trained_condition):
        """Return evaluation perturbations and target data for a split.

        Args:
            split: Evaluation split name.
            trained_condition: Perturbations available for training this round.

        Returns:
            tuple: ``(perts_to_pred, adata_target)`` for the requested split.
        """

    @abstractmethod
    def init_random_sample(self, target_budget):
        """Build the first-round cold-start training set by random sampling.

        Args:
            target_budget: Requested budget percentage.

        Returns:
            tuple: Source budget, actual result budget, and selected perturbations.
        """

    @abstractmethod
    def sample(self, adata_pred, trained_condition, target_budget, strategy):
        """Select the next-round training set and return budgets plus selection.

        Args:
            adata_pred: Prediction used by the acquisition strategy.
            trained_condition: Perturbations already available for training.
            target_budget: Requested budget percentage.
            strategy: Sampling strategy name.

        Returns:
            tuple: Source budget, actual result budget, and selected perturbations.
        """

    # ------------------------------------------------------------------
    # Generic AnnData / bulk utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _deduplicate_preserve_order(items):
        """Drop duplicates from ``items`` while preserving first-seen order.

        Args:
            items: Iterable of hashable values.

        Returns:
            list: Deduplicated values in first-seen order.
        """
        return list(dict.fromkeys(items))

    def _require_perturbation_column(self, adata):
        """Require ``adata.obs`` to contain a ``perturbation`` column.

        Args:
            adata: ``AnnData`` object to validate.

        Raises:
            KeyError: If ``adata.obs`` lacks ``perturbation``.
        """
        if "perturbation" not in adata.obs.columns:
            raise KeyError("adata.obs must contain 'perturbation'.")

    def _to_numpy_matrix(self, X):
        """Return ``X`` as a dense NumPy array, densifying sparse input.

        Args:
            X: Dense or SciPy sparse matrix.

        Returns:
            numpy.ndarray: Dense matrix view or copy.
        """
        return X.toarray() if sp.issparse(X) else np.asarray(X)

    def _subset_by_perturbations(self, adata, perts):
        """Return the ``adata`` view whose perturbation is in ``perts``.

        Args:
            adata: Source ``AnnData``.
            perts: Perturbations to retain.

        Returns:
            anndata.AnnData: Matching row subset.
        """
        self._require_perturbation_column(adata)
        return adata[adata.obs["perturbation"].isin(perts)]

    def _group_mean_by_perturbation(self, adata):
        """Return per-perturbation mean expression as a perturbation by gene table.

        Args:
            adata: Source ``AnnData`` with perturbation labels.

        Returns:
            pandas.DataFrame: Mean expression indexed by perturbation.
        """
        self._require_perturbation_column(adata)
        X = self._to_numpy_matrix(adata.X)
        df = pd.DataFrame(
            X,
            index=adata.obs["perturbation"].astype(str).to_numpy(),
            columns=adata.var_names,
        )
        return df.groupby(level=0, observed=False).mean()

    def _get_candidates(self, adata_pred, trained_condition):
        """Split the untrained pool into model-visible candidates and blind spots.

        Args:
            adata_pred: Predicted ``AnnData`` whose perturbations are model-visible.
            trained_condition: Perturbations already in training.

        Returns:
            tuple: ``(model_cand, blind_spot)`` perturbation lists.

        Raises:
            ValueError: If ``self.pool`` is unset or ``adata_pred`` is ``None``.
        """
        if self.pool is None:
            raise ValueError("self.pool must be initialized before calling _get_candidates().")
        if adata_pred is None:
            raise ValueError("adata_pred cannot be None when selecting candidates.")

        self._require_perturbation_column(adata_pred)

        trained_set = set(trained_condition)
        all_cand = [pert for pert in self.pool if pert not in trained_set]

        predicted_perts = set(adata_pred.obs["perturbation"].unique())
        model_cand = [pert for pert in all_cand if pert in predicted_perts]
        blind_spot = [pert for pert in all_cand if pert not in predicted_perts]

        print(f"All candidates: {len(all_cand)}")
        print(f"model_visible: {len(model_cand)}")
        print(f"blind_spot: {len(blind_spot)}")

        return model_cand, blind_spot

    def _get_bulk_feat(self, adata, perts):
        """Return per-perturbation mean pseudobulk for requested perturbations.

        Args:
            adata: Source ``AnnData``.
            perts: Perturbations to aggregate.

        Returns:
            pandas.DataFrame: Per-perturbation pseudobulk, empty if none match.
        """
        sub = self._subset_by_perturbations(adata, perts)
        if sub.n_obs == 0:
            return pd.DataFrame(columns=adata.var_names, dtype=float)
        return self._group_mean_by_perturbation(sub)

    def _get_mean_feat(self, adata):
        """Return the mean expression vector across all cells of ``adata``.

        Args:
            adata: Source ``AnnData``.

        Returns:
            numpy.ndarray: One-dimensional mean expression vector.
        """
        return np.asarray(adata.X.mean(axis=0)).ravel()

    def _get_bulk_feat_delta(self, adata, perts, ref):
        """Return per-perturbation mean pseudobulk minus a reference vector.

        Args:
            adata: Source ``AnnData``.
            perts: Perturbations to aggregate.
            ref: Reference vector of length ``adata.n_vars`` subtracted per cell.

        Returns:
            pandas.DataFrame: Per-perturbation delta pseudobulk (empty if none match).

        Raises:
            ValueError: If ``ref`` length does not match ``adata.n_vars``.
        """
        ref = np.asarray(ref).ravel()
        if ref.shape[0] != adata.n_vars:
            raise ValueError(
                f"Reference vector length {ref.shape[0]} does not match adata.n_vars {adata.n_vars}."
            )

        sub = self._subset_by_perturbations(adata, perts)
        if sub.n_obs == 0:
            return pd.DataFrame(columns=adata.var_names, dtype=float)

        X = self._to_numpy_matrix(sub.X) - ref
        df = pd.DataFrame(
            X,
            index=sub.obs["perturbation"].astype(str).to_numpy(),
            columns=adata.var_names,
        )
        return df.groupby(level=0, observed=False).mean()
