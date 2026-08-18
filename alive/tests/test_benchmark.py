import unittest

from alive.benchmark import build_benchmark_config


class BuildBenchmarkConfigTests(unittest.TestCase):
    def setUp(self):
        self.default_config = {
            "data_dir": "/data/home/shengkun/benchmark/data",
            "results_dir": "/data/home/shengkun/benchmark/results",
            "atlas": "essential_atlas",
            "dataset": "nadig_hepg2_essential",
            "cell_type": "hepg2",
            "model_name": "PRESAGE",
            "task": "in_distribution",
            "strategy": "random_sampling",
            "start_mode": "random",
            "seed": 56,
            "repeat": 3,
            "budget_plan": [10.0, 20.0],
            "use_support_data": False,
        }

    def test_in_distribution_uses_split_subdir_for_default_predefined_condition_path(self):
        runtime_config = build_benchmark_config(
            {
                **self.default_config,
                "task": "in_distribution",
            },
            enable_predefined_condition=True,
        )

        self.assertEqual(
            runtime_config["predefined_condition_path"],
            "/data/home/shengkun/benchmark/data/essential_atlas/split/"
            "in_distribution/nadig_hepg2_essential_hepg2_cond.json",
        )

    def test_out_of_distribution_uses_split_subdir_for_default_paths(self):
        runtime_config = build_benchmark_config(
            {
                **self.default_config,
                "task": "out_of_distribution",
            },
            enable_predefined_condition=True,
        )

        self.assertEqual(
            runtime_config["predefined_condition_path"],
            "/data/home/shengkun/benchmark/data/essential_atlas/split/"
            "out_of_distribution/nadig_hepg2_essential_hepg2_cond.json",
        )
        self.assertEqual(
            runtime_config["cluster_condition_path"],
            "/data/home/shengkun/benchmark/data/essential_atlas/split/"
            "out_of_distribution/nadig_hepg2_essential_hepg2_cluster_cond.json",
        )

    def test_disable_predefined_condition_removes_default_path_for_out_of_distribution(self):
        runtime_config = build_benchmark_config(
            {
                **self.default_config,
                "task": "out_of_distribution",
            },
            enable_predefined_condition=False,
        )

        self.assertNotIn("predefined_condition_path", runtime_config)

    def test_legacy_objection_is_rejected(self):
        with self.assertRaises(ValueError):
            build_benchmark_config(
                {
                    **self.default_config,
                    "objection": "in_distribution",
                },
                enable_predefined_condition=True,
            )

    def test_unsupported_task_is_rejected(self):
        with self.assertRaises(ValueError):
            build_benchmark_config(
                {
                    **self.default_config,
                    "task": "active_extrapolation",
                },
                enable_predefined_condition=False,
            )

    def test_strategy_must_match_task(self):
        invalid_configs = [
            {
                **self.default_config,
                "task": "in_distribution",
                "strategy": "diversity_sampling",
            },
            {
                **self.default_config,
                "task": "out_of_distribution",
                "strategy": "magnitude_sampling",
            },
            {
                **self.default_config,
                "task": "active_learning",
                "strategy": "active_diversity_sampling_mse",
            },
        ]

        for config in invalid_configs:
            with self.subTest(
                task=config["task"],
                strategy=config["strategy"],
            ), self.assertRaises(ValueError):
                build_benchmark_config(
                    config,
                    enable_predefined_condition=False,
                )

    def test_invalid_repeat_seed_and_support_flag_are_rejected(self):
        invalid_values = [
            ("repeat", 0),
            ("repeat", True),
            ("seed", 1.5),
            ("use_support_data", "yes"),
        ]
        for field, value in invalid_values:
            with self.subTest(field=field, value=value), self.assertRaises(
                (TypeError, ValueError)
            ):
                build_benchmark_config(
                    {
                        **self.default_config,
                        field: value,
                    },
                    enable_predefined_condition=True,
                )

    def test_boolean_budget_values_are_rejected(self):
        with self.assertRaises(TypeError):
            build_benchmark_config(
                {
                    **self.default_config,
                    "budget_plan": [False, True],
                },
                enable_predefined_condition=True,
            )


if __name__ == "__main__":
    unittest.main()
