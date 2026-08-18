import importlib
import sys
import unittest


class BenchmarkImportTests(unittest.TestCase):
    def test_benchmark_import_does_not_require_scanpy_until_data_loading(self):
        sys.modules.pop("alive.benchmark", None)
        sys.modules["scanpy"] = None
        self.addCleanup(sys.modules.pop, "scanpy", None)

        module = importlib.import_module("alive.benchmark")

        self.assertTrue(hasattr(module, "build_benchmark_config"))


if __name__ == "__main__":
    unittest.main()
