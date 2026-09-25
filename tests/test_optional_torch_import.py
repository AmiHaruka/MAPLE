import subprocess
import sys
import unittest
from pathlib import Path


class OptionalTorchImportTests(unittest.TestCase):
    def test_single_structure_optimizer_algorithms_import_without_torch(self):
        repo_root = Path(__file__).resolve().parents[1]
        probe = """
import sys

sys.modules["torch"] = None

import maple.function.dispatcher.optimization.algorithm as algorithm

assert algorithm.LBFGS is not None
assert algorithm.RFO is not None
assert algorithm.SDCG is not None
assert (
    "maple.function.dispatcher.optimization.algorithm.blbfgs"
    not in sys.modules
)
"""
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
