import subprocess
import sys
import unittest
from pathlib import Path


class OptionalTorchImportTests(unittest.TestCase):
    def test_scalar_ts_scan_and_parser_modules_import_without_optional_ml_stacks(self):
        repo_root = Path(__file__).resolve().parents[1]
        probe = r"""
import importlib.abc
import sys

class OptionalMLBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (
            fullname == "torch"
            or fullname.startswith("torch.")
            or fullname == "fairchem"
            or fullname.startswith("fairchem.")
        ):
            raise ModuleNotFoundError(
                f"optional ML dependency deliberately blocked: {fullname}"
            )
        return None

sys.meta_path.insert(0, OptionalMLBlocker())

import maple.function.read as read_package
import maple.function.dispatcher.ts as ts_package

assert "InputReader" in read_package.__all__
assert "ExplicitSolv" in read_package.__all__
assert "TransitionState" in ts_package.__all__
assert "maple.function.read.input_reader" not in sys.modules
assert "maple.function.dispatcher.ts.ts" not in sys.modules

from maple.function.dispatcher.scan.scan import Scan
from maple.function.dispatcher.ts.algorithm.PRFO import PRFO
from maple.function.dispatcher.ts.algorithm.dimer import Dimer
from maple.function.read.command_control import CommandControl

assert Scan is not None
assert PRFO is not None
assert Dimer is not None
assert CommandControl is not None
assert "maple.function.read.input_reader" not in sys.modules
assert "maple.function.dispatcher.ts.ts" not in sys.modules

assert read_package.ExplicitSolv.__name__ == "ExplicitSolv"
for package, export_name in (
    (read_package, "InputReader"),
    (ts_package, "TransitionState"),
):
    try:
        getattr(package, export_name)
    except ModuleNotFoundError as exc:
        assert "torch" in str(exc)
    else:
        raise AssertionError(f"{export_name} unexpectedly imported without torch")
"""
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

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
