"""Guard the active generator against accidental historical cross-imports."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


DATA_GENERATION = Path(__file__).resolve().parents[1]
ACTIVE_PATHS = (
    DATA_GENERATION / "generation_core",
    DATA_GENERATION / "domain_packages",
    DATA_GENERATION / "run_data_generation.py",
    DATA_GENERATION / "run_parameter_calibration.py",
    DATA_GENERATION / "llm_client.py",
)
DISALLOWED_IMPORT_PREFIXES = (
    "legacy_stpp",
    "demo_sts_stpp",
    "stpp_adapter",
    "stpp_release_runtime",
    "stpp_stable_runtime",
    "data_generation.legacy",
)


class ActiveLayoutTests(unittest.TestCase):
    def test_active_runtime_does_not_import_historical_modules(self) -> None:
        violations = []
        for path in _active_python_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                module = ""
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith(DISALLOWED_IMPORT_PREFIXES):
                            violations.append(f"{path}: import {alias.name}")
                    continue
                if module.startswith(DISALLOWED_IMPORT_PREFIXES):
                    violations.append(f"{path}: from {module}")
        self.assertEqual(violations, [])

    def test_versioned_entrypoints_are_not_in_active_root(self) -> None:
        forbidden = []
        for pattern in (
            "demo_sts_stpp*.py",
            "stpp_*runtime.py",
            "stpp_adapter*.py",
            "run_*v1.py",
        ):
            forbidden.extend(path.name for path in DATA_GENERATION.glob(pattern))
        self.assertEqual(sorted(forbidden), [])


def _active_python_files():
    for path in ACTIVE_PATHS:
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from path.rglob("*.py")


if __name__ == "__main__":
    unittest.main()
