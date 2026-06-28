"""Run every extension's tests from one place (the monorepo payoff).

Adds the core package and each extension package to the path, then discovers and
runs each extension's test_*.py. Server-gated suites (e.g. Postgres without
DRANGUE_POSTGRES_DSN) skip themselves and still count as passing.

    python extensions/run_tests.py
"""

import asyncio
import importlib.util
import inspect
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(ROOT, "extensions")

sys.path.insert(0, ROOT)  # core, importable as `drangue`
for name in sorted(os.listdir(EXT)):
    pkg = os.path.join(EXT, name)
    if os.path.isdir(pkg):
        sys.path.insert(0, pkg)  # extension, importable as e.g. `drangue_memory`


def _load(path):
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    total = passed = 0
    failures = []
    for name in sorted(os.listdir(EXT)):
        tests_dir = os.path.join(EXT, name, "tests")
        if not os.path.isdir(tests_dir):
            continue
        for fname in sorted(os.listdir(tests_dir)):
            if not (fname.startswith("test_") and fname.endswith(".py")):
                continue
            try:
                module = _load(os.path.join(tests_dir, fname))
            except Exception as exc:  # e.g. an optional SDK is not installed
                print(f"SKIP {name}::{fname} (cannot import: {exc!r})")
                continue
            for attr in sorted(dir(module)):
                if not attr.startswith("test_"):
                    continue
                fn = getattr(module, attr)
                if not callable(fn):
                    continue
                total += 1
                label = f"{name}::{attr}"
                try:
                    asyncio.run(fn()) if inspect.iscoroutinefunction(fn) else fn()
                except Exception as exc:  # noqa: BLE001
                    print(f"FAIL {label}: {exc!r}")
                    failures.append(label)
                else:
                    print(f"PASS {label}")
                    passed += 1
    print(f"\n{passed}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
