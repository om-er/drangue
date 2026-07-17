"""Tiny test runner so the suite works without pytest.

Discovers test_* functions in the test modules and runs them, awaiting any that
are coroutines. Run with: python run_tests.py
"""

import asyncio
import importlib
import inspect
import os
import sys

# Discovered, not listed: a new tests/test_*.py must never be silently skipped.
_TESTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
MODULES = sorted(
    f"tests.{fname[:-3]}"
    for fname in os.listdir(_TESTS_DIR)
    if fname.startswith("test_") and fname.endswith(".py")
)


def main() -> int:
    total = passed = 0
    failures = []
    for name in MODULES:
        module = importlib.import_module(name)
        for attr in sorted(dir(module)):
            if not attr.startswith("test_"):
                continue
            fn = getattr(module, attr)
            if not callable(fn):
                continue
            total += 1
            label = f"{name.split('.')[-1]}::{attr}"
            try:
                if inspect.iscoroutinefunction(fn):
                    asyncio.run(fn())
                else:
                    fn()
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
