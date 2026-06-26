"""Tiny test runner so the suite works without pytest.

Discovers test_* functions in the test modules and runs them, awaiting any that
are coroutines. Run with: python run_tests.py
"""

import asyncio
import importlib
import inspect
import sys

MODULES = [
    "tests.test_agent",
    "tests.test_openai_model",
    "tests.test_engine",
    "tests.test_observability",
    "tests.test_durability",
    "tests.test_hardening",
    "tests.test_cost",
    "tests.test_security",
    "tests.test_rollout",
]


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
