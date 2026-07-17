"""Richer @tool schemas: docstring Args, Enum, defaults, and honest warnings.

Schema quality is prompt quality; these lock in what the model is shown.
"""

import enum
import warnings

from drangue import tool


def test_docstring_args_become_parameter_descriptions():
    @tool
    def deploy(service: str, replicas: int = 2) -> str:
        """Deploy a service.

        Args:
            service: The service name, e.g. "checkout".
            replicas: How many instances to run; keep it small
                for staging environments.

        Returns:
            A deploy id.
        """
        return "ok"

    props = deploy.to_schema()["input_schema"]["properties"]
    assert props["service"]["description"] == 'The service name, e.g. "checkout".'
    # Wrapped continuation lines are joined.
    assert props["replicas"]["description"] == (
        "How many instances to run; keep it small for staging environments.")
    assert props["replicas"]["default"] == 2


def test_enum_annotations_become_enum_schemas():
    class Env(enum.Enum):
        STAGING = "staging"
        PROD = "prod"

    @tool
    def switch(env: Env = Env.STAGING) -> str:
        """Switch environment."""
        return "ok"

    prop = switch.to_schema()["input_schema"]["properties"]["env"]
    assert prop["enum"] == ["staging", "prod"]
    assert prop["default"] == "staging"


def test_additional_properties_are_rejected_by_the_schema():
    @tool
    def f(a: int) -> str:
        """Doc."""
        return "ok"

    assert f.to_schema()["input_schema"]["additionalProperties"] is False


def test_unmappable_annotations_warn_instead_of_silently_degrading():
    class Custom:
        pass

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @tool
        def f(x: Custom) -> str:
            """Doc."""
            return "ok"

    assert any("Custom" in str(w.message) and "strings" in str(w.message)
               for w in caught)
    # The schema still degrades predictably.
    assert f.to_schema()["input_schema"]["properties"]["x"] == {"type": "string"}


def test_mapped_annotations_do_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @tool
        def f(a: int, b: str | None = None, c: list[int] = None) -> str:
            """Doc."""
            return "ok"

    assert not [w for w in caught if "advertised" in str(w.message)]
