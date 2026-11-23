import importlib.util
import os

import pytest

HAS_QANT = importlib.util.find_spec("qant_native_computing_toolkit") is not None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if HAS_QANT:
        return
    if os.environ.get("REQUIRE_QANT") == "1":
        raise pytest.UsageError("REQUIRE_QANT=1 but qant_native_computing_toolkit is not installed")
    skip = pytest.mark.skip(reason="qant toolkit not installed (scripts/install_qant_sdk.py)")
    for item in items:
        if "qant" in item.keywords:
            item.add_marker(skip)
