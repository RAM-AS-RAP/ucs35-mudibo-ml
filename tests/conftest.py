"""Keep explicitly marked real-database tests opt-in."""

import pytest


def pytest_addoption(parser):
    parser.addoption("--run-db", action="store_true", default=False, help="Enable real-database tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-db"):
        return
    skip = pytest.mark.skip(reason="Real database tests require --run-db")
    for item in items:
        if "db" in item.keywords:
            item.add_marker(skip)