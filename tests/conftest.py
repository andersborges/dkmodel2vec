# conftest.py
"""
Pytest configuration - only shared settings, no fixtures.
Since we only have one test file, fixtures are defined there.
"""

import pytest


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (model loading)")


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests that use the model as slow."""
    for item in items:
        # Any test using tiny_llm2vec fixture gets marked as slow
        if "tiny_llm2vec" in item.fixturenames:
            item.add_marker(pytest.mark.slow)
