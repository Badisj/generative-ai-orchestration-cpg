# FILE: tests/conftest.py
"""
Pytest configuration and fixtures.
"""

import pytest
import asyncio
from pathlib import Path


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "Vitamin C is an essential nutrient found in citrus fruits.",
        "The fermentation process converts sugars into alcohol.",
        "Emulsifiers help mix oil and water in food products.",
        "Antioxidants prevent oxidation and extend shelf life.",
    ]


@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"
