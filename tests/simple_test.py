import pytest


@pytest.mark.fast
def test_simple() -> None:
    """Simple Test Example"""
    assert 1 == 0 + 1
