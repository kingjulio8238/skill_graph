import pytest
from pathlib import Path


@pytest.fixture
def fixtures_dir():
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def skills_dir(fixtures_dir):
    return fixtures_dir / "skills"
