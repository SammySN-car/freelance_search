import pytest

from freelance_search.db import Database


@pytest.fixture
def database(tmp_path):
    database = Database(tmp_path / "test.db")
    database.initialize()
    return database
