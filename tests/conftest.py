from __future__ import annotations

import pytest

from amos import Amos


@pytest.fixture()
def amos(tmp_path):
    service = Amos(tmp_path / "amos.sqlite3")
    try:
        yield service
    finally:
        service.close()
