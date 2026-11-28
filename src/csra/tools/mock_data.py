"""Read-only loader for simulated Stage 1 mock data."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_STORE_PATH = Path("data/mock/store.json")


@lru_cache
def load_mock_store(path: str | Path = DEFAULT_STORE_PATH) -> dict[str, Any]:
    """Load the simulated customer, order, payment, and delivery store."""
    store_path = Path(path)
    with store_path.open("r", encoding="utf-8") as f:
        return json.load(f)
