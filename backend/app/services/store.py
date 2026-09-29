"""In-memory store for MVP. Later replace with SQLite/Postgres."""
from typing import Dict

DB: Dict[str, dict] = {}
_counter = 0


def next_id() -> str:
    global _counter
    _counter += 1
    # Format: AD-2026-000001
    return f"AD-2026-{_counter:06d}"
