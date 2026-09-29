"""Append episode returns to the project's scores table.

The original stub already had ``data/scores.db`` with columns
``date TEXT``, ``scores REAL``, ``time REAL``. Training writes that same shape:
calendar date, episode return, and a Unix timestamp.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def log_score(score: float, path: str | Path = "data/scores.db") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS scores ("
            "date TEXT, scores REAL, time REAL)"
        )
        con.execute(
            "INSERT INTO scores (date, scores, time) VALUES (?, ?, ?)",
            (
                datetime.now(timezone.utc).date().isoformat(),
                float(score),
                datetime.now(timezone.utc).timestamp(),
            ),
        )
