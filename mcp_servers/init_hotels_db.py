"""Initialize the SQLite hotels database for MCP Toolbox."""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "toolbox_hotels.db"

_HOTELS = (
    (1, "Hilton Basel", "Basel", "Luxury", 0),
    (2, "Marriott Zurich", "Zurich", "Upscale", 0),
    (3, "Hyatt Regency Basel", "Basel", "Upper Upscale", 0),
)


def init_hotels_db(db_path: Path | None = None) -> Path:
    """Create the hotels table and seed sample rows."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hotels(
              id INTEGER NOT NULL PRIMARY KEY,
              name TEXT NOT NULL,
              location TEXT NOT NULL,
              price_tier TEXT NOT NULL,
              booked INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO hotels(id, name, location, price_tier, booked)
            VALUES (?, ?, ?, ?, ?)
            """,
            _HOTELS,
        )
        conn.commit()

    return path


if __name__ == "__main__":
    created = init_hotels_db()
    print(f"Initialized hotels database at {created}")
