from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


DB_PATH = Path("instance/app.db")


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                filename TEXT NOT NULL,
                rows_total INTEGER,
                rows_after_preprocessing INTEGER,
                probability REAL NOT NULL,
                predicted_class INTEGER NOT NULL,
                predicted_label TEXT NOT NULL,
                threshold REAL NOT NULL,
                top_features_json TEXT
            )
            """
        )
        conn.commit()


def save_prediction(
    filename: str,
    rows_total: int,
    rows_after_preprocessing: int,
    probability: float,
    predicted_class: int,
    predicted_label: str,
    threshold: float,
    top_features: list[dict[str, Any]] | None = None,
) -> int:
    payload = json.dumps(top_features or [], ensure_ascii=False)

    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions (
                filename,
                rows_total,
                rows_after_preprocessing,
                probability,
                predicted_class,
                predicted_label,
                threshold,
                top_features_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                rows_total,
                rows_after_preprocessing,
                probability,
                predicted_class,
                predicted_label,
                threshold,
                payload,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_prediction_history(limit: int = 100) -> list[sqlite3.Row]:
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT *
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return cur.fetchall()


def get_prediction_by_id(prediction_id: int) -> sqlite3.Row | None:
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT *
            FROM predictions
            WHERE id = ?
            """,
            (prediction_id,),
        )
        return cur.fetchone()