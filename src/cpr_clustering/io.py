"""Input and output helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def read_table(path: str | Path, index_column: Optional[str] = None) -> pd.DataFrame:
    """Read a CSV or Excel file and optionally set an index column."""

    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    if index_column and index_column in df.columns:
        df = df.set_index(index_column)
    return df


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_table(df: pd.DataFrame, path: str | Path, allow_private_output: bool = False) -> None:
    """Save a table only when private-output writing is explicitly allowed."""

    if not allow_private_output:
        return

    path = Path(path)
    ensure_dir(path.parent)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=True, encoding="utf-8-sig")
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(path, index=True)
    else:
        raise ValueError(f"Unsupported output file type: {path.suffix}")
