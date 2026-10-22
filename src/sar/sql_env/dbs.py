"""Where each database lives: (family, db_id) -> path of the SQLite file.

Every training or eval example names its database by a family (which
benchmark it comes from) and a db_id. Paths are resolved here and nowhere
else, against one root (the OmniSQL ``data/`` directory) that differs between
a laptop, a cluster node's local disk and a container.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# Relative to the OmniSQL data/ directory (OmniSQL-datasets/data.zip).
FAMILY_DIRS = {
    "synsql": "SynSQL-2.5M/databases",
    "spider": "spider/database",  # Spider train + dev; Spider-Syn and Spider-Realistic reuse it
    "spider_test": "spider/test_database",
    "spider_dk": "Spider-DK/database",
    "bird_train": "bird/train/train_databases",
    "bird_dev": "bird/dev_20240627/dev_databases",
}

_DB_ID = re.compile(r"^[A-Za-z0-9_\-.]+$")


def db_root() -> Path:
    root = os.environ.get("SAR_DB_ROOT")
    if not root:
        raise RuntimeError("SAR_DB_ROOT is not set (the OmniSQL data/ directory)")
    return Path(root)


def resolve(family: str, db_id: str, root: str | Path | None = None) -> Path:
    """Path of the database file. db_id comes from data files, so it is
    checked before it becomes part of a path."""
    if family not in FAMILY_DIRS:
        raise ValueError(f"unknown database family {family!r}")
    if not _DB_ID.match(db_id) or ".." in db_id:
        raise ValueError(f"invalid db_id {db_id!r}")
    base = Path(root) if root is not None else db_root()
    return base / FAMILY_DIRS[family] / db_id / f"{db_id}.sqlite"
