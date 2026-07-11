"""Runtime configuration, read from the environment at call time (not import time)
so tests can point ARTIFACTS_DIR at a synthetic fixture directory before startup.
"""

import os
from pathlib import Path


def get_artifacts_dir() -> Path:
    return Path(os.environ.get("ARTIFACTS_DIR", "./artifacts"))


def get_items_metadata_path() -> Path:
    default = get_artifacts_dir() / "items_metadata.parquet"
    return Path(os.environ.get("ITEMS_METADATA_PATH", str(default)))


def get_model_dir(model_name: str) -> Path:
    return get_artifacts_dir() / model_name
