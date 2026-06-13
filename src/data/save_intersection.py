"""Save Books/Movies rows for users appearing in both train splits.

Outputs are written as uncompressed CSV files under ``data/``. The raw benchmark
files are streamed from the public Amazon Reviews 2023 5-core split URLs.
"""

from __future__ import annotations

from pathlib import Path
import sys
from urllib.error import URLError

import pandas as pd

try:
    from src.data.overlap_check import (
        BASE_URL,
        CHUNKSIZE,
        DOMAINS,
        load_unique_users,
        open_public_dataset_url,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.overlap_check import (
        BASE_URL,
        CHUNKSIZE,
        DOMAINS,
        load_unique_users,
        open_public_dataset_url,
    )


BOOKS_DOMAIN = "Books"
MOVIES_DOMAIN = "Movies"
SPLITS = ("train", "valid", "test")
COLUMNS = ["user_id", "parent_asin", "rating", "timestamp"]
OUTPUT_DIR = Path("data")
MAX_DOWNLOAD_ATTEMPTS = 3


def split_url(domain: str, split: str) -> str:
    """Build the URL for a domain split from the train filename in DOMAINS."""
    train_filename = DOMAINS[domain]
    split_filename = train_filename.replace(".train.csv.gz", f".{split}.csv.gz")
    return f"{BASE_URL}/{split_filename}"


def output_path(domain: str, split: str) -> Path:
    """Return the local output path for a filtered split."""
    prefix = "books" if domain == BOOKS_DOMAIN else "movies"
    return OUTPUT_DIR / f"{prefix}_{split}.csv"


def remove_empty_dirs(root: Path) -> None:
    """Remove empty child directories under root, leaving root itself in place."""
    if not root.exists():
        return

    for path in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def summarize_saved_csv(path: Path) -> dict[str, int]:
    """Summarize a saved CSV file without loading all rows at once."""
    unique_users: set[str] = set()
    unique_items: set[str] = set()
    rating_count = 0

    chunks = pd.read_csv(path, usecols=COLUMNS, chunksize=CHUNKSIZE)
    for chunk in chunks:
        unique_users.update(chunk["user_id"].dropna().unique())
        unique_items.update(chunk["parent_asin"].dropna().unique())
        rating_count += len(chunk)

    return {
        "unique_users": len(unique_users),
        "unique_items": len(unique_items),
        "ratings": rating_count,
    }


def save_filtered_split(
    domain: str,
    split: str,
    intersecting_users: set[str],
) -> dict[str, int]:
    """Stream one split, save intersecting-user rows, and return summary stats."""
    url = split_url(domain, split)
    path = output_path(domain, split)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 0:
        print(f"Skipping existing {path}.")
        return summarize_saved_csv(path)

    return stream_filtered_split(url, path, intersecting_users)


def stream_filtered_split(
    url: str,
    path: Path,
    intersecting_users: set[str],
) -> dict[str, int]:
    """Stream one URL to a filtered CSV with retry support."""
    last_error: Exception | None = None

    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        if path.exists():
            path.unlink()

        try:
            return write_filtered_split(url, path, intersecting_users)
        except (EOFError, URLError) as exc:
            last_error = exc
            print(
                f"Download failed for {path} "
                f"(attempt {attempt}/{MAX_DOWNLOAD_ATTEMPTS}): {exc}"
            )

    raise RuntimeError(f"Failed to save {path} after retries.") from last_error


def write_filtered_split(
    url: str,
    path: Path,
    intersecting_users: set[str],
) -> dict[str, int]:
    """Write one filtered split and return summary stats."""
    unique_users: set[str] = set()
    unique_items: set[str] = set()
    rating_count = 0
    wrote_header = False

    with open_public_dataset_url(url) as response:
        chunks = pd.read_csv(
            response,
            usecols=COLUMNS,
            chunksize=CHUNKSIZE,
            compression="gzip",
        )
        for chunk in chunks:
            filtered = chunk[chunk["user_id"].isin(intersecting_users)]
            if filtered.empty:
                continue

            filtered.to_csv(
                path,
                mode="w" if not wrote_header else "a",
                index=False,
                header=not wrote_header,
            )
            wrote_header = True

            unique_users.update(filtered["user_id"].dropna().unique())
            unique_items.update(filtered["parent_asin"].dropna().unique())
            rating_count += len(filtered)

    if not wrote_header:
        pd.DataFrame(columns=COLUMNS).to_csv(path, index=False)

    return {
        "unique_users": len(unique_users),
        "unique_items": len(unique_items),
        "ratings": rating_count,
    }


def print_summary(path: Path, summary: dict[str, int]) -> None:
    """Print summary stats for one saved file."""
    print(
        f"{path}: "
        f"{summary['unique_users']:,} unique users, "
        f"{summary['unique_items']:,} unique items, "
        f"{summary['ratings']:,} ratings"
    )


def main() -> None:
    books_train_url = f"{BASE_URL}/{DOMAINS[BOOKS_DOMAIN]}"
    movies_train_url = f"{BASE_URL}/{DOMAINS[MOVIES_DOMAIN]}"

    print("Loading unique users from Books and Movies train splits...")
    books_users = load_unique_users(books_train_url)
    movies_users = load_unique_users(movies_train_url)
    intersecting_users = books_users & movies_users
    print(f"Books ∩ Movies train users: {len(intersecting_users):,}")

    summaries: list[tuple[Path, dict[str, int]]] = []
    for domain in (BOOKS_DOMAIN, MOVIES_DOMAIN):
        for split in SPLITS:
            path = output_path(domain, split)
            print(f"Saving {path}...")
            summary = save_filtered_split(domain, split, intersecting_users)
            summaries.append((path, summary))

    remove_empty_dirs(OUTPUT_DIR)

    print("\nSaved file summary:")
    for path, summary in summaries:
        print_summary(path, summary)


if __name__ == "__main__":
    main()
