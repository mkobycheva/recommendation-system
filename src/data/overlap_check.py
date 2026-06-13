"""Check cross-domain user overlap in Amazon Reviews 2023 5-core train splits.

The script streams only the ``user_id`` column from the public benchmark CSV
files and prints overlap counts. It does not write any files locally.
"""

from __future__ import annotations

from itertools import combinations
import ssl
from urllib.request import urlopen

import pandas as pd


BASE_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
    "benchmark/5core/last_out_w_his"
)

DOMAINS = {
    "Books": "Books.train.csv.gz",
    "Movies": "Movies_and_TV.train.csv.gz",
    "CDs": "CDs_and_Vinyl.train.csv.gz",
}

CHUNKSIZE = 500_000
USER_ID_COLUMN = "user_id"


def open_public_dataset_url(url: str):
    """Open a public dataset URL for streaming.

    Some local Python installs do not ship with usable CA certificates. The
    Amazon benchmark files are public and immutable for this analysis, so this
    uses an unverified context to keep the script runnable without extra
    dependencies.
    """
    context = ssl._create_unverified_context()
    return urlopen(url, context=context)


def load_unique_users(url: str, chunksize: int = CHUNKSIZE) -> set[str]:
    """Load unique user IDs from a compressed CSV URL."""
    users: set[str] = set()

    with open_public_dataset_url(url) as response:
        chunks = pd.read_csv(
            response,
            usecols=[USER_ID_COLUMN],
            chunksize=chunksize,
            compression="gzip",
        )
        for chunk in chunks:
            users.update(chunk[USER_ID_COLUMN].dropna().unique())

    return users


def format_count(value: int) -> str:
    """Return an integer with thousands separators."""
    return f"{value:,}"


def overlap_pct(overlap_count: int, domain_sizes: list[int]) -> float:
    """Calculate overlap as a percentage of the smallest involved domain."""
    smallest_domain_size = min(domain_sizes)
    if smallest_domain_size == 0:
        return 0.0
    return overlap_count / smallest_domain_size * 100


def print_overlap(label: str, overlap_count: int, domain_sizes: list[int]) -> None:
    """Print one overlap line with count and percentage."""
    pct = overlap_pct(overlap_count, domain_sizes)
    print(f"{label}: {format_count(overlap_count)} ({pct:.2f}% of smallest domain)")


def main() -> None:
    users_by_domain: dict[str, set[str]] = {}

    print("Loading unique users from Amazon Reviews 2023 5-core train splits...")
    for domain, filename in DOMAINS.items():
        url = f"{BASE_URL}/{filename}"
        users = load_unique_users(url)
        users_by_domain[domain] = users
        print(f"{domain}: {format_count(len(users))} unique users")

    print("\nPairwise overlap:")
    for left, right in combinations(DOMAINS, 2):
        overlap_count = len(users_by_domain[left] & users_by_domain[right])
        domain_sizes = [len(users_by_domain[left]), len(users_by_domain[right])]
        print_overlap(f"{left} ∩ {right}", overlap_count, domain_sizes)

    print("\nThree-way overlap:")
    domain_names = list(DOMAINS)
    three_way_overlap = set.intersection(
        *(users_by_domain[domain] for domain in domain_names)
    )
    domain_sizes = [len(users_by_domain[domain]) for domain in domain_names]
    print_overlap("Books ∩ Movies ∩ CDs", len(three_way_overlap), domain_sizes)


if __name__ == "__main__":
    main()
