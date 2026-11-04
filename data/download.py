#!/usr/bin/env python3
"""Download the UCI Poker Hand dataset (id 158) into data/.

Fetches the official static bundle from archive.ics.uci.edu, extracts the
three files, and writes sha256 checksums next to them. Raw data is never
committed to git (see .gitignore); this script is the reproducible source.

Deliberately does not parse or print any row contents: it only moves bytes,
counts lines/columns structurally, and hashes files. See DECISIONS.md for why
that distinction matters for this project's Phase 1 label-blindness claim.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/158/poker+hand.zip"
DATA_DIR = Path(__file__).resolve().parent
EXPECTED_MEMBERS = [
    "poker-hand-training-true.data",
    "poker-hand-testing.data",
    "poker-hand.names",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    zip_path = DATA_DIR / "poker-hand.zip"
    print(f"Downloading {UCI_ZIP_URL} ...")
    urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)

    print(f"Extracting to {DATA_DIR} ...")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        missing = [m for m in EXPECTED_MEMBERS if m not in names]
        if missing:
            raise SystemExit(f"zip is missing expected members: {missing}, got {names}")
        zf.extractall(DATA_DIR)

    checksums_path = DATA_DIR / "CHECKSUMS.txt"
    lines = []
    for member in EXPECTED_MEMBERS:
        p = DATA_DIR / member
        digest = sha256_of(p)
        n_lines = sum(1 for _ in p.open("rb"))
        lines.append(f"{digest}  {member}  ({n_lines} lines)")
        print(lines[-1])
    checksums_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {checksums_path}")

    # Structural-only sanity check: column COUNT of the first data row, never
    # its values. This confirms the 10-feature + 1-class layout documented in
    # poker-hand.names without reading what any value means.
    for member in ("poker-hand-training-true.data", "poker-hand-testing.data"):
        p = DATA_DIR / member
        with p.open() as f:
            first = f.readline()
        n_cols = len(first.strip().split(","))
        print(f"{member}: first row has {n_cols} columns (expected 11)")
        if n_cols != 11:
            raise SystemExit(f"{member}: unexpected column count {n_cols}")


if __name__ == "__main__":
    main()
