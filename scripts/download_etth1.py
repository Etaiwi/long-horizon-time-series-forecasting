"""Download and validate the public ETTh1 dataset."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from urllib.request import urlopen

from ts_project.data.etth1 import load_etth1

SOURCE_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/"
    "main/ETT-small/ETTh1.csv"
)
EXPECTED_SHA256 = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "ETTh1.csv"


def verify_file(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(
            f"ETTh1 SHA-256 mismatch: {digest}; expected {EXPECTED_SHA256}."
        )
    load_etth1(path)


def download(output: Path, *, force: bool = False) -> None:
    output = output.resolve()
    if output.exists() and not force:
        verify_file(output)
        frame = load_etth1(output)
        print(f"ETTh1 already exists and is valid: {output} ({len(frame):,} rows)")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".csv.part")
    try:
        print(f"Downloading ETTh1 from {SOURCE_URL}")
        with urlopen(SOURCE_URL, timeout=60) as response:
            with temporary.open("wb") as destination:
                shutil.copyfileobj(response, destination)
        temporary.replace(output)
        verify_file(output)
        frame = load_etth1(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        raise

    print(
        f"Saved and validated {len(frame):,} hourly rows "
        f"from {frame.index.min()} to {frame.index.max()} at {output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    download(arguments.output, force=arguments.force)


if __name__ == "__main__":
    main()
