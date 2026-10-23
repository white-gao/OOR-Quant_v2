"""Explicitly download ONLY the three auxiliary RecIF test parquet files."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from shared.paths import benchmark_data_root
from .recif_tasks import TASKS, DATASET_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=benchmark_data_root())
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--revision", default=DATASET_REVISION)
    args = parser.parse_args()
    # Download beside (not into) AD/product/video. Never replace existing data.
    # local_dir preserves the official benchmark_data/<task>/... hierarchy.
    if args.data_dir.name != "benchmark_data":
        parser.error("data_dir must end in benchmark_data to preserve the upstream layout")
    for task in args.tasks:
        target = args.data_dir / task / f"{task}_test.parquet"
        if target.exists():
            print(f"[keep] {target}")
            continue
        try:
            path = hf_hub_download(repo_id="OpenOneRec/OpenOneRec-RecIF", repo_type="dataset",
                                   filename=f"benchmark_data/{task}/{task}_test.parquet",
                                   revision=args.revision, local_dir=str(args.data_dir.parent))
        except HfHubHTTPError as exc:
            if exc.response is not None and exc.response.status_code in (401, 403):
                raise SystemExit(
                    "RecIF is gated: request/accept access at "
                    "https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF, "
                    "then run `hf auth login` with an authorized account and retry. "
                    "Existing data are preserved.") from None
            raise
        print(f"[downloaded] {path}")


if __name__ == "__main__":
    main()
