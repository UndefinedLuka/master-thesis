"""Static registry: dataset download specs.

URLs / Google-Drive ids change across dataset releases — they are centralized here and are
overridable via the SPIDER_GDRIVE_ID / BIRD_DEV_URL environment variables. Treat the
defaults as a starting point to VERIFY before the first run, not as guaranteed-live.

Spider 1.0 is distributed as a ~1.7 GB Google Drive zip containing
``spider_data/{database/<db_id>/<db_id>.sqlite, dev.json, tables.json, train_spider.json}``.

BIRD dev is a direct HTTPS zip on Alibaba Cloud OSS; its outer zip nests an inner
``dev_databases.zip`` that must be extracted a second time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

SourceKind = Literal["gdrive", "http"]


@dataclass(frozen=True, slots=True)
class DownloadSpec:
    """How to fetch and lay out one dataset."""

    name: str
    source: SourceKind
    url_or_id: str  # HTTP url, or Google Drive file id
    archive_name: str  # local filename for the downloaded archive
    expected_marker: str  # path (relative to the dataset dir) that must exist post-extract
    recurse_inner_zip: bool = False  # BIRD nests dev_databases.zip inside the outer zip


# --- defaults (override via env) ----------------------------------------------------------

# Spider 1.0 official Google Drive file id.
_SPIDER_GDRIVE_ID_DEFAULT = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"

# BIRD dev release zip (override via BIRD_DEV_URL if the host path changes).
_BIRD_DEV_URL_DEFAULT = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"

DATASET_NAMES = ("spider", "bird")


def get_download_spec(name: str) -> DownloadSpec:
    """The download spec for ``name``, with env overrides read at call time (not import)."""
    if name == "spider":
        return DownloadSpec(
            name="spider",
            source="gdrive",
            url_or_id=os.environ.get("SPIDER_GDRIVE_ID", _SPIDER_GDRIVE_ID_DEFAULT),
            archive_name="spider.zip",
            expected_marker="spider_data/dev.json",
        )
    if name == "bird":
        return DownloadSpec(
            name="bird",
            source="http",
            url_or_id=os.environ.get("BIRD_DEV_URL", _BIRD_DEV_URL_DEFAULT),
            archive_name="dev.zip",
            expected_marker="dev_20240627/dev.json",
            recurse_inner_zip=True,
        )
    raise ValueError(f"No download spec for dataset {name!r}")
