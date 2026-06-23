"""Build a ``Dataset`` from a ``DatasetConfig`` by kind."""

from __future__ import annotations

from ..config import DatasetConfig
from ..enums import DatasetKind
from .base import Dataset
from .bird import BirdDataset
from .spider import SpiderDataset


def build_dataset(config: DatasetConfig) -> Dataset:
    match config.kind:
        case DatasetKind.SPIDER:
            return SpiderDataset(config)
        case DatasetKind.BIRD:
            return BirdDataset(config)
