"""Maps each ``SerializationFormat`` to its serializer instance.

The registry is the single lookup point so callers select a serializer by enum —
``get_serializer(SerializationFormat.YAML)`` — and an unsupported format fails
loudly rather than silently producing nothing.
"""

from __future__ import annotations

from ..enums import SerializationFormat
from ..errors import SerializationError
from .base import Serializer
from .csv_fmt import CsvSerializer
from .json_fmt import JsonSerializer
from .markdown_fmt import MarkdownSerializer
from .snapshot import DatabaseSnapshot
from .toon_fmt import ToonSerializer
from .xml_fmt import XmlSerializer
from .yaml_fmt import YamlSerializer

_REGISTRY: dict[SerializationFormat, Serializer] = {
    SerializationFormat.JSON: JsonSerializer(),
    SerializationFormat.XML: XmlSerializer(),
    SerializationFormat.YAML: YamlSerializer(),
    SerializationFormat.CSV: CsvSerializer(),
    SerializationFormat.MARKDOWN: MarkdownSerializer(),
    SerializationFormat.TOON: ToonSerializer(),
}


def get_serializer(fmt: SerializationFormat) -> Serializer:
    try:
        return _REGISTRY[fmt]
    except KeyError as exc:  # pragma: no cover - guarded by the enum
        raise SerializationError(f"No serializer registered for {fmt!r}") from exc


def serialize(fmt: SerializationFormat, snapshot: DatabaseSnapshot) -> str:
    """Convenience: render ``snapshot`` in ``fmt``, wrapping encoder errors."""
    try:
        return get_serializer(fmt).serialize(snapshot)
    except SerializationError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize any encoder failure
        raise SerializationError(
            f"Failed to serialize db {snapshot.db_id!r} as {fmt.value}: {exc}"
        ) from exc
