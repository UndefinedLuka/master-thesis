"""Single entry point that maps a format name to its encoder."""

from __future__ import annotations

from collections.abc import Callable

from serialization.csv_fmt import to_csv
from serialization.json_fmt import to_json
from serialization.markdown_fmt import to_markdown
from serialization.toon_fmt import to_toon
from serialization.types import DatabaseContents, SerializationFormat
from serialization.xml_fmt import to_xml
from serialization.yaml_fmt import to_yaml

_ENCODERS: dict[SerializationFormat, Callable[[DatabaseContents], str]] = {
    "json": to_json,
    "xml": to_xml,
    "yaml": to_yaml,
    "csv": to_csv,
    "markdown": to_markdown,
    "toon": to_toon,
}


def serialize(contents: DatabaseContents, fmt: SerializationFormat) -> str:
    """Serialize a database into the requested format."""
    try:
        encoder = _ENCODERS[fmt]
    except KeyError:
        raise ValueError(f"Unknown serialization format: {fmt!r}") from None
    return encoder(contents)
