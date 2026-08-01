"""Serialization of database contents into the six experiment formats.

Each ``to_*`` function is a pure ``DatabaseContents -> str`` mapping; :func:`serialize`
dispatches on the format name.
"""

from serialization.csv_fmt import to_csv
from serialization.dispatch import serialize
from serialization.json_fmt import to_json
from serialization.markdown_fmt import to_markdown
from serialization.toon_fmt import to_toon
from serialization.types import (
    FORMAT_EXTENSIONS,
    FORMATS,
    DatabaseContents,
    Row,
    SerializationFormat,
)
from serialization.xml_fmt import to_xml
from serialization.yaml_fmt import to_yaml

__all__ = [
    "serialize",
    "to_json",
    "to_xml",
    "to_yaml",
    "to_csv",
    "to_markdown",
    "to_toon",
    "SerializationFormat",
    "DatabaseContents",
    "Row",
    "FORMATS",
    "FORMAT_EXTENSIONS",
]
