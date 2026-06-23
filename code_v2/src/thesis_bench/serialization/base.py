"""The serializer boundary.

A ``Serializer`` turns a ``DatabaseSnapshot`` into a single text blob suitable for
embedding in a prompt. It is a ``Protocol`` so each format is just a small object
(or any callable-bearing class) without an inheritance requirement, and so the
registry can treat them uniformly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..enums import SerializationFormat
from .snapshot import DatabaseSnapshot


@runtime_checkable
class Serializer(Protocol):
    @property
    def fmt(self) -> SerializationFormat: ...

    def serialize(self, snapshot: DatabaseSnapshot) -> str:
        """Render the snapshot's table contents as text in this format."""
        ...
