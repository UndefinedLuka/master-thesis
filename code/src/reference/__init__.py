"""Reference-answer generation: execute gold SQL to obtain ground-truth result rows."""

from reference.answers import (
    ReferenceAnswer,
    build_reference_set,
    generate_reference_answer,
    load_reference_file,
)

__all__ = [
    "ReferenceAnswer",
    "generate_reference_answer",
    "build_reference_set",
    "load_reference_file",
]
