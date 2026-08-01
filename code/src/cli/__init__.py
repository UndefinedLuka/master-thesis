"""Command-line entry points. The only layer that wires config to the other packages.

Each module exposes ``main(argv) -> int`` and is runnable as ``python -m cli.<x>``
or via the console scripts declared in pyproject (``thesis-download`` etc.).
"""
