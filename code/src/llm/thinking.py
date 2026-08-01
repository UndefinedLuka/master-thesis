"""Reasoning-time measurement for thinking models (Chapter 3 §System Latency).

Reasoning models emit an internal monologue wrapped in ``<think> ... </think>`` before the
usable answer. Chapter 3 splits generation into a *reasoning* phase and a *generation* phase;
this tracker consumes streamed content deltas, stamps the moment the closing tag completes
(reasoning ends), and exposes the answer with the think block stripped so raw outputs and
accuracy scoring see only the usable response.

Non-thinking models never produce the tag: ``reasoning_time_s`` is 0.0 and the text passes
through unchanged.
"""

from __future__ import annotations

import re
import time

_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class ThinkTracker:
    """Accumulates streamed deltas and times the ``</think>`` boundary.

    The closing tag can be split across deltas, so detection searches the accumulated tail
    (bounded — the search window never exceeds delta length + tag length, keeping feed O(1)).
    """

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._tail = ""  # last few chars, for tag detection across delta boundaries
        self._first_feed_at: float | None = None
        self._close_seen_at: float | None = None

    def feed(self, content: str) -> None:
        now = time.perf_counter()
        if self._first_feed_at is None:
            self._first_feed_at = now
        self._parts.append(content)
        if self._close_seen_at is None:
            window = self._tail + content
            if _CLOSE_TAG in window:
                self._close_seen_at = now
            self._tail = window[-(len(_CLOSE_TAG) - 1) :]

    def raw_text(self) -> str:
        """The full generated text, think block included (token accounting uses this)."""
        return "".join(self._parts)

    def visible_text(self) -> str:
        """The usable answer: the think block (or an unopened ``</think>`` prefix) removed.

        A think block that never closes (generation truncated by ``max_tokens`` mid-thought)
        is stripped entirely — leaking the raw monologue into the recorded answer would let
        accuracy metrics score reasoning text instead of a usable response.
        """
        text = self.raw_text()
        if _CLOSE_TAG not in text:
            open_at = text.find(_OPEN_TAG)
            if open_at != -1:  # truncated mid-think: drop the unfinished monologue
                return text[:open_at].rstrip()
            return text
        stripped = _THINK_BLOCK.sub("", text, count=1)
        if _CLOSE_TAG in stripped:  # closing tag without an opening one (template quirk)
            stripped = stripped.split(_CLOSE_TAG, 1)[1].lstrip()
        return stripped

    def reasoning_time_s(self) -> float:
        """Seconds from the first output token to the end of the think block; 0.0 if none."""
        if self._first_feed_at is None or self._close_seen_at is None:
            return 0.0
        return self._close_seen_at - self._first_feed_at
