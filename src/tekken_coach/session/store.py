"""On-disk session event log — the .jsonl contract (docs/03 §5, §6).

One session = one JSON Lines file. Line 1 is a :class:`SessionHeader`; every subsequent
line is one :class:`LabeledInteraction`. Append-only, flushed at round end so a crash
mid-match loses at most one round (00 §4).

This module owns the write and read sides plus the ``schema_version`` compatibility gate
(03 §6): a log whose header major version is unknown is rejected; an additive minor bump is
tolerated (unknown additive fields are ignored by the models).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from tekken_coach.schemas import LabeledInteraction, SessionHeader

# The schema version this build emits. Semver: MAJOR = breaking, MINOR = additive (03 §6).
# 1.1.0: additive — Interaction gained attacker_char_id / defender_char_id (03 §2, C2/05 §4).
SCHEMA_VERSION = "1.1.0"


def _major(version: str) -> int:
    """Return the semver MAJOR component of ``version`` (e.g. "1.2.3" -> 1)."""
    try:
        return int(version.split(".", 1)[0])
    except (ValueError, IndexError) as exc:
        raise IncompatibleSchemaVersionError(
            f"malformed schema_version {version!r} (expected semver like '1.0.0')"
        ) from exc


SCHEMA_MAJOR = _major(SCHEMA_VERSION)


class IncompatibleSchemaVersionError(Exception):
    """Raised when a session log's ``schema_version`` is not compatible with this build."""


def check_compatibility(schema_version: str) -> None:
    """Gate a log's ``schema_version`` against this build (03 §6).

    Compatible iff the MAJOR version matches this build's. A differing MINOR (additive) is
    tolerated in either direction: newer additive fields are ignored on load, and older
    logs simply omit fields. A differing MAJOR is a breaking change and is rejected.
    """
    if _major(schema_version) != SCHEMA_MAJOR:
        raise IncompatibleSchemaVersionError(
            f"incompatible schema_version {schema_version!r}: "
            f"log major {_major(schema_version)} != supported major {SCHEMA_MAJOR}"
        )


# ---------------------------------------------------------------------------
# Write side
# ---------------------------------------------------------------------------


class SessionWriter:
    """Append-only writer for a session .jsonl log.

    The header is written immediately on open (line 1). Interactions are buffered in memory
    and written to disk on :meth:`flush` (call at round end). Usable as a context manager;
    :meth:`close` flushes any remaining buffer.
    """

    def __init__(self, path: str | Path, header: SessionHeader) -> None:
        self.path = Path(path)
        self.header = header
        self._buffer: list[LabeledInteraction] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self._fh.write(header.model_dump_json() + "\n")
        self._fh.flush()

    def append(self, interaction: LabeledInteraction) -> None:
        """Buffer one labeled interaction; written to disk on the next :meth:`flush`."""
        self._buffer.append(interaction)

    def flush(self) -> None:
        """Write buffered interactions to disk and flush the file (round-end flush)."""
        if not self._buffer:
            return
        lines = "".join(i.model_dump_json() + "\n" for i in self._buffer)
        self._fh.write(lines)
        self._fh.flush()
        self._buffer.clear()

    def close(self) -> None:
        """Flush remaining buffer and close the underlying file."""
        if self._fh.closed:
            return
        self.flush()
        self._fh.close()

    def __enter__(self) -> SessionWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Read side
# ---------------------------------------------------------------------------


@dataclass
class LoadedSession:
    """A fully-loaded session: its header plus every labeled interaction."""

    header: SessionHeader
    interactions: list[LabeledInteraction]


def read_header(path: str | Path) -> SessionHeader:
    """Read and validate line 1 of a session log, gating its ``schema_version`` (03 §6)."""
    with Path(path).open("r", encoding="utf-8") as fh:
        first = fh.readline()
    if not first.strip():
        raise ValueError(f"session log {path} is empty or missing its header line")
    header = SessionHeader.model_validate_json(first)
    check_compatibility(header.schema_version)
    return header


def iter_interactions(path: str | Path) -> Iterator[LabeledInteraction]:
    """Stream the body (lines 2..N) of a session log as :class:`LabeledInteraction`.

    Validates the header's ``schema_version`` before yielding any interaction.
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        first = fh.readline()
        if not first.strip():
            raise ValueError(f"session log {path} is empty or missing its header line")
        header = SessionHeader.model_validate_json(first)
        check_compatibility(header.schema_version)
        for line in fh:
            if not line.strip():
                continue
            yield LabeledInteraction.model_validate_json(line)


def load_session(path: str | Path) -> LoadedSession:
    """Load a whole session log into memory (header + all interactions), version-gated."""
    header = read_header(path)
    interactions = [LabeledInteraction.model_validate_json(line) for line in _body_lines(path)]
    return LoadedSession(header=header, interactions=interactions)


def _body_lines(path: str | Path) -> Iterator[str]:
    """Yield non-blank body lines (skipping the header) of a session log."""
    with Path(path).open("r", encoding="utf-8") as fh:
        fh.readline()  # skip header
        for line in fh:
            if line.strip():
                yield line
