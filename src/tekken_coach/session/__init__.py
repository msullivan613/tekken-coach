"""Session store: buffer, persist, and load the .jsonl event log (docs/00 §3, 03 §5)."""

from tekken_coach.session.store import (
    SCHEMA_MAJOR,
    SCHEMA_VERSION,
    IncompatibleSchemaVersionError,
    LoadedSession,
    SessionWriter,
    check_compatibility,
    iter_interactions,
    load_session,
    read_header,
)

__all__ = [
    "SCHEMA_MAJOR",
    "SCHEMA_VERSION",
    "IncompatibleSchemaVersionError",
    "LoadedSession",
    "SessionWriter",
    "check_compatibility",
    "iter_interactions",
    "load_session",
    "read_header",
]
