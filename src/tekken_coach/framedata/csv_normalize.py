"""Normalize ``pbruvoll/tekkendocs`` Wavu-converted CSV rows into the docs/05 §3.2 schema.

This is the pure, deterministic core of the ``fetch-framedata`` ingest (docs/05 §3.3),
separated from any I/O so it can be unit-tested against a recorded CSV fixture (mock HTTP).

The source is semicolon-delimited CSV with this header (docs/05 §3.1):

    Command; Hit level; Damage; Start up frame; Block frame; Hit frame; Counter hit frame;
    Notes; Tags; Transitions; Name; Recovery; Image; Video; Wavu id; Character id

Real-data shapes this handles (verified against the pinned snapshot):

* Multi-hit strings carry comma-lists per cell, e.g. Paul ``df+1,1,2`` -> hit level ``m, h, m``,
  startup ``i14, ,i22~23`` (note the **blank middle cell** -> ``None`` per-hit startup).
* A move with commas in its *command* is not necessarily a string: EWGF ``f,n,d,df+2`` has a
  single hit level ``h`` -> a single move. String detection keys on the **hit-level token
  count**, never on commas in the command.
* Frame cells carry annotations: startup ``i15~16`` (range) / ``i`` prefix; block ``+0c`` /
  ``-11a`` / ``-13~-8`` (range); hit/CH ``+32a (+24)``. We parse the leading signed integer and
  preserve the raw cell.
* Hit-level tokens are richer than the five-value enum: ``m! M h! sp sl t th th(h)`` and case
  variants. We map recognizable heights to :class:`MoveProperty` and preserve the raw token.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Iterable, Iterator, Sequence

from tekken_coach.framedata.models import CharFrameData, FrameDataMove, Hit
from tekken_coach.schemas import MoveProperty

# The exact source header, in order (docs/05 §3.1). Used to locate columns by name so a
# column reorder upstream is tolerated rather than silently misread.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "Command",
    "Hit level",
    "Damage",
    "Start up frame",
    "Block frame",
    "Hit frame",
    "Counter hit frame",
    "Notes",
    "Tags",
    "Transitions",
    "Name",
    "Recovery",
    "Image",
    "Video",
    "Wavu id",
    "Character id",
)


class CsvFormatError(Exception):
    """Raised when the source CSV header is missing a required column (docs/05 §3.3 guardrail)."""


# Map a raw Wavu hit-level token to the five-value MoveProperty enum where the height is clear.
# Unrecognized tokens (e.g. "sp", stance markers) map to None; the raw token is always preserved.
_HIT_LEVEL_MAP: dict[str, MoveProperty] = {
    "h": MoveProperty.high,
    "m": MoveProperty.mid,
    "l": MoveProperty.low,
    "t": MoveProperty.throw,
    "th": MoveProperty.throw,
    "sm": MoveProperty.mid,  # special mid — blocked as a mid
    "sl": MoveProperty.low,  # special low
    "sh": MoveProperty.high,  # special high
}

_LEAD_INT = re.compile(r"[+-]?\d+")


def parse_hit_level(token: str) -> MoveProperty | None:
    """Map a raw hit-level token to :class:`MoveProperty`, or ``None`` if unmappable.

    Strips ``!``/``*`` power/break markers and a ``(...)`` suffix (e.g. ``th(h)`` -> ``th``),
    lowercases, then looks up the canonical height. Preserving the raw token is the caller's
    job; this only decides the enum mapping.
    """
    base = token.strip().rstrip("!*").split("(", 1)[0].strip().lower()
    return _HIT_LEVEL_MAP.get(base)


def parse_frames(cell: str) -> int | None:
    """Return the leading signed integer of a frame cell, ignoring annotations.

    ``"i15~16"`` -> 15, ``"+0c"`` -> 0, ``"-13~-8"`` -> -13, ``"+32a (+24)"`` -> 32,
    blank -> ``None``. Never raises; an uparseable cell returns ``None``.
    """
    m = _LEAD_INT.search(cell)
    if m is None:
        return None
    try:
        return int(m.group())
    except ValueError:  # pragma: no cover - regex guarantees a valid int
        return None


def _split_cells(cell: str) -> list[str]:
    """Split a per-hit comma-list into stripped tokens, preserving blanks for index alignment."""
    return [part.strip() for part in cell.split(",")]


def _split_tags(cell: str) -> list[str]:
    """Split the Tags column into raw tokens (whitespace- and comma-delimited)."""
    return [tok for tok in re.split(r"[\s,]+", cell.strip()) if tok]


def _at(cells: Sequence[str], index: int) -> str:
    """Return ``cells[index]`` stripped, or ``""`` if the list is shorter (blank-cell tolerance)."""
    return cells[index].strip() if index < len(cells) else ""


def normalize_row(row: dict[str, str]) -> FrameDataMove | None:
    """Normalize one CSV row (as a header->value dict) into a :class:`FrameDataMove`.

    Returns ``None`` for a row with no ``Command`` (blank/separator rows). Never raises on a
    malformed cell — unparseable frame values degrade to ``None`` with the raw preserved.
    """
    command = row.get("Command", "").strip()
    if not command:
        return None

    level_tokens = _split_cells(row.get("Hit level", ""))
    nonblank_levels = [t for t in level_tokens if t]
    is_string = len(nonblank_levels) > 1

    startup_tokens = _split_cells(row.get("Start up frame", ""))
    damage_tokens = _split_cells(row.get("Damage", ""))

    block_raw = row.get("Block frame", "").strip() or None
    hit_raw = row.get("Hit frame", "").strip() or None
    ch_raw = row.get("Counter hit frame", "").strip() or None
    recovery_raw = row.get("Recovery", "").strip() or None
    notes = row.get("Notes", "").strip() or None
    name = row.get("Name", "").strip() or None
    wavu_id = row.get("Wavu id", "").strip() or None

    first_startup = startup_tokens[0] if startup_tokens else ""
    first_damage = damage_tokens[0] if damage_tokens else ""

    hits: list[Hit] = []
    if is_string:
        for i, raw_level in enumerate(level_tokens):
            su_tok = _at(startup_tokens, i)
            dmg_tok = _at(damage_tokens, i)
            hits.append(
                Hit(
                    hit_level=parse_hit_level(raw_level),
                    hit_level_raw=raw_level,
                    startup=parse_frames(su_tok) if su_tok else None,
                    startup_raw=su_tok or None,
                    damage=parse_frames(dmg_tok) if dmg_tok else None,
                )
            )

    return FrameDataMove(
        key=command,
        is_string=is_string,
        startup=parse_frames(first_startup) if first_startup else None,
        startup_raw=first_startup or None,
        on_block=parse_frames(block_raw) if block_raw else None,
        block_raw=block_raw,
        on_hit=hit_raw,
        on_ch=ch_raw,
        damage=(parse_frames(first_damage) if first_damage else None) if not is_string else None,
        hit_level=(
            parse_hit_level(nonblank_levels[0]) if (nonblank_levels and not is_string) else None
        ),
        hit_level_raw=(nonblank_levels[0] if nonblank_levels and not is_string else None),
        properties=_split_tags(row.get("Tags", "")),
        recovery=parse_frames(recovery_raw) if recovery_raw else None,
        recovery_raw=recovery_raw,
        name=name,
        wavu_id=wavu_id,
        notes=notes,
        hits=hits,
    )


def parse_csv(text: str) -> Iterator[dict[str, str]]:
    """Parse semicolon-delimited CSV text into header->value dicts, validating the header.

    Raises :class:`CsvFormatError` if a required column is missing (docs/05 §3.3: a format
    change upstream must fail loudly at ingest, not silently corrupt the snapshot).
    """
    # Use StringIO (not splitlines) so csv handles newlines embedded in quoted cells — the
    # Notes column is multi-line and splitlines would corrupt it and drop the newlines.
    reader = csv.reader(io.StringIO(text), delimiter=";")
    try:
        header = next(reader)
    except StopIteration:
        raise CsvFormatError("empty CSV: no header row") from None
    missing = [c for c in EXPECTED_COLUMNS if c not in header]
    if missing:
        raise CsvFormatError(f"CSV header missing required columns: {missing}")
    for values in reader:
        if not any(v.strip() for v in values):
            continue  # skip fully-blank separator rows
        yield dict(zip(header, values, strict=False))


def normalize_char_csvs(
    char_name: str,
    csv_texts: Iterable[str],
) -> CharFrameData:
    """Normalize one or more CSV files for a single character into a :class:`CharFrameData`.

    A character may have several CSV files under its directory (docs/05 §3.3); their rows are
    merged into one keyed-by-``framedata_key`` table. The character slug is taken from the CSV
    ``Character id`` column (falls back to a slugified ``char_name`` if the column is blank).
    """
    moves: dict[str, FrameDataMove] = {}
    char_slug = ""
    for text in csv_texts:
        for row in parse_csv(text):
            if not char_slug:
                char_slug = row.get("Character id", "").strip()
            move = normalize_row(row)
            if move is not None:
                moves[move.key] = move
    if not char_slug:
        char_slug = char_name.strip().lower().replace(" ", "-")
    return CharFrameData(char_slug=char_slug, char_name=char_name, moves=moves)
