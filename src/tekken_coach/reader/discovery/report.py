"""The structured diagnostic report ``update-offsets`` emits (docs/02 §4).

Calibration is a paste-back loop: the tool prints *what it found, what it derived, and what it
could not resolve*, and the user pastes that back so we can adjust the probe manifest. This module
builds that report from a :class:`~.derive.DerivationResult` plus the seed/table context and renders
it to text. It is **data** (a dataclass) with a renderer, honoring the silent-producer boundary
(docs/02 §2) — the command layer prints it; nothing here prints on its own.

The report separates three tiers so calibration is focused:

* **derived (high/medium)** — located this run; the doctor (docs/02 §6) validates exactly these.
* **seeded** — carried from the previous table; verify after the doctor passes on the derived core.
* **unresolved** — a derivable anchor the scan missed; widen the manifest window and re-run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tekken_coach.reader.discovery.derive import Confidence, DerivationResult
from tekken_coach.reader.offsets import OffsetTable

# The Windows calibration runbook, carried on every report so the paste-back loop is self-contained.
CALIBRATION_RUNBOOK = """\
Windows calibration runbook (docs/02 §4) — run in native Python, not WSL:

  1. Launch Tekken 8. Open PRACTICE mode with P1 = Jin, P2 = Kazuya. Let the round START (both
     health bars full) before capturing — the round-start full-health value is the stride anchor.
  2. Run:  python -m tekken_coach.reader.commands update-offsets
     The tool takes a first snapshot at round start, prompts you to ACT (walk P1 forward a step and
     press a button), then takes a second snapshot. The move/position/frame-counter CHANGES between
     the two snapshots are what locate move_id, position, and the frame counter.
     It writes assets/offsets/<detected-version>.json and registers the version in index.json.
  3. Invalidate move/frame data for the new version if the balance patch also changed (see docs/05).
  4. Validate:  python -m tekken_coach.reader.commands doctor
     Green -> the derived core (char ids, health, frame counter, move id, positions) is correct and
     capture is usable. Then run `capture` to produce the first real FrameRecord fixtures.

If a DERIVED field is wrong or an anchor is UNRESOLVED:
  - Widen or relocate the scan window in assets/offsets/probe-manifest.json (player_window /
    global_window). Player structs behind a pointer chain will not fall in a module-relative
    window — set that window `absolute` to an address you located, or calibrate the anchor manually.
  - Adjust plausibility bounds (stride_min/max, char_id_max, move_id_max) if the scan locked onto a
    coincidental match.
Paste the report below back so we can adjust the probe manifest together.
"""


@dataclass
class DiagnosticReport:
    """The full re-discovery diagnostic (docs/02 §4), rendered by :meth:`render`."""

    game_version: str
    module: str
    module_base: int
    result: DerivationResult
    seed_version: str
    seeded_player_fields: list[str] = field(default_factory=list)
    seeded_global_fields: list[str] = field(default_factory=list)
    table_written: str | None = None
    index_written: str | None = None
    runbook: str = CALIBRATION_RUNBOOK

    @property
    def ok(self) -> bool:
        return self.result.ok

    def render(self) -> str:
        r = self.result
        lines: list[str] = []
        status = "OK (confident core resolved)" if self.ok else "INCOMPLETE — see unresolved below"
        lines.append(f"update-offsets diagnostic — {status}")
        lines.append(f"  detected game version : {self.game_version}")
        lines.append(f"  module                : {self.module} @ 0x{self.module_base:x}")
        if r.stride is not None:
            lines.append(f"  player struct stride  : {r.stride} (0x{r.stride:x}) bytes")
        if r.player_char_ids is not None:
            jin, kaz = r.player_char_ids
            lines.append(f"  char ids              : P1(Jin)={jin}  P2(Kazuya)={kaz}")
        if r.player_anchor is not None:
            lines.append(
                f"  player anchor         : {r.player_anchor.module}"
                f"+0x{r.player_anchor.base_offset:x}"
            )
        if r.global_anchor is not None:
            lines.append(
                f"  global anchor         : {r.global_anchor.module}"
                f"+0x{r.global_anchor.base_offset:x}"
            )

        lines.append("")
        lines.append("  derived this run (doctor validates these):")
        for df in sorted(r.fields, key=lambda f: (f.scope, f.offset)):
            tag = "!!" if df.confidence is Confidence.medium else "ok"
            lines.append(
                f"    [{tag}] {df.scope:<6} {df.name:<14} +0x{df.offset:<4x} {df.kind:<6}"
                f" @0x{df.example_address:x}  ({df.confidence.value}: {df.method})"
            )

        if r.unresolved:
            lines.append("")
            lines.append("  UNRESOLVED anchors (widen the manifest window and re-run):")
            for name in r.unresolved:
                lines.append(f"    [XX] {name}")

        seeded = self.seeded_player_fields + self.seeded_global_fields
        if seeded:
            lines.append("")
            lines.append(
                f"  seeded from table {self.seed_version} (verify after the doctor passes): "
                f"{', '.join(sorted(seeded))}"
            )

        if r.notes:
            lines.append("")
            lines.append("  notes:")
            for note in r.notes:
                lines.append(f"    - {note}")

        if self.table_written:
            lines.append("")
            lines.append(f"  wrote offset table : {self.table_written}")
        if self.index_written:
            lines.append(f"  updated index      : {self.index_written}")

        lines.append("")
        lines.append(self.runbook)
        return "\n".join(lines)


def build_report(
    result: DerivationResult,
    *,
    game_version: str,
    module_base: int,
    seed: OffsetTable,
    seed_version: str,
) -> DiagnosticReport:
    """Build a :class:`DiagnosticReport`, computing which table fields were seeded vs derived."""
    derived_player = set(result.player_offsets())
    derived_global = set(result.global_offsets())
    seeded_player = [f for f in seed.players.fields if f not in derived_player]
    seeded_global = [f for f in seed.global_struct.fields if f not in derived_global]
    return DiagnosticReport(
        game_version=game_version,
        module=result.module,
        module_base=module_base,
        result=result,
        seed_version=seed_version,
        seeded_player_fields=seeded_player,
        seeded_global_fields=seeded_global,
    )
