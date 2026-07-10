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

# How each confidence tier reads in the report's field table. ``??`` marks an offset the scan wrote
# from layout *facts* without proving it this run (the encoded state words) — the calibration focus.
_CONFIDENCE_TAG: dict[Confidence, str] = {
    Confidence.high: "ok",
    Confidence.medium: "!!",
    Confidence.seeded: "??",
}

# The Windows calibration runbook, carried on every report so the paste-back loop is self-contained.
CALIBRATION_RUNBOOK = """\
Windows calibration runbook (docs/02 §4) — run in native Python, not WSL:

  1. Launch Tekken 8. Open PRACTICE mode with P1 = Jin, P2 = Kazuya. Let the round START (both
     health bars full, neither player has taken damage) before capturing — round-start full health
     and damage_taken == 0 are the anchors both techniques rely on.
  2. Run ONE of:

     a) python -m tekken_coach.reader.commands update-offsets --base-scan     [C4d, PREFERRED]
        Code-signature derivation. Parses the module's PE header, sweeps its .data sections for the
        static pointer that leads to the player struct, follows the pointer chain, and accepts the
        candidate only when BOTH players resolve (P1 char id plausible, P2 = Kazuya's id 12,
        move ids plausible, damage_taken == 0). This is the technique to use on Tekken 8: the
        entity struct is HEAP-ALLOCATED and reallocates on every character change / round, so a
        module-relative window or a raw heap address goes stale immediately. The derived anchor is
        module_base + base_offset + pointer_path, which the reader re-resolves every frame.

     b) python -m tekken_coach.reader.commands update-offsets --derive         [C4h, FULLY DERIVED]
        Seeds NO within-struct offset and NO pointer chain — use this when --base-scan finds nothing
        because its seeded offsets have gone stale (a new season/patch: the fork they came from is
        long dead). It locates the entity struct on the ENUMERATED HEAP by behavior (Kazuya's id 12
        beside a plausible id at a similar-struct stride whose acting player's move_id changes
        across the action window), derives every field offset + stride + Jin's id as OUTPUTS, then
        reverse-scans the static data for a pointer path and keeps only one that SURVIVES A
        REALLOCATION. It prompts you to act AND to reset the round once (so the struct moves and the
        durable path can be confirmed). A patch becomes a re-run, not a re-seed.

     c) python -m tekken_coach.reader.commands update-offsets                 [C4c, heap value-scan]
        Only useful if the struct happens to sit at a fixed module-relative offset.

     Either way the tool prompts you to ACT between snapshots (walk P1 forward, jab P2, jump); the
     position/move changes are what locate pos_{x,y,z}, move_id, and the frame counter. It writes
     assets/offsets/<detected-version>.json and registers the version.
  3. Invalidate move/frame data for the new version if the balance patch also changed (see docs/05).
  4. Validate:  python -m tekken_coach.reader.commands doctor
     Green -> the derived core (char ids, health, frame counter, move id, positions) is correct and
     capture is usable. Then run `capture` to produce the first real FrameRecord fixtures.

If the BASE SCAN (--base-scan) found nothing:
  - Confirm the setup really is P1 Jin vs P2 Kazuya at round start with 0 damage taken.
  - The chain shape may have moved. Edit base_scan.pointer_path in probe-manifest.json (it is DATA).
  - Check base_scan.round_start_health matches this build's full HP, and char_id/move_id/
    damage_taken offsets still describe the struct (these are the oracle; if they are wrong,
    nothing can validate).
  - If the seeded offsets themselves are stale (a season/patch), stop editing them and run
    --derive instead: it seeds none of them and derives the whole layout from behavior.

If the DERIVE scan (--derive) found nothing:
  - "no heap struct BEHAVED like the acting player": you must ACT the WHOLE window — walk P1 (Jin)
    forward, jab P2, jump, on repeat. move_id is transient; the scan accepts a change in ANY sample,
    but there has to be one.
  - "found NO static pointer path that survives a reallocation": you did not reset the round when
    prompted (the struct must MOVE so a durable path can be told apart from a coincidental one).
    Reset the round / swap a character, return to the SAME Jin-vs-Kazuya setup, then confirm. Or
    widen derive_scan.reverse_max_depth / reverse_max_offset if the real chain is deeper.
  - "no structural char-id pair found": confirm P2 is Kazuya (id 12). Lower similarity_min /
    min_shared_words only if the two structs genuinely share little at round start.

If it reports the TWO-LEVEL P2 case (P1 located, no constant stride to Kazuya):
  - Raise base_scan.max_stride if P2 is merely farther away than the ceiling.
  - Otherwise P2 is a separate allocation and the single-anchor + stride PlayerStruct cannot express
    it. That needs a per-player-anchor SCHEMA change — stop and report; do not hand-edit a stride.

If the GLOBAL anchor did not resolve (frame_counter unresolved; `doctor` fails frame_monotonic):
  - The global struct is behind its own static pointer + chain. Widen global_scan.pointer_paths in
    probe-manifest.json (a list of candidate chain shapes; the first that validates wins).
  - global_scan.field_offsets is an UNASSIGNED list of within-struct offsets: the tool decides which
    is frame_counter / round / timer_ms / match_phase by BEHAVIOR (one ticks up, one holds a round
    number, one counts down). Add offsets there rather than assigning them by hand.
  - Practice mode often freezes the round clock, so timer_ms is frequently unassignable. That is
    expected and does not block the anchor: frame_counter + round are what the oracle requires.

If POSITION is unresolved (`doctor` fails positions_change):
  - You must actually WALK P1 between the two snapshots — a step, not a twitch.
  - Position is not in the entity struct on Tekken 8; it lives in a transform component the entity
    points at. Widen base_scan.component_scan (slot_span / probe_span / inner_span / max_depth).

If a DERIVED field is wrong or an anchor is UNRESOLVED (C4c path):
  - Widen or relocate the scan window in assets/offsets/probe-manifest.json (player_window /
    global_window). Player structs behind a pointer chain will not fall in a module-relative
    window — that is exactly what --base-scan exists to solve.
  - Adjust plausibility bounds (stride_min/max, char_id_max, move_id_max) if the scan locked onto a
    coincidental match.

The ENCODED STATE MAP is a separate, one-time calibration (docs/02 §8). The scan proves WHERE the
state words are; only observation proves what their VALUES mean. Until you run that protocol
(`update-offsets` prints "encoded state map: NOT CALIBRATED"), every action_state decodes to
`neutral` and the stun/throw/juggle flags are always false — structurally valid, semantically empty.
Run: python -m tekken_coach.reader.commands probe-state
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
            chain = r.player_anchor.pointer_path
            chain_txt = (
                " -> " + " -> ".join(f"+0x{o:x}" for o in chain) if chain else " (static, no chain)"
            )
            lines.append(
                f"  player anchor         : {r.player_anchor.module}"
                f"+0x{r.player_anchor.base_offset:x}{chain_txt}"
            )
            if r.player_anchor.signature is not None:
                sig = r.player_anchor.signature
                lines.append(
                    f"  base AOB signature    : slot_delta 0x{sig.slot_delta:x}  {sig.pattern}"
                )
        if r.global_anchor is not None:
            gchain = r.global_anchor.pointer_path
            gtxt = (
                " -> " + " -> ".join(f"+0x{o:x}" for o in gchain)
                if gchain
                else " (static, no chain)"
            )
            lines.append(
                f"  global anchor         : {r.global_anchor.module}"
                f"+0x{r.global_anchor.base_offset:x}{gtxt}"
            )

        for name, comp in sorted(r.components.items()):
            hops = " -> ".join(f"+0x{o:x}" for o in comp.pointer_path) or "(direct)"
            label = f"{name} component"
            lines.append(
                f"  {label:<21} : deref(player+0x{comp.slot_offset:x}) {hops}"
                f"  fields {sorted(comp.fields)}"
            )

        lines.append("")
        lines.append("  derived this run (doctor validates these):")
        for df in sorted(r.fields, key=lambda f: (f.scope, f.offset)):
            tag = _CONFIDENCE_TAG[df.confidence]
            lines.append(
                f"    [{tag}] {df.scope:<6} {df.name:<19} +0x{df.offset:<5x} {df.kind:<6}"
                f" @0x{df.example_address:x}  ({df.confidence.value}: {df.method})"
            )

        if r.encoded_state is not None:
            status = "CALIBRATED" if r.encoded_state.calibrated else "NOT CALIBRATED"
            lines.append("")
            lines.append(f"  encoded state map    : {status} ({len(r.encoded_state.flags)} fields)")
            if not r.encoded_state.calibrated:
                lines.append(
                    "    -> every state decodes to `neutral` until you run the docs/02 §8 "
                    "observation protocol. The reader will run; its stun/hit flags will be empty."
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
    """Build a :class:`DiagnosticReport`, computing which table fields were seeded vs derived.

    A field the derivation *dropped* (the placeholder booleans an encoded state map supersedes) is
    neither derived nor seeded — it is gone from the written table, so listing it as "verify this"
    would send calibration after an offset that no longer exists.
    """
    derived_player = set(result.player_offsets()) | set(result.drop_player_fields)
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
