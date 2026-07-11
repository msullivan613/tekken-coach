"""C4i acceptance: adopt the holder model end-to-end — discover -> build -> decode -> doctor.

The claim the chunk is judged on: a planted world with **Tekken 8's real player-addressing model** —
a holder object found by an AoB *code* signature (RIP-relative to a ``.data`` slot), two per-player
pointer slots to **separate** allocations (not a stride), encoded state words, and position in a
transform component — is enough for ``update-offsets --holder-scan`` to write a table the *shipped
decoder* reads and the *shipped doctor* passes.

Nothing tells the scan where the holder is: it AoB-matches the storing instruction and RIP-decodes
the displacement, so passing proves the code-signature path works. Jin's id (6) and Kazuya's (12)
are the round-start oracle, and the acting player's ``move_id`` changing across the window is the
behavioral confirmation — the same C4f/C4g argument, one addressing model over.

As in C4e, the fixture hands the decoder **garbage** at the still-seeded ``match_phase`` offset, so
``decode_frame`` reports ``MatchState.unknown`` and carries on (the doctor validates the mechanical
core anyway) while the capture gate still refuses to record.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tekken_coach.reader.decode import decode_frame, poll_frames, resolve_anchor
from tekken_coach.reader.discovery.manifest import ProbeManifest, load_probe_manifest
from tekken_coach.reader.discovery.orchestrate import discover_holder, persist
from tekken_coach.reader.doctor import run_doctor
from tekken_coach.reader.offsets import (
    POSITION_COMPONENT,
    OffsetTable,
    load_offset_table,
    select_offset_table,
)
from tekken_coach.schemas import ActionState, MatchState
from tests.fixtures.reader.planted_chain import MODULE, MODULE_BASE, ROUND_START_HEALTH
from tests.fixtures.reader.planted_holder import (
    AOB_PATTERN,
    DISP32_POS,
    HOLDER_SLOT_RVA,
    HOLDER_SLOTS,
    JIN,
    KAZUYA,
    P1_BASE,
    P1_POS_BEFORE,
    P2_BASE,
    component_frame,
    holder_decode_source,
    planted_holder,
    resolved_player_bases,
)
from tests.fixtures.reader.state_map import calibrated_state_map
from tests.fixtures.reader.ticking import TickingFlatSource

DETECTED_EXE_VERSION = "5.02.01"
KNOWN_CHAR_IDS = {JIN, KAZUYA, 12}


def _manifest() -> ProbeManifest:
    return load_probe_manifest(Path("assets/offsets/probe-manifest.json"))


def _seed() -> OffsetTable:
    return load_offset_table(Path("assets/offsets/2.01.01.json"))


@pytest.fixture
def derived(tmp_path: Path) -> OffsetTable:
    """Run the real holder discovery against the planted holder world and load what it wrote."""
    world = planted_holder()
    table, report = discover_holder(
        world.before,
        module=MODULE,
        module_base=MODULE_BASE,
        game_version=DETECTED_EXE_VERSION,
        manifest=_manifest(),
        seed=_seed(),
        seed_version="2.01.01",
        source_after=world.after,
        during=world.during,
        state_map=calibrated_state_map(),
        discovered_at="2026-07-10T00:00:00Z",
    )
    assert table is not None, "the confident core must resolve on the planted holder world"
    assert report.ok, f"unresolved: {report.result.unresolved}"
    persist(table, report, offsets_dir=tmp_path)
    return select_offset_table(DETECTED_EXE_VERSION, tmp_path)


def test_the_written_table_uses_the_holder_model(derived: OffsetTable) -> None:
    # The anchor is the RIP-decoded .data slot, dereferenced once to the holder object.
    assert derived.players.anchor.base_offset == HOLDER_SLOT_RVA
    assert derived.players.anchor.pointer_path == [0]
    # It carries a RIP-relative AoB code signature (the durable, self-healing re-find artifact).
    sig = derived.players.anchor.signature
    assert sig is not None and sig.pattern == AOB_PATTERN and sig.disp32_pos == DISP32_POS
    # Two per-player pointer slots, NO stride — the regression the schema change exists to allow.
    assert derived.players.stride is None
    assert [s.slot_offset for s in derived.players.player_slots] == HOLDER_SLOTS
    # Health is computed (HP encrypted): max_health is set and damage_taken is the read field, so
    # the decoder ignores any carried `health` field. Position is in a component; state words
    # replace the placeholder booleans.
    assert derived.players.max_health == ROUND_START_HEALTH
    assert "damage_taken" in derived.players.fields
    assert derived.players.components[POSITION_COMPONENT] is not None
    assert derived.state_codes.encoded_state is not None
    assert "block_stun" not in derived.players.fields


def test_the_two_players_resolve_to_separate_allocations(derived: OffsetTable) -> None:
    # The whole point of the holder model: P1 and P2 are separate allocations reached through their
    # own slots, not P1 + a constant stride.
    p1, p2 = resolved_player_bases(derived)
    assert (p1, p2) == (P1_BASE, P2_BASE)


def test_the_written_table_decodes_a_framerecord(derived: OffsetTable) -> None:
    source = holder_decode_source(derived)
    decoded = decode_frame(source, derived)
    expected = component_frame()

    assert decoded.frame == expected.frame
    assert decoded.round == expected.round
    # Garbage at the still-seeded match_phase: decode describes and carries on (it used to raise).
    assert decoded.match_state is MatchState.unknown

    p1 = decoded.players[0]
    assert p1.char_id == expected.players[0].char_id
    assert p1.move_id == expected.players[0].move_id
    assert p1.health == ROUND_START_HEALTH  # computed from damage_taken
    assert p1.action_state is ActionState.attack  # through the encoded state words
    assert p1.pos == pytest.approx(P1_POS_BEFORE)  # through the transform component


def _ticking(derived: OffsetTable, count: int) -> TickingFlatSource:
    frames = [holder_decode_source(derived, step=i) for i in range(count)]
    g = derived.global_struct
    advance_on = resolve_anchor(frames[0], g.anchor) + g.fields["frame_counter"].offset
    return TickingFlatSource(frames, advance_on=advance_on)


def test_the_doctor_passes_on_the_holder_table(derived: OffsetTable) -> None:
    report = run_doctor(_ticking(derived, 6), derived, known_char_ids=KNOWN_CHAR_IDS, frames=6)
    assert report.ok, [(c.name, c.detail) for c in report.failures()]
    assert {c.name for c in report.checks} == {
        "char_ids_known",
        "health_plausible",
        "frame_monotonic",
        "move_id_stable",
        "positions_change",
    }
    assert any("match_phase" in note and "REFUSE" in note for note in report.notes)


def test_frames_poll_forward_with_no_gaps(derived: OffsetTable) -> None:
    reads = poll_frames(_ticking(derived, 4), derived, 4)
    assert [r.frame.frame for r in reads] == [component_frame().frame + i for i in range(4)]
    assert all(r.gap == 0 and r.gap_note is None for r in reads)
