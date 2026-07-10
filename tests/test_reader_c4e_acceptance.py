"""C4e acceptance: derive -> build -> persist -> select -> decode -> doctor, on the T8 shape.

The individual pieces are covered elsewhere. This is the end-to-end claim the chunk is judged on: a
planted world with **Tekken 8's real shape** — the player struct behind a static pointer + chain,
the global/match struct behind its own static pointer + chain, encoded state words instead of
booleans, and position in a transform component rather than the entity struct — is enough for
``update-offsets`` to write a table the *shipped decoder* reads and the *shipped doctor* passes.

Every offset the scan needs is planted at a value the probe manifest does not carry, so passing
proves the scan found them. The one thing no scan can prove — what an encoded value *means* — is
supplied by a calibrated state map, standing in for the docs/02 §8 observation protocol.

C4f took one prop away. The fixture no longer hands the decoder a valid ``match_phase``: the global
scan does not derive that offset, so on the real build it holds garbage, and the C4e fixture's tidy
phase code hid a ``decode_frame`` that raised on every live frame. Now the same world proves the
harder claim — the doctor validates the mechanical core *anyway*, and the capture gate still refuses
to record on a phase it cannot read.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tekken_coach.reader.decode import (
    decode_frame,
    poll_frames,
    read_state_signal,
    resolve_anchor,
)
from tekken_coach.reader.discovery.manifest import ProbeManifest, load_probe_manifest
from tekken_coach.reader.discovery.orchestrate import discover_base, persist
from tekken_coach.reader.doctor import run_doctor
from tekken_coach.reader.faults import DecodeError
from tekken_coach.reader.offsets import (
    POSITION_COMPONENT,
    OffsetTable,
    load_offset_table,
    select_offset_table,
)
from tekken_coach.reader.state import SignalKind, classify_state
from tekken_coach.schemas import ActionState, MatchState
from tests.fixtures.reader.planted_chain import (
    COMPONENT_SLOT_OFFSET,
    GARBAGE_MATCH_PHASE,
    GLOBAL_BASE_OFFSET,
    GLOBAL_POINTER_PATH,
    JIN,
    KAZUYA,
    MODULE,
    MODULE_BASE,
    P1_POS_BEFORE,
    ROUND_START_HEALTH,
    component_decode_source,
    component_frame,
    planted_component,
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
    """Run the real discovery against the planted T8-shaped world and load what it wrote."""
    chain = planted_component()
    table, report = discover_base(
        chain.before,
        module=MODULE,
        module_base=MODULE_BASE,
        game_version=DETECTED_EXE_VERSION,
        manifest=_manifest(),
        seed=_seed(),
        seed_version="2.01.01",
        source_after=chain.after,
        state_map=calibrated_state_map(),
        discovered_at="2026-07-09T00:00:00Z",
    )
    assert table is not None, "the confident core must resolve on the planted T8-shaped world"
    assert report.ok, f"unresolved: {report.result.unresolved}"
    persist(table, report, offsets_dir=tmp_path)
    return select_offset_table(DETECTED_EXE_VERSION, tmp_path)


def test_the_written_table_has_all_three_c4e_pieces(derived: OffsetTable) -> None:
    # Phase 1: the global anchor is its own static slot + chain, not the seed's static +0x0.
    assert derived.global_struct.anchor.base_offset == GLOBAL_BASE_OFFSET
    assert derived.global_struct.anchor.pointer_path == GLOBAL_POINTER_PATH
    # Phase 2: encoded state words replace the placeholder booleans.
    assert derived.state_codes.encoded_state is not None
    assert derived.players.fields["stun_type"].offset == 0x644
    assert "block_stun" not in derived.players.fields
    # Phase 3: position moved out of the entity struct into a component.
    assert derived.players.components[POSITION_COMPONENT].slot_offset == COMPONENT_SLOT_OFFSET
    assert "pos_x" not in derived.players.fields


def test_the_written_table_decodes_a_framerecord(derived: OffsetTable) -> None:
    source = component_decode_source(derived)
    decoded = decode_frame(source, derived)
    expected = component_frame()

    assert decoded.frame == expected.frame
    assert decoded.round == expected.round
    assert decoded.timer_ms == expected.timer_ms
    # The fixture plants garbage at the still-seeded match_phase offset, as the real build holds.
    # decode_frame *describes* the frame, so it says so and carries on; it used to raise here, which
    # is why not one live frame ever decoded.
    assert decoded.match_state is MatchState.unknown

    p1 = decoded.players[0]
    assert p1.char_id == expected.players[0].char_id
    assert p1.move_id == expected.players[0].move_id
    assert p1.move_frame == expected.players[0].move_frame
    assert p1.health == ROUND_START_HEALTH
    assert p1.action_state is ActionState.attack  # round-tripped through the encoded state words
    assert p1.pos == pytest.approx(P1_POS_BEFORE)  # read through the transform component
    # The raw words ride along so a mis-decode is diagnosable from the captured frame alone.
    assert p1.raw_state is not None and p1.raw_state["simple_move_state"] == 1


def _ticking(derived: OffsetTable, count: int) -> TickingFlatSource:
    frames = [component_decode_source(derived, step=i) for i in range(count)]
    g = derived.global_struct
    advance_on = resolve_anchor(frames[0], g.anchor) + g.fields["frame_counter"].offset
    return TickingFlatSource(frames, advance_on=advance_on)


def test_the_doctor_passes_on_the_written_table(derived: OffsetTable) -> None:
    # The gate the chunk exists to turn green. It needs a *live* process: the frame counter must
    # advance and the players must move, so the source replays successive planted snapshots. It runs
    # to completion on a garbage match_phase — none of the five checks reads it, so an uncalibrated
    # phase must not hide the anchors, stride and field offsets the scan *did* prove.
    report = run_doctor(_ticking(derived, 6), derived, known_char_ids=KNOWN_CHAR_IDS, frames=6)
    assert report.ok, [(c.name, c.detail) for c in report.failures()]
    assert {c.name for c in report.checks} == {
        "char_ids_known",
        "health_plausible",
        "frame_monotonic",
        "move_id_stable",
        "positions_change",
    }
    # ... and it says what it could not check, rather than letting green read as "fully calibrated".
    assert any("match_phase" in note and "REFUSE" in note for note in report.notes)


def test_the_capture_gate_still_refuses_the_unknown_phase(derived: OffsetTable) -> None:
    # The boundary C4f draws. A diagnostic that tolerates a phase it cannot read must not loosen the
    # gate deciding whether to record: `read_state_signal` is what clean mode's online refusal
    # (docs/01 §4.3) consults, and it fails closed on the very frame `decode_frame` just accepted.
    source = component_decode_source(derived)
    assert decode_frame(source, derived).match_state is MatchState.unknown
    with pytest.raises(DecodeError, match=f"unknown match_phase code {GARBAGE_MATCH_PHASE}"):
        read_state_signal(source, derived)


def test_an_unknown_phase_is_never_an_active_match(derived: OffsetTable) -> None:
    # Even where the unknown phase does flow onwards, it is inert: it is not an active phase, so it
    # cannot read as a live match, and clean mode will not buffer on it.
    signal = classify_state(MatchState.unknown, "practice")
    assert signal.kind is SignalKind.idle
    assert not signal.should_buffer_clean


def test_frames_poll_forward_with_no_gaps(derived: OffsetTable) -> None:
    reads = poll_frames(_ticking(derived, 4), derived, 4)
    assert [r.frame.frame for r in reads] == [component_frame().frame + i for i in range(4)]
    assert all(r.gap == 0 and r.gap_note is None for r in reads)
