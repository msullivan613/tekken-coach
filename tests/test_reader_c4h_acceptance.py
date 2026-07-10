"""C4h acceptance: derive (no seeded offsets) -> build -> persist -> select -> decode -> doctor.

The end-to-end claim the chunk is judged on. C4e proved a planted T8-shaped world is enough for
``update-offsets --base-scan`` to write a doctor-passing table *given the seeded within-struct
offsets and pointer chain*. C4h proves the harder thing: with those seeds gone — the entity struct
located on the enumerated heap purely by behavior, the field offsets and stride and Jin's id derived
as outputs, and the pointer path found by a reverse scan confirmed across a reallocation — the same
shipped decoder reads the table and the same shipped doctor passes.

The reverse scan roots the anchor at the pointer target (the address the game's pointer holds), so
the derived ``char_id`` offset is non-zero and every offset is relative to that base — which is why
this is a genuinely derived table, not the seeded one rediscovered. The one thing no scan can prove,
what an encoded value *means*, is supplied by a calibrated state map (the docs/02 §8 protocol); the
state-word *offsets* are seeded best-effort, translated onto the derived base and flagged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tekken_coach.reader.decode import decode_frame, poll_frames, resolve_anchor
from tekken_coach.reader.discovery.heapscan import DeriveInputs
from tekken_coach.reader.discovery.manifest import ProbeManifest, load_probe_manifest
from tekken_coach.reader.discovery.orchestrate import discover_derive, persist
from tekken_coach.reader.doctor import run_doctor
from tekken_coach.reader.offsets import (
    POSITION_COMPONENT,
    OffsetTable,
    load_offset_table,
    select_offset_table,
)
from tekken_coach.schemas import ActionState, MatchState
from tests.fixtures.reader.planted_chain import (
    JIN,
    KAZUYA,
    MODULE,
    MODULE_BASE,
    P1_POS_BEFORE,
    ROUND_START_HEALTH,
    component_frame,
    heap_decode_source,
    planted_heap_world,
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
    """Run the real C4h derivation against the planted heap world and load what it wrote."""
    caps = planted_heap_world()
    inputs = DeriveInputs(
        before=caps.before, during=caps.during, after=caps.after, realloc=caps.realloc
    )
    table, report = discover_derive(
        inputs,
        module=MODULE,
        module_base=MODULE_BASE,
        game_version=DETECTED_EXE_VERSION,
        manifest=_manifest(),
        seed=_seed(),
        seed_version="2.01.01",
        state_map=calibrated_state_map(),
        discovered_at="2026-07-10T00:00:00Z",
    )
    assert table is not None, "the confident core must resolve on the planted heap world"
    assert report.ok, f"unresolved: {report.result.unresolved}"
    persist(table, report, offsets_dir=tmp_path)
    return select_offset_table(DETECTED_EXE_VERSION, tmp_path)


def test_the_table_is_fully_derived_not_seeded(derived: OffsetTable) -> None:
    # A DERIVED pointer path (the seeded C4d chain ends in +0x30; the reverse scan roots at the
    # pointer target, so the last hop is +0x0 and char_id carries the residual offset).
    assert derived.players.anchor.pointer_path[-1] == 0x0
    assert derived.players.fields["char_id"].offset != 0  # relative to the pointer target
    # Jin's id is discovered (an output), not seeded — the manifest only carries Kazuya's 12.
    assert JIN not in {_manifest().kazuya_char_id}
    # damage_taken is derived from the landed jab; health is computed from it (T8 has no HP field).
    assert derived.players.max_health == ROUND_START_HEALTH
    assert "damage_taken" in derived.players.fields
    # position moved out to the transform component; the placeholder booleans are gone.
    assert POSITION_COMPONENT in derived.players.components
    assert "pos_x" not in derived.players.fields
    assert "block_stun" not in derived.players.fields


def test_the_written_table_decodes_a_framerecord(derived: OffsetTable) -> None:
    source = heap_decode_source(derived)
    decoded = decode_frame(source, derived)
    expected = component_frame()

    assert decoded.players[0].char_id == expected.players[0].char_id
    assert decoded.players[0].move_id == expected.players[0].move_id
    assert decoded.players[0].health == ROUND_START_HEALTH  # computed from damage_taken
    assert decoded.players[0].action_state is ActionState.attack  # via the encoded state words
    assert decoded.players[0].pos == pytest.approx(P1_POS_BEFORE)  # via the transform component
    # match_phase is still seeded garbage on the real build, so decode says 'unknown' and goes on.
    assert decoded.match_state is MatchState.unknown


def _ticking(derived: OffsetTable, count: int) -> TickingFlatSource:
    frames = [heap_decode_source(derived, step=i) for i in range(count)]
    g = derived.global_struct
    advance_on = resolve_anchor(frames[0], g.anchor) + g.fields["frame_counter"].offset
    return TickingFlatSource(frames, advance_on=advance_on)


def test_the_doctor_passes_on_the_written_table(derived: OffsetTable) -> None:
    report = run_doctor(_ticking(derived, 6), derived, known_char_ids=KNOWN_CHAR_IDS, frames=6)
    assert report.ok, [(c.name, c.detail) for c in report.failures()]
    assert {c.name for c in report.checks} == {
        "char_ids_known",
        "health_plausible",
        "frame_monotonic",
        "move_id_stable",
        "positions_change",
    }


def test_frames_poll_forward_with_no_gaps(derived: OffsetTable) -> None:
    reads = poll_frames(_ticking(derived, 4), derived, 4)
    assert [r.frame.frame for r in reads] == [component_frame().frame + i for i in range(4)]
    assert all(r.gap == 0 and r.gap_note is None for r in reads)
