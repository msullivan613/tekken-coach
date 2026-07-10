"""C4d/C4e/C4f code-signature derivation: candidate scan + the two oracles (docs/02 §3/§4).

The planted world (``tests/fixtures/reader/planted_chain.py``) hides the player struct behind a
static pointer in the module's ``.data`` plus a 4-level pointer chain — the shape Tekken 8's
reallocating heap struct forces — and hides the global/match struct behind a second static pointer
and its own chain. Nothing tells the scan where either slot is: ``BASE_OFFSET``,
``GLOBAL_BASE_OFFSET``, the ``STRIDE``, and the ``health``/``pos`` offsets are planted at values the
manifest does not carry. So a passing derivation proves the scan *found* them, and the final
acceptance test proves the resulting table drives the real C4a decoder through the chains.

The global struct's field offsets *are* seeded, but deliberately **scrambled** relative to the field
list, so only the behavioral oracle (ticks up / holds / counts down) can name them.

C4f plants the one thing the offline world was missing and the live game had: a **decoy** struct
that satisfies the player oracle at a single instant and never moves. The single-instant oracle
takes it — asserted here, so the regression stays visible — and the behavioral one rejects it.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from pydantic import ValidationError

from tekken_coach.reader.decode import decode_frame, resolve_anchor
from tekken_coach.reader.discovery.basescan import (
    _player_oracle_ok,
    _section_passes,
    assign_global_fields,
    derive_base_layout,
    extract_signature,
    find_by_signature,
    find_candidate_slots,
    frame_delta_band,
    locate_global_struct,
    locate_player_struct,
)
from tekken_coach.reader.discovery.builder import build_offset_table
from tekken_coach.reader.discovery.derive import Confidence, DerivationResult
from tekken_coach.reader.discovery.manifest import (
    GlobalScanSpec,
    ProbeManifest,
    load_probe_manifest,
)
from tekken_coach.reader.discovery.orchestrate import discover_base, persist
from tekken_coach.reader.discovery.pe import parse_module_image
from tekken_coach.reader.discovery.report import build_report
from tekken_coach.reader.faults import OffsetTableError
from tekken_coach.reader.offsets import (
    POSITION_COMPONENT,
    AobSignature,
    EncodedStateSpec,
    OffsetTable,
    load_offset_table,
    load_state_map,
    select_offset_table,
)
from tests.fixtures.reader.flat_source import FlatMemorySource
from tests.fixtures.reader.planted_chain import (
    BASE_OFFSET,
    COMPONENT_SLOT_OFFSET,
    COMPONENT_TRIPLE_OFFSET,
    DECOY_BASE_OFFSET,
    DECOY_P1_BASE,
    FRAME_STEP,
    GLOBAL_ALT_BASE_OFFSET,
    GLOBAL_BASE,
    GLOBAL_BASE_OFFSET,
    GLOBAL_FRAME_OFFSET,
    GLOBAL_POINTER_PATH,
    GLOBAL_ROUND_OFFSET,
    GLOBAL_TIMER_OFFSET,
    HEALTH_OFFSET,
    JIN,
    KAZUYA,
    MODULE,
    MODULE_BASE,
    MOVE_ID_OFFSET,
    P1_BASE,
    POINTER_PATH,
    POS_OFFSET,
    STRIDE,
    decode_source,
    expected_frame,
    global_duplicate_slot_source,
    global_two_structs_source,
    planted_chain,
    planted_component,
    planted_decoy,
    planted_decoy_nobody_acted,
    planted_transient_action,
    planted_whiffed_jab,
    relocated_pointer_source,
    two_level_source,
    zeroed_landing_source,
)
from tests.fixtures.reader.state_map import calibrated_state_map

SEED_PATH = Path("assets/offsets/2.01.01.json")
MANIFEST_PATH = Path("assets/offsets/probe-manifest.json")
DETECTED_EXE_VERSION = "5.02.01"  # the finding from the C4b smoke test (distinct from 2.01.01)


def _seed() -> OffsetTable:
    return load_offset_table(SEED_PATH)


def _manifest() -> ProbeManifest:
    return load_probe_manifest(MANIFEST_PATH)


def _derive() -> DerivationResult:
    chain = planted_chain()
    return derive_base_layout(
        chain.before,
        module=MODULE,
        module_base=MODULE_BASE,
        manifest=_manifest(),
        seed=_seed(),
        source_after=chain.after,
    )


# ---------------------------------------------------------------------------
# Candidate generation is bounded by the PE parse
# ---------------------------------------------------------------------------


def test_candidate_slots_are_bounded_to_the_data_section() -> None:
    source = planted_chain().before
    spec = _manifest().base_scan
    assert spec is not None
    image = parse_module_image(lambda rva, n: source.read(MODULE_BASE + rva, n))
    slots = find_candidate_slots(source, MODULE_BASE, image.data_sections())
    assert slots, "the planted root pointer must be generated as a candidate"
    assert BASE_OFFSET in slots
    # Every candidate lies inside .data — the sweep never touches .text or the headers.
    data = image.data_sections()[0]
    assert all(data.rva <= rva < data.end_rva for rva in slots)


def test_writable_data_is_swept_before_readonly() -> None:
    # The root pointer is a runtime-written global (writable .data). Sweeping .data first is both
    # likelier to hit and far cheaper than the big read-only .rdata; the .rdata pass is a fallback.
    source = planted_chain().before
    image = parse_module_image(lambda rva, n: source.read(MODULE_BASE + rva, n))
    spec = _manifest().base_scan
    assert spec is not None
    passes = _section_passes(image, spec)
    assert [label for label, _ in passes][0] == "writable .data"
    assert any(s.name == ".data" for _, sections in passes[:1] for s in sections)


def test_locate_reports_progress() -> None:
    # The long live sweep must be observable: the command layer passes a progress sink and the
    # library streams what it is doing (PE parse -> which pass it is sweeping -> how many landings
    # the behavioral oracle had to choose between, and on what evidence it chose).
    chain = planted_chain()
    msgs: list[str] = []
    located = locate_player_struct(
        chain.before,
        module=MODULE,
        module_base=MODULE_BASE,
        manifest=_manifest(),
        during=[chain.after],
        progress=msgs.append,
    )
    assert located is not None and located.match.strong
    joined = "\n".join(msgs)
    assert "parsed PE" in joined
    assert "writable .data" in joined
    assert "player anchor" in joined
    assert "acting move_id changed" in joined


def test_derive_reuses_a_prelocated_struct() -> None:
    # The live path locates the struct once (to freeze the round-start snapshot) and passes that
    # into derive so the expensive sweep is not repeated. Passing `located` must yield the same
    # anchor as letting derive locate itself.
    chain = planted_chain()
    prelocated = locate_player_struct(
        chain.before, module=MODULE, module_base=MODULE_BASE, manifest=_manifest()
    )
    assert prelocated is not None
    reused = derive_base_layout(
        chain.before,
        module=MODULE,
        module_base=MODULE_BASE,
        manifest=_manifest(),
        seed=_seed(),
        source_after=chain.after,
        located=prelocated,
    )
    fresh = _derive()
    assert reused.player_anchor is not None
    assert reused.player_anchor == fresh.player_anchor
    assert reused.stride == fresh.stride


def test_health_falls_back_to_computed_when_no_direct_hp_field() -> None:
    # The real Tekken 8 case: no offset reads full HP (the struct tracks damage_taken, not HP). The
    # derivation then emits damage_taken + sets max_health so health is COMPUTED, not left
    # unresolved. Forced here by pointing round_start_health at a value the planted struct lacks.
    chain = planted_chain()
    manifest = _manifest()
    assert manifest.base_scan is not None
    manifest.base_scan.round_start_health = 12345  # not present in the planted struct
    result = derive_base_layout(
        chain.before,
        module=MODULE,
        module_base=MODULE_BASE,
        manifest=manifest,
        seed=_seed(),
        source_after=chain.after,
    )
    assert result.max_health == 12345
    player_fields = result.player_offsets()
    assert "damage_taken" in player_fields  # emitted so the decoder can compute health
    assert "health" not in player_fields  # not a direct field
    assert "health" not in result.unresolved  # computed, so not reported as a gap


# ---------------------------------------------------------------------------
# The oracle recovers the planted static pointer -> chain -> struct layout
# ---------------------------------------------------------------------------


def test_locates_the_planted_base_offset_and_stride() -> None:
    chain = planted_chain()
    located = locate_player_struct(
        chain.before,
        module=MODULE,
        module_base=MODULE_BASE,
        manifest=_manifest(),
        during=[chain.after],
    )
    assert located is not None
    assert located.match.base_offset == BASE_OFFSET
    assert located.match.p1_base == P1_BASE
    assert located.match.stride == STRIDE
    assert located.match.strong


def test_discovers_jin_as_the_kazuya_counterpart() -> None:
    result = _derive()
    assert result.player_char_ids == (JIN, KAZUYA)


def test_derived_anchor_is_module_relative_with_the_pointer_chain() -> None:
    result = _derive()
    anchor = result.player_anchor
    assert anchor is not None
    assert anchor.module == MODULE
    assert anchor.base_offset == BASE_OFFSET
    assert anchor.pointer_path == POINTER_PATH
    # The decoder's own resolver walks it back to the struct — no decode change needed.
    assert resolve_anchor(planted_chain().before, anchor) == P1_BASE


def test_in_struct_scans_recover_health_and_position_offsets() -> None:
    # These are NOT seeded: value_scan for round-start max and a moving float triple find them,
    # tractably, inside the located struct.
    result = _derive()
    fields = result.player_offsets()
    assert fields["health"].offset == HEALTH_OFFSET
    assert fields["pos_x"].offset == POS_OFFSET
    assert fields["pos_y"].offset == POS_OFFSET + 4
    assert fields["pos_z"].offset == POS_OFFSET + 8
    assert result.unresolved == []


def test_oracle_field_offsets_come_from_the_seed_layout() -> None:
    result = _derive()
    fields = result.player_offsets()
    assert fields["char_id"].offset == 0x168
    assert fields["move_id"].offset == 0x528


def test_no_match_when_the_chain_shape_is_wrong() -> None:
    # A wrong pointer_path dereferences into unmapped memory: every candidate prunes, nothing is
    # invented, and the derivation reports the anchors as unresolved (no false positive).
    manifest = _manifest()
    assert manifest.base_scan is not None
    manifest.base_scan.pointer_path = [0x18, 0x18, 0x18]
    result = derive_base_layout(
        planted_chain().before,
        module=MODULE,
        module_base=MODULE_BASE,
        manifest=manifest,
        seed=_seed(),
    )
    assert result.player_anchor is None
    assert not result.ok
    assert set(result.unresolved) >= {"char_id", "move_id", "health", "pos_x"}


# ---------------------------------------------------------------------------
# C4f: the oracle is behavioral. A struct that looks right and never moves is not the player's.
# ---------------------------------------------------------------------------
#
# The planted decoy is the live 5.02.01 failure, offline: a struct whose char_id/move_id/damage read
# perfectly at round start, whose Kazuya-shaped neighbour sits at a constant stride, and whose
# fields never change — reached from a static slot that is swept *before* the real one.


def test_a_single_instant_oracle_locks_onto_the_frozen_decoy() -> None:
    # The C4e oracle, reproduced: with only one snapshot, "struct-shaped at round start" is all
    # there is to go on, and the decoy satisfies it first. This test exists to show the bug is real
    # and the fixture models it — not to bless the behavior.
    decoy = planted_decoy()
    located = locate_player_struct(
        decoy.before, module=MODULE, module_base=MODULE_BASE, manifest=_manifest()
    )
    assert located is not None
    assert located.match.p1_base == DECOY_P1_BASE
    assert located.match.base_offset == DECOY_BASE_OFFSET
    assert located.behavior is None  # nothing confirmed it; nothing could


def test_the_behavioral_oracle_rejects_the_decoy_and_locks_onto_the_real_struct() -> None:
    # The C4f acceptance. Same world, same sweep order — but now each candidate is asked to move,
    # and only the struct the player is actually controlling does.
    decoy = planted_decoy()
    located = locate_player_struct(
        decoy.before,
        module=MODULE,
        module_base=MODULE_BASE,
        manifest=_manifest(),
        during=[decoy.after],
    )
    assert located is not None
    assert located.match.p1_base == P1_BASE
    assert located.match.base_offset == BASE_OFFSET
    assert located.match.stride == STRIDE
    assert located.considered == 2  # both structs were structurally plausible ...
    assert located.accepted == 1 and not located.ambiguous  # ... exactly one behaved
    behavior = located.behavior
    assert behavior is not None
    assert behavior.acting_move_changed and behavior.opponent_damaged
    assert behavior.score == 3


def test_a_frozen_decoy_is_not_accepted_even_when_nobody_acted() -> None:
    # If the user never performs the action, no struct's move_id changes and NOTHING is accepted.
    # The scan fails closed and tells them to act, rather than falling back to the structural guess
    # that picked the decoy in the first place.
    still = planted_decoy_nobody_acted()
    assert (
        locate_player_struct(
            still.before,
            module=MODULE,
            module_base=MODULE_BASE,
            manifest=_manifest(),
            during=[still.after],
        )
        is None
    )
    result = derive_base_layout(
        still.before,
        module=MODULE,
        module_base=MODULE_BASE,
        manifest=_manifest(),
        seed=_seed(),
        source_after=still.after,
    )
    assert result.player_anchor is None
    notes = " ".join(result.notes)
    assert "BEHAVED" in notes and "jab P2, and jump" in notes


def test_a_whiffed_jab_still_confirms_the_anchor() -> None:
    # The corroborators rank; only the acting player's move_id gates. A jab that misses leaves the
    # dummy's move_id and damage_taken untouched, and requiring either would reject the real struct.
    whiff = planted_whiffed_jab()
    located = locate_player_struct(
        whiff.before,
        module=MODULE,
        module_base=MODULE_BASE,
        manifest=_manifest(),
        during=[whiff.after],
    )
    assert located is not None and located.match.p1_base == P1_BASE
    behavior = located.behavior
    assert behavior is not None
    assert behavior.acting_move_changed
    assert not behavior.opponent_damaged and not behavior.opponent_move_changed
    assert behavior.score == 1


def test_a_zeroed_page_is_never_a_strong_candidate_at_char_id_min_zero() -> None:
    # C4f set char_id_min to 1 to keep zeroed pages out; C4g put it back to 0, because Jin's real id
    # may BE 0 on this build and a sieve that drops the answer cannot be repaired downstream. The
    # flooding worry is real at the per-struct level and empty at the level that matters: strong
    # acceptance needs a struct reading Kazuya's 12 at a constant stride, and a zeroed page has no
    # 12 to offer. (The behavioral oracle is the second, independent backstop — a zeroed move_id
    # cannot change; see the decoy tests above, which now run with char_id_min at 0.)
    manifest = _manifest()
    assert manifest.char_id_min == 0
    spec = manifest.base_scan
    assert spec is not None
    zeroed = zeroed_landing_source()

    # The per-struct check passes on zeroes. Stated plainly, because it is the premise of the worry.
    assert _player_oracle_ok(zeroed, P1_BASE, spec, manifest) == 0

    located = locate_player_struct(
        zeroed,
        module=MODULE,
        module_base=MODULE_BASE,
        manifest=manifest,
        during=[zeroed_landing_source()],
    )
    assert located is not None
    assert not located.match.strong, "zeroes must never resolve a P2 stride"
    assert located.behavior is None  # nothing to confirm: a weak landing has no opponent to compare

    table, _report = discover_base(
        zeroed,
        module=MODULE,
        module_base=MODULE_BASE,
        game_version=DETECTED_EXE_VERSION,
        manifest=manifest,
        seed=_seed(),
        seed_version="2.01.01",
        source_after=zeroed_landing_source(),
    )
    assert table is None, "a zeroed landing must never reach a written table"


# ---------------------------------------------------------------------------
# C4g: the oracle is WINDOWED. move_id is transient, so "changed at the end" is unsatisfiable.
# ---------------------------------------------------------------------------
#
# This pair is the regression guard for exactly what bit the user on the live game. The planted
# struct's move_id changes only in the MIDDLE sample of the window: Jin jabs and jumps, then lands
# and idles back to the move_id he had at round start. C4f compared round start against the last
# instant, so it saw a frozen field and rejected the player's own struct — the live scan found zero
# behaving candidates and wrote no table, not because the struct was missing but because a human
# cannot alt-tab and press Enter inside a half-second animation.


def _p1_move_id(source: FlatMemorySource) -> int:
    return int(struct.unpack("<I", source.read(P1_BASE + MOVE_ID_OFFSET, 4))[0])


def test_the_two_instant_oracle_rejects_a_move_id_that_was_only_transiently_different() -> None:
    window = planted_transient_action()
    # The struct really does read identically at both ends — that is the whole trap.
    assert _p1_move_id(window.before) == _p1_move_id(window.during[-1])
    assert any(_p1_move_id(s) != _p1_move_id(window.before) for s in window.during)

    assert (
        locate_player_struct(
            window.before,
            module=MODULE,
            module_base=MODULE_BASE,
            manifest=_manifest(),
            during=[window.during[-1]],  # the C4f oracle: round start vs. one late instant
        )
        is None
    ), "the two-instant oracle rejects the real struct — the bug C4g exists to fix"


def test_the_windowed_oracle_accepts_a_move_id_that_changed_in_any_sample() -> None:
    window = planted_transient_action()
    located = locate_player_struct(
        window.before,
        module=MODULE,
        module_base=MODULE_BASE,
        manifest=_manifest(),
        during=window.during,
    )
    assert located is not None
    assert located.match.p1_base == P1_BASE
    assert located.match.base_offset == BASE_OFFSET
    assert located.match.stride == STRIDE

    behavior = located.behavior
    assert behavior is not None
    assert behavior.samples == 3
    assert behavior.acting_move_changed
    # The corroborators are "ever" signals too: the dummy's hit reaction and its damage_taken are
    # both back to their round-start readings by the last sample, exactly as the jab is.
    assert behavior.opponent_move_changed and behavior.opponent_damaged
    assert behavior.score == 3
    # The position scan's "after" base is the LAST sample's resolve, not the middle one's.
    assert behavior.p1_after_base == P1_BASE


def test_a_windowed_derivation_writes_a_table_a_two_instant_one_would_not(tmp_path: Path) -> None:
    # The same claim one level up: the window is what turns a fail-closed run into a written table.
    window = planted_transient_action()
    two_instant, _report = discover_base(
        window.before,
        module=MODULE,
        module_base=MODULE_BASE,
        game_version=DETECTED_EXE_VERSION,
        manifest=_manifest(),
        seed=_seed(),
        seed_version="2.01.01",
        source_after=window.during[-1],
    )
    assert two_instant is None, "no table, and the user is told to act — the live C4f outcome"

    table, report = discover_base(
        window.before,
        module=MODULE,
        module_base=MODULE_BASE,
        game_version=DETECTED_EXE_VERSION,
        manifest=_manifest(),
        seed=_seed(),
        seed_version="2.01.01",
        source_after=window.during[-1],
        during=window.during,
        discovered_at="2026-07-09T00:00:00Z",
    )
    assert table is not None and report.ok
    persist(table, report, offsets_dir=tmp_path)
    written = select_offset_table(DETECTED_EXE_VERSION, tmp_path)
    assert written.players.anchor.base_offset == BASE_OFFSET
    assert written.players.stride == STRIDE


def test_the_derivation_says_when_a_landing_was_never_confirmed() -> None:
    # Without a second snapshot the scan can only apply the single-instant check. It must say so:
    # a report that reads the same either way is how C4e's wrong landing looked plausible.
    result = derive_base_layout(
        planted_chain().before,
        module=MODULE,
        module_base=MODULE_BASE,
        manifest=_manifest(),
        seed=_seed(),
    )
    assert result.player_anchor is not None
    assert "SINGLE INSTANT" in " ".join(result.notes)


def test_live_path_may_pre_sweep_the_player_without_derive_re_sweeping() -> None:
    # Same contract the global sweep has, and for the same reason: the live orchestration sweeps
    # before the user acts and confirms after, so a `located=None` coming back from it means
    # "nothing behaved", not "not swept yet". Re-sweeping here would re-read one instant — minutes
    # of work that can only reproduce the single-instant answer C4f exists to reject.
    chain = planted_chain()
    result = derive_base_layout(
        chain.before,
        module=MODULE,
        module_base=MODULE_BASE,
        manifest=_manifest(),
        seed=_seed(),
        source_after=chain.after,
        sweep_player=False,
    )
    assert result.player_anchor is None
    assert "BEHAVED" in " ".join(result.notes)


def test_the_acting_player_index_is_bounded_to_p1_or_p2() -> None:
    # `moving_player` indexes a two-element (P1, P2) tuple inside the behavioral oracle. Bounding it
    # at load time turns a typo into a manifest validation error instead of an IndexError raised
    # minutes into a live sweep, after the user has already performed the action.
    with pytest.raises(ValidationError):
        ProbeManifest.model_validate(_manifest().model_dump() | {"moving_player": 2})


def test_the_derivation_reports_the_behavioral_evidence() -> None:
    result = _derive()
    notes = " ".join(result.notes)
    assert "confirmed BEHAVIORALLY" in notes
    assert "1 of 1 structurally-plausible landings passed" in notes


# ---------------------------------------------------------------------------
# The AOB signature is the durable re-find artifact
# ---------------------------------------------------------------------------


def test_signature_wildcards_the_pointer_and_rematches_the_slot() -> None:
    source = planted_chain().before
    spec = _manifest().base_scan
    assert spec is not None
    signature = extract_signature(source, MODULE_BASE, BASE_OFFSET, spec)
    assert signature is not None
    tokens = signature.pattern.split()
    # 16 context + 8 wildcarded pointer bytes + 16 context; only the pointer is wild.
    assert len(tokens) == spec.aob_window_before + 8 + spec.aob_window_after
    assert tokens.count("??") == 8
    assert all(t == "??" for t in tokens[16:24])
    assert signature.slot_delta == spec.aob_window_before

    image = parse_module_image(lambda rva, n: source.read(MODULE_BASE + rva, n))
    assert find_by_signature(source, MODULE_BASE, image, signature) == BASE_OFFSET


def test_signature_rematches_when_the_pointer_value_changes() -> None:
    # The whole point: the slot's *contents* shift every build/run; the surrounding bytes do not.
    # A signature derived from one image must still find the slot when the pointer has moved.
    spec = _manifest().base_scan
    assert spec is not None
    signature = extract_signature(planted_chain().before, MODULE_BASE, BASE_OFFSET, spec)
    assert signature is not None
    moved = relocated_pointer_source(0x7FFE_1234_5000)
    image = parse_module_image(lambda rva, n: moved.read(MODULE_BASE + rva, n))
    assert find_by_signature(moved, MODULE_BASE, image, signature) == BASE_OFFSET


def test_signature_is_persisted_on_the_derived_anchor() -> None:
    result = _derive()
    assert result.player_anchor is not None
    assert result.player_anchor.signature is not None
    assert "??" in result.player_anchor.signature.pattern


def test_a_seed_signature_takes_the_fast_re_find_path() -> None:
    # A table carrying a signature lets the next run skip the full candidate sweep — but the slot it
    # re-finds is still put through the oracle before it is accepted.
    first = _derive()
    assert first.player_anchor is not None
    seed = _seed()
    seed.players.anchor.signature = first.player_anchor.signature

    located = locate_player_struct(
        planted_chain().before,
        module=MODULE,
        module_base=MODULE_BASE,
        manifest=_manifest(),
        hint=seed.players.anchor.signature,
    )
    assert located is not None and located.from_signature
    assert located.match.base_offset == BASE_OFFSET

    second = derive_base_layout(
        planted_chain().before,
        module=MODULE,
        module_base=MODULE_BASE,
        manifest=_manifest(),
        seed=seed,
    )
    assert "fast path" in " ".join(second.notes)


def test_a_stale_seed_signature_falls_back_to_the_full_sweep() -> None:
    # A signature that no longer matches must not short-circuit the scan into a wrong answer: it
    # falls through to candidate-generate-and-validate, which finds the slot anyway.
    seed = _seed()
    seed.players.anchor.signature = AobSignature(pattern="DE AD BE EF ?? ?? ?? ??", slot_delta=4)
    result = derive_base_layout(
        planted_chain().before,
        module=MODULE,
        module_base=MODULE_BASE,
        manifest=_manifest(),
        seed=seed,
    )
    assert result.player_anchor is not None
    assert result.player_anchor.base_offset == BASE_OFFSET
    assert "fast path" not in " ".join(result.notes)


# ---------------------------------------------------------------------------
# The two-level P2 case: report, do not invent a stride
# ---------------------------------------------------------------------------


def test_two_level_p2_reports_the_p1_anchor_and_writes_no_table() -> None:
    table, report = discover_base(
        two_level_source(),
        module=MODULE,
        module_base=MODULE_BASE,
        game_version=DETECTED_EXE_VERSION,
        manifest=_manifest(),
        seed=_seed(),
        seed_version="2.01.01",
    )
    assert table is None, "a stride must never be invented when P2 is a separate allocation"
    result = report.result
    assert result.player_anchor is not None
    assert result.player_anchor.base_offset == BASE_OFFSET
    assert result.stride is None
    assert result.player_char_ids is None
    notes = " ".join(result.notes)
    assert "TWO-LEVEL P2" in notes
    assert "per-player anchors" in notes
    assert "no table written" in notes


# ---------------------------------------------------------------------------
# C4e Phase 1: the global/match anchor, assigned by behavior
# ---------------------------------------------------------------------------


def _global_spec() -> GlobalScanSpec:
    spec = _manifest().global_scan
    assert spec is not None
    return spec


def test_locates_the_planted_global_anchor_and_chain() -> None:
    # The manifest seeds SEVERAL candidate chain shapes, because the layout source records the frame
    # counter as one long offset run and does not say where the chain ends and the fields begin. The
    # longer shapes are tried first and dereference into nothing here; the oracle rejects them and
    # the scan keeps going. That is what makes listing guesses safe.
    spec = _global_spec()
    assert len(spec.pointer_paths) > 1
    assert GLOBAL_POINTER_PATH in spec.pointer_paths
    assert spec.pointer_paths[0] != GLOBAL_POINTER_PATH

    chain = planted_chain()
    located = locate_global_struct(
        chain.before, chain.after, module=MODULE, module_base=MODULE_BASE, spec=spec
    )
    assert located is not None
    assert located.match.base_offset == GLOBAL_BASE_OFFSET
    assert list(located.match.pointer_path) == GLOBAL_POINTER_PATH
    assert not located.ambiguous


def test_global_fields_are_assigned_by_behavior_not_by_order() -> None:
    # The planted offsets are deliberately scrambled relative to the field list: the frame counter
    # sits at the MIDDLE offset. Only the two-snapshot behavior (ticks up / holds / counts down)
    # tells them apart, which is the whole point — we know these offsets carry match state, we do
    # not know which is which, and a positional guess would bake a coincidence into a data file.
    chain = planted_chain()
    located = locate_global_struct(
        chain.before, chain.after, module=MODULE, module_base=MODULE_BASE, spec=_global_spec()
    )
    assert located is not None
    assert located.match.offsets == {
        "frame_counter": GLOBAL_FRAME_OFFSET,
        "round": GLOBAL_ROUND_OFFSET,
        "timer_ms": GLOBAL_TIMER_OFFSET,
    }


def test_a_frozen_round_clock_does_not_block_the_global_anchor() -> None:
    # Practice mode usually freezes the round clock, making timer_ms indistinguishable from any
    # other constant. frame_counter + round is what the oracle requires; timer_ms is a bonus.
    spec = _global_spec()
    before = {GLOBAL_TIMER_OFFSET: 41200, GLOBAL_FRAME_OFFSET: 100, GLOBAL_ROUND_OFFSET: 2}
    after = {GLOBAL_TIMER_OFFSET: 41200, GLOBAL_FRAME_OFFSET: 128, GLOBAL_ROUND_OFFSET: 2}
    assigned = assign_global_fields(before, after, spec)
    assert assigned["frame_counter"] == GLOBAL_FRAME_OFFSET
    assert assigned["round"] == GLOBAL_ROUND_OFFSET
    assert "timer_ms" not in assigned


def test_a_ticking_counter_alone_is_not_a_global_struct() -> None:
    # Any long-running process has u32s that tick. Without a steady, in-range round beside it, the
    # landing is rejected rather than accepted as "probably the frame counter".
    spec = _global_spec()
    before = {GLOBAL_TIMER_OFFSET: 900000, GLOBAL_FRAME_OFFSET: 100, GLOBAL_ROUND_OFFSET: 77}
    after = {GLOBAL_TIMER_OFFSET: 900001, GLOBAL_FRAME_OFFSET: 128, GLOBAL_ROUND_OFFSET: 77}
    assert "round" not in assign_global_fields(before, after, spec)


def test_a_counter_jumping_further_than_the_prompt_allows_is_not_a_frame_counter() -> None:
    # C4f Phase 3: the two snapshots straddle ONE interactive prompt, so the counter advances by
    # seconds-to-minutes of frames. A u32 that jumped by ten million is a hash or a byte count, and
    # narrowing the accept set is the cheapest ambiguity reduction available.
    spec = _global_spec()
    assert spec.frame_delta_max < 100_000
    before = {GLOBAL_TIMER_OFFSET: 41200, GLOBAL_FRAME_OFFSET: 100, GLOBAL_ROUND_OFFSET: 2}
    after = {
        GLOBAL_TIMER_OFFSET: 40000,
        GLOBAL_FRAME_OFFSET: 100 + spec.frame_delta_max + 1,
        GLOBAL_ROUND_OFFSET: 2,
    }
    assert "frame_counter" not in assign_global_fields(before, after, spec)


def test_two_slots_reaching_one_struct_are_one_landing_not_two() -> None:
    # The live run's alarming "22 accepted" counted accepting *slots*. Several static globals
    # legitimately point into the same match struct; that is not ambiguity, and reporting it as such
    # would cry wolf on every run. Distinct landings — deduped by RESOLVED STRUCT (gbase), not by
    # slot and not by (gbase, field assignment) — are what the pick has to choose between.
    located = locate_global_struct(
        global_duplicate_slot_source(),
        global_duplicate_slot_source(step=1),
        module=MODULE,
        module_base=MODULE_BASE,
        spec=_global_spec(),
    )
    assert located is not None
    assert located.slots == 2
    assert located.accepted == 1
    assert not located.ambiguous


# ---------------------------------------------------------------------------
# C4g Phase 3: de-ambiguating the global oracle with a timed frame-delta band
# ---------------------------------------------------------------------------


def test_frame_delta_band_tracks_the_windows_duration() -> None:
    spec = _global_spec()
    low, high = frame_delta_band(spec, 5.0)
    assert low <= int(spec.frame_rate * 5.0) <= high  # a real 60fps counter over a 5s window
    assert low > 1, "'it ticked up by one' must fall outside the band"
    assert high < spec.frame_delta_max
    # A caller that could not time the window gets the spec's absolute bounds back, not a band.
    assert frame_delta_band(spec, 0.0) == (spec.frame_delta_min, spec.frame_delta_max)


def test_two_distinct_structs_both_pass_the_unbanded_oracle_and_are_reported_ambiguous() -> None:
    # Both structs tick a counter beside a steady, in-range round: the impostor's advances by 1 per
    # step, the real one's by FRAME_STEP. "Strictly increased, by at most ten minutes of frames"
    # cannot separate them — which is how the live runs accepted 22-28 landings and picked a
    # different base each time.
    located = locate_global_struct(
        global_two_structs_source(),
        global_two_structs_source(step=10),
        module=MODULE,
        module_base=MODULE_BASE,
        spec=_global_spec(),
    )
    assert located is not None
    assert located.accepted == 2 and located.ambiguous
    # The impostor's slot is swept first, so "take the first that passed" would take it. The pick
    # instead prefers the landing whose behavior named the most fields — the real struct's round
    # clock counts down, the impostor's is frozen.
    assert located.match.base_offset == GLOBAL_BASE_OFFSET != GLOBAL_ALT_BASE_OFFSET
    assert located.match.gbase == GLOBAL_BASE
    assert "timer_ms" in located.match.offsets


def test_the_frame_delta_band_rejects_the_coincidental_ticker() -> None:
    # The real struct advances FRAME_STEP per step, so over the window it spans 10 * FRAME_STEP
    # frames; at ~60fps that window lasted about that many sixtieths of a second. Banding the delta
    # to what the window's measured duration allows leaves exactly one landing standing.
    spec = _global_spec()
    seconds = 10 * FRAME_STEP / spec.frame_rate
    located = locate_global_struct(
        global_two_structs_source(),
        global_two_structs_source(step=10),
        module=MODULE,
        module_base=MODULE_BASE,
        spec=spec,
        frame_delta=frame_delta_band(spec, seconds),
    )
    assert located is not None
    assert located.accepted == 1 and not located.ambiguous
    assert located.match.base_offset == GLOBAL_BASE_OFFSET


def test_global_anchor_seeded_and_flagged_when_the_chain_shapes_are_wrong() -> None:
    # A wrong chain shape must leave the seed anchor in place, name frame_counter unresolved, and
    # say so — never invent an anchor the doctor would then silently fail on.
    manifest = _manifest()
    assert manifest.global_scan is not None
    manifest.global_scan.pointer_paths = [[0x999]]
    chain = planted_chain()
    result = derive_base_layout(
        chain.before,
        module=MODULE,
        module_base=MODULE_BASE,
        manifest=manifest,
        seed=_seed(),
        source_after=chain.after,
    )
    assert result.global_anchor == _seed().global_struct.anchor
    assert "frame_counter" in result.unresolved
    assert "global_scan.pointer_paths" in " ".join(result.notes)


def test_live_path_may_pre_sweep_the_global_without_derive_re_sweeping() -> None:
    # The live orchestration must straddle the user's action prompt to observe a frame delta, so it
    # sweeps itself and passes sweep_global=False. "Swept, found nothing" must then stay None rather
    # than triggering a second (same-instant, useless) sweep inside derive.
    chain = planted_chain()
    result = derive_base_layout(
        chain.before,
        module=MODULE,
        module_base=MODULE_BASE,
        manifest=_manifest(),
        seed=_seed(),
        source_after=chain.after,
        sweep_global=False,
    )
    assert result.global_anchor == _seed().global_struct.anchor
    assert "frame_counter" in result.unresolved


# ---------------------------------------------------------------------------
# C4e Phase 2: the encoded state words are seeded, and supersede the legacy booleans
# ---------------------------------------------------------------------------


def test_encoded_state_offsets_are_seeded_into_the_table() -> None:
    result = _derive()
    fields = result.player_offsets()
    assert fields["simple_move_state"].offset == 0x640
    assert fields["stun_type"].offset == 0x644
    assert fields["complex_move_state"].offset == 0x68C
    assert fields["move_frame"].offset == 0x370
    assert fields["counter_state"].offset == 0x5F0
    # They are facts, not findings: no round-start oracle can prove where stun_type lives, so the
    # report must not present them as derived alongside char_id/move_id.
    assert fields["stun_type"].confidence is Confidence.seeded
    assert fields["char_id"].confidence is Confidence.high


def test_legacy_booleans_survive_when_no_state_map_replaces_them() -> None:
    # Dropping the placeholder's per-flag booleans without installing the encoded path would leave
    # the decoder unable to read state at all.
    result = _derive()
    assert result.drop_player_fields == []


def test_a_state_map_supersedes_the_legacy_booleans() -> None:
    chain = planted_chain()
    result = derive_base_layout(
        chain.before,
        module=MODULE,
        module_base=MODULE_BASE,
        manifest=_manifest(),
        seed=_seed(),
        source_after=chain.after,
        state_map=calibrated_state_map(),
    )
    assert "block_stun" in result.drop_player_fields
    assert "simple_state" in result.drop_player_fields
    table = build_offset_table(result, _seed(), game_version="x", discovered_at="now", notes="test")
    assert table.state_codes.encoded_state is not None
    assert "block_stun" not in table.players.fields
    assert table.players.fields["stun_type"].offset == 0x644


def test_a_state_map_naming_an_absent_field_is_refused() -> None:
    # The decoder would raise on every frame; fail at build time with an actionable message instead.
    chain = planted_chain()
    result = derive_base_layout(
        chain.before,
        module=MODULE,
        module_base=MODULE_BASE,
        manifest=_manifest(),
        seed=_seed(),
        source_after=chain.after,
        state_map=EncodedStateSpec(calibrated=True, flags={"no_such_field": {"0": []}}),
    )
    with pytest.raises(OffsetTableError, match="no_such_field"):
        build_offset_table(result, _seed(), game_version="x", discovered_at="now", notes="test")


def test_shipped_state_map_is_an_uncalibrated_skeleton() -> None:
    # It ships empty on purpose (docs/02 §8): only observation can say what stun_type == 3 means,
    # and a guess would be worse than nothing. Every field it names must exist in the manifest's
    # state_fields, or the built table would be unreadable.
    shipped = load_state_map(Path("assets/offsets/state-map.json"))
    spec = _manifest().base_scan
    assert spec is not None
    assert not shipped.calibrated
    assert set(shipped.flags) <= set(spec.state_fields)
    assert all(codes == {} for codes in shipped.flags.values())


# ---------------------------------------------------------------------------
# C4e Phase 3: position lives in a transform component, not the entity struct
# ---------------------------------------------------------------------------


def _derive_component() -> DerivationResult:
    chain = planted_component()
    return derive_base_layout(
        chain.before,
        module=MODULE,
        module_base=MODULE_BASE,
        manifest=_manifest(),
        seed=_seed(),
        source_after=chain.after,
        state_map=calibrated_state_map(),
    )


def test_position_is_found_in_a_component_when_the_struct_has_none() -> None:
    result = _derive_component()
    component = result.components[POSITION_COMPONENT]
    assert component.slot_offset == COMPONENT_SLOT_OFFSET
    assert component.pointer_path == []
    assert component.fields["pos_x"].offset == COMPONENT_TRIPLE_OFFSET
    assert component.fields["pos_z"].offset == COMPONENT_TRIPLE_OFFSET + 8
    assert "pos_x" not in result.unresolved
    # In-struct pos_{x,y,z} are dropped: they would be wrong, not merely absent.
    assert "pos_x" in result.drop_player_fields
    assert "transform component" in " ".join(result.notes)


def test_component_scan_requires_p2_to_resolve_through_the_same_path() -> None:
    # A coincidental moving triple near P1 will not also be a plausible, distinct coordinate at the
    # identical path from P2's base. Without a component_scan spec there is nowhere to look at all.
    manifest = _manifest()
    assert manifest.base_scan is not None
    manifest.base_scan.component_scan = None
    chain = planted_component()
    result = derive_base_layout(
        chain.before,
        module=MODULE,
        module_base=MODULE_BASE,
        manifest=manifest,
        seed=_seed(),
        source_after=chain.after,
    )
    assert result.components == {}
    assert "pos_x" in result.unresolved


def test_built_component_table_carries_the_anchor_through_json(tmp_path: Path) -> None:
    result = _derive_component()
    table = build_offset_table(
        result, _seed(), game_version=DETECTED_EXE_VERSION, discovered_at="now", notes="test"
    )
    persist(
        table,
        build_report(
            result,
            game_version=DETECTED_EXE_VERSION,
            module_base=MODULE_BASE,
            seed=_seed(),
            seed_version="2.01.01",
        ),
        offsets_dir=tmp_path,
    )
    selected = select_offset_table(DETECTED_EXE_VERSION, tmp_path)
    component = selected.players.components[POSITION_COMPONENT]
    assert component.slot_offset == COMPONENT_SLOT_OFFSET
    assert "pos_x" not in selected.players.fields


# ---------------------------------------------------------------------------
# Full acceptance: derive -> build -> write -> select -> decode a real FrameRecord
# ---------------------------------------------------------------------------


def _discovered(tmp_path: Path) -> OffsetTable:
    chain = planted_chain()
    table, report = discover_base(
        chain.before,
        module=MODULE,
        module_base=MODULE_BASE,
        game_version=DETECTED_EXE_VERSION,
        manifest=_manifest(),
        seed=_seed(),
        seed_version="2.01.01",
        source_after=chain.after,
        discovered_at="2026-07-09T00:00:00Z",
    )
    assert table is not None and report.ok
    persist(table, report, offsets_dir=tmp_path)
    return select_offset_table(DETECTED_EXE_VERSION, tmp_path)


def test_built_table_overlays_derived_offsets_on_the_seed(tmp_path: Path) -> None:
    selected = _discovered(tmp_path)
    seed = _seed()
    # Derived: chain anchor, planted stride, oracle + scanned field offsets.
    assert selected.players.anchor.base_offset == BASE_OFFSET
    assert selected.players.anchor.pointer_path == POINTER_PATH
    assert selected.players.stride == STRIDE != seed.players.stride
    assert selected.players.fields["char_id"].offset == 0x168
    assert selected.players.fields["health"].offset == HEALTH_OFFSET
    # Seeded: a field the scan cannot prove is carried forward unchanged, flagged for calibration.
    seeded = selected.players.fields["heat_timer_ms"].offset
    assert seeded == seed.players.fields["heat_timer_ms"].offset
    # The global anchor IS re-derived (C4e): its own static slot + chain, not the seed's static 0.
    assert selected.global_struct.anchor != seed.global_struct.anchor
    assert selected.global_struct.anchor.base_offset == GLOBAL_BASE_OFFSET
    assert selected.global_struct.anchor.pointer_path == GLOBAL_POINTER_PATH


def test_signature_survives_the_json_round_trip(tmp_path: Path) -> None:
    selected = _discovered(tmp_path)
    sig = selected.players.anchor.signature
    assert sig is not None and sig.slot_delta == 16
    assert sig.pattern.split().count("??") == 8


def test_derived_table_decodes_a_frame_through_the_pointer_chain(tmp_path: Path) -> None:
    # The full acceptance path: the table update-offsets writes drives the real C4a decoder, which
    # resolves the player base by following the derived chain into the heap.
    selected = _discovered(tmp_path)
    source = decode_source(selected)
    decoded = decode_frame(source, selected)
    frame = expected_frame()
    for dp, fp in zip(decoded.players, frame.players, strict=True):
        assert dp.pos == pytest.approx(fp.pos, rel=1e-6)
        assert dp.model_copy(update={"pos": fp.pos}) == fp
    assert decoded.model_copy(update={"players": frame.players}) == frame


def test_report_shows_the_chain_and_the_signature(tmp_path: Path) -> None:
    chain = planted_chain()
    _table, report = discover_base(
        chain.before,
        module=MODULE,
        module_base=MODULE_BASE,
        game_version=DETECTED_EXE_VERSION,
        manifest=_manifest(),
        seed=_seed(),
        seed_version="2.01.01",
        source_after=chain.after,
    )
    text = report.render()
    assert f"+0x{BASE_OFFSET:x} -> +0x10 -> +0x68 -> +0x8 -> +0x30" in text
    assert "base AOB signature" in text
    assert f"P1(Jin)={JIN}" in text and f"P2(Kazuya)={KAZUYA}" in text
    assert f"global anchor         : {MODULE}+0x{GLOBAL_BASE_OFFSET:x}" in text
    # The oracle could only claim three of the four match fields (three seeded offsets); the fourth
    # is named as still-seeded rather than quietly left looking derived.
    assert "match_phase not assigned" in " ".join(report.result.notes)
