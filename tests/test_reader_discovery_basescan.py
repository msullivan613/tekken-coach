"""C4d code-signature base derivation: candidate scan + layout oracle (docs/02 §3/§4).

The planted world (``tests/fixtures/reader/planted_chain.py``) hides the player struct behind a
static pointer in the module's ``.data`` plus a 4-level pointer chain — the shape Tekken 8's
reallocating heap struct forces. Nothing tells the scan where the slot is: ``BASE_OFFSET``, the
``STRIDE``, and the ``health``/``pos`` offsets are planted at values the manifest does not carry.
So a passing derivation proves the scan *found* them, and the final acceptance test proves the
resulting table drives the real C4a decoder through the chain.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tekken_coach.reader.decode import decode_frame, resolve_anchor
from tekken_coach.reader.discovery.basescan import (
    _section_passes,
    derive_base_layout,
    extract_signature,
    find_by_signature,
    find_candidate_slots,
    locate_player_struct,
)
from tekken_coach.reader.discovery.derive import DerivationResult
from tekken_coach.reader.discovery.manifest import ProbeManifest, load_probe_manifest
from tekken_coach.reader.discovery.orchestrate import discover_base, persist
from tekken_coach.reader.discovery.pe import parse_module_image
from tekken_coach.reader.offsets import (
    AobSignature,
    OffsetTable,
    load_offset_table,
    select_offset_table,
)
from tests.fixtures.reader.planted_chain import (
    BASE_OFFSET,
    HEALTH_OFFSET,
    JIN,
    KAZUYA,
    MODULE,
    MODULE_BASE,
    P1_BASE,
    POINTER_PATH,
    POS_OFFSET,
    STRIDE,
    decode_source,
    expected_frame,
    planted_chain,
    relocated_pointer_source,
    two_level_source,
)

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
    # library streams what it is doing (PE parse -> which pass it is sweeping -> the strong match).
    source = planted_chain().before
    msgs: list[str] = []
    located = locate_player_struct(
        source,
        module=MODULE,
        module_base=MODULE_BASE,
        manifest=_manifest(),
        progress=msgs.append,
    )
    assert located is not None and located.match.strong
    joined = "\n".join(msgs)
    assert "parsed PE" in joined
    assert "writable .data" in joined
    assert "strong match" in joined


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
    located = locate_player_struct(
        planted_chain().before, module=MODULE, module_base=MODULE_BASE, manifest=_manifest()
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
    # The global anchor is not re-derived by the base scan; it comes from the seed.
    assert selected.global_struct.anchor == seed.global_struct.anchor


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
    assert "global anchoring is a separate calibration" in " ".join(report.result.notes)
