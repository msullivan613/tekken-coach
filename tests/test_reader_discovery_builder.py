"""Builder + orchestration: derived layout -> schema-valid, decoder-consumable table (C4c)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tekken_coach.reader.decode import decode_frame
from tekken_coach.reader.discovery.builder import build_offset_table
from tekken_coach.reader.discovery.derive import (
    DerivationResult,
    DiscoverySnapshots,
    derive_layout,
)
from tekken_coach.reader.discovery.manifest import load_probe_manifest
from tekken_coach.reader.discovery.orchestrate import discover, persist
from tekken_coach.reader.faults import OffsetTableError, UnknownGameVersionError
from tekken_coach.reader.memory_source import FakeMemorySource
from tekken_coach.reader.offsets import OffsetTable, load_offset_table, select_offset_table
from tests.factories import make_frame_record
from tests.fixtures.reader.encode import advance_on_for, encode_frame, module_base_for
from tests.fixtures.reader.planted import (
    PLANTED_MODULE_BASE,
    PLANTED_PLAYER_BASE_OFFSET,
    PLANTED_STRIDE,
    PlantedScan,
    planted_scan,
)

SEED_PATH = Path("assets/offsets/2.01.01.json")
MANIFEST_PATH = Path("assets/offsets/probe-manifest.json")
DETECTED_EXE_VERSION = "5.02.01"  # the finding from the C4b smoke test (distinct from 2.01.01)


def _seed() -> OffsetTable:
    return load_offset_table(SEED_PATH)


def _derivation() -> tuple[DerivationResult, OffsetTable, PlantedScan]:
    seed = _seed()
    scan = planted_scan(seed)
    snap = DiscoverySnapshots(
        player_before=scan.player_before,
        player_after=scan.player_after,
        global_before=scan.global_before,
        global_after=scan.global_after,
    )
    result = derive_layout(
        snap,
        module=seed.players.anchor.module,
        module_base=PLANTED_MODULE_BASE,
        manifest=load_probe_manifest(MANIFEST_PATH),
    )
    return result, seed, scan


def test_build_overlays_derived_offsets_on_seed() -> None:
    result, seed, _ = _derivation()
    table = build_offset_table(
        result, seed, game_version=DETECTED_EXE_VERSION, discovered_at="2026-07-08T00:00:00Z",
        notes="test",
    )
    # Derived fields take the planted offsets (differ from the seed's).
    assert seed.players.fields["move_id"].offset == 4
    assert table.players.fields["move_id"].offset == 0x44
    assert table.players.stride == PLANTED_STRIDE != seed.players.stride
    assert table.players.anchor.base_offset == PLANTED_PLAYER_BASE_OFFSET
    # A non-derived field is carried forward from the seed unchanged.
    seeded = table.players.fields["heat_timer_ms"].offset
    assert seeded == seed.players.fields["heat_timer_ms"].offset
    # The state-code maps and sanity bounds come from the seed.
    assert table.state_codes == seed.state_codes
    assert table.sanity == seed.sanity
    assert table.game_version == DETECTED_EXE_VERSION


def test_build_refuses_without_the_confident_core() -> None:
    # A derivation that resolved nothing must not yield a table (would omit doctor-gated fields).
    from tekken_coach.reader.discovery.derive import DerivationResult

    empty = DerivationResult(module="m", module_base=0)
    with pytest.raises(OffsetTableError, match="confident core"):
        build_offset_table(
            empty, _seed(), game_version="x", discovered_at="t", notes="n"
        )


def test_built_table_round_trips_through_select_and_decodes(tmp_path: Path) -> None:
    # The full acceptance path: derive -> build -> write+register -> select_offset_table ->
    # decode a FrameRecord with the REAL C4a decoder. Proves the tool's output is reader-consumable.
    result, seed, _ = _derivation()
    table, report = discover(
        planted_snapshots(),
        module=seed.players.anchor.module,
        module_base=PLANTED_MODULE_BASE,
        game_version=DETECTED_EXE_VERSION,
        manifest=load_probe_manifest(MANIFEST_PATH),
        seed=seed,
        seed_version="2.01.01",
        discovered_at="2026-07-08T00:00:00Z",
    )
    assert table is not None and report.ok
    persist(table, report, offsets_dir=tmp_path)

    assert (tmp_path / f"{DETECTED_EXE_VERSION}.json").exists()
    selected = select_offset_table(DETECTED_EXE_VERSION, tmp_path)
    assert selected.game_version == DETECTED_EXE_VERSION

    # Encode a frame with the selected (derived) table and decode it back through decode_frame.
    frame = make_frame_record()
    mb = 0x150000000
    image = encode_frame(frame, selected, module_base=mb)
    source = FakeMemorySource(
        [image], module_bases=module_base_for(selected, mb), advance_on=advance_on_for(selected, mb)
    )
    decoded = decode_frame(source, selected)
    # Everything round-trips; positions match within f32 precision (1.42 is not f32-exact).
    for dp, fp in zip(decoded.players, frame.players, strict=True):
        assert dp.pos == pytest.approx(fp.pos, rel=1e-6)
        assert dp.model_copy(update={"pos": fp.pos}) == fp
    assert decoded.model_copy(update={"players": frame.players}) == frame


def test_persist_is_additive_and_preserves_fail_closed(tmp_path: Path) -> None:
    result, seed, _ = _derivation()
    table, report = discover(
        planted_snapshots(),
        module=seed.players.anchor.module,
        module_base=PLANTED_MODULE_BASE,
        game_version=DETECTED_EXE_VERSION,
        manifest=load_probe_manifest(MANIFEST_PATH),
        seed=seed,
        seed_version="2.01.01",
    )
    assert table is not None
    persist(table, report, offsets_dir=tmp_path)
    # The new version resolves; an unknown *other* version still fails closed (docs/02 §3).
    assert select_offset_table(DETECTED_EXE_VERSION, tmp_path).game_version == DETECTED_EXE_VERSION
    with pytest.raises(UnknownGameVersionError):
        select_offset_table("9.99.99", tmp_path)


def test_report_render_shows_derived_seeded_and_runbook() -> None:
    result, seed, _ = _derivation()
    _table, report = discover(
        planted_snapshots(),
        module=seed.players.anchor.module,
        module_base=PLANTED_MODULE_BASE,
        game_version=DETECTED_EXE_VERSION,
        manifest=load_probe_manifest(MANIFEST_PATH),
        seed=seed,
        seed_version="2.01.01",
    )
    text = report.render()
    assert DETECTED_EXE_VERSION in text
    assert "char_id" in text and "frame_counter" in text
    assert "P1(Jin)=1" in text and "P2(Kazuya)=12" in text
    # A seeded field appears in the calibration list, not the derived list.
    assert "seeded from table 2.01.01" in text
    assert "practice" in text.lower()  # the runbook rode along
    assert "update-offsets" in text


def planted_snapshots() -> DiscoverySnapshots:
    scan = planted_scan(_seed())
    return DiscoverySnapshots(
        player_before=scan.player_before,
        player_after=scan.player_after,
        global_before=scan.global_before,
        global_after=scan.global_after,
    )
