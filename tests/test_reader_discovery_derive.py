"""Correlation/derivation recovers a planted layout exactly (C4c, docs/02 §4)."""

from __future__ import annotations

from pathlib import Path

from tekken_coach.reader.discovery.derive import (
    DerivationResult,
    DiscoverySnapshots,
    derive_layout,
)
from tekken_coach.reader.discovery.manifest import ProbeManifest, load_probe_manifest
from tekken_coach.reader.discovery.scanners import Region
from tekken_coach.reader.offsets import load_offset_table
from tests.fixtures.reader.planted import (
    JIN_CHAR_ID,
    KAZUYA_CHAR_ID,
    PLANTED_GLOBAL_BASE_OFFSET,
    PLANTED_MODULE,
    PLANTED_MODULE_BASE,
    PLANTED_PLAYER_BASE_OFFSET,
    PLANTED_STRIDE,
    PlantedScan,
    planted_scan,
)

SEED_PATH = Path("assets/offsets/2.01.01.json")
MANIFEST_PATH = Path("assets/offsets/probe-manifest.json")


def _manifest() -> ProbeManifest:
    return load_probe_manifest(MANIFEST_PATH)


def _derive() -> tuple[DerivationResult, PlantedScan]:
    seed = load_offset_table(SEED_PATH)
    scan = planted_scan(seed)
    snap = DiscoverySnapshots(
        player_before=scan.player_before,
        player_after=scan.player_after,
        global_before=scan.global_before,
        global_after=scan.global_after,
    )
    result = derive_layout(
        snap, module=PLANTED_MODULE, module_base=PLANTED_MODULE_BASE, manifest=_manifest()
    )
    return result, scan


def test_recovers_stride_and_anchors_exactly() -> None:
    result, scan = _derive()
    assert result.ok
    assert result.stride == PLANTED_STRIDE
    assert result.player_anchor is not None
    assert result.player_anchor.base_offset == PLANTED_PLAYER_BASE_OFFSET
    assert result.player_anchor.module == PLANTED_MODULE
    assert result.global_anchor is not None
    assert result.global_anchor.base_offset == PLANTED_GLOBAL_BASE_OFFSET


def test_recovers_player_field_offsets_exactly() -> None:
    result, scan = _derive()
    truth = scan.ground_truth.players.fields
    derived = result.player_offsets()
    for name in ("char_id", "health", "move_id", "pos_x", "pos_y", "pos_z"):
        assert derived[name].offset == truth[name].offset, name
        assert derived[name].kind == truth[name].kind, name


def test_recovers_global_frame_counter() -> None:
    result, _ = _derive()
    fc = result.global_offsets()["frame_counter"]
    assert fc.offset == 0
    assert fc.kind == "u32"


def test_discovers_jin_as_the_kazuya_counterpart() -> None:
    result, _ = _derive()
    assert result.player_char_ids == (JIN_CHAR_ID, KAZUYA_CHAR_ID)


def test_no_unresolved_fields_on_a_clean_scan() -> None:
    result, _ = _derive()
    assert result.unresolved == []


def test_fails_to_resolve_when_windows_are_empty() -> None:
    # An all-zero region has no health/Kazuya-id anchors and no changing frame counter: the
    # derivation resolves nothing and reports every derivable field unresolved (no false positives).
    empty = Region(base=PLANTED_MODULE_BASE + PLANTED_PLAYER_BASE_OFFSET, data=bytes(0x2000))
    empty_g = Region(base=PLANTED_MODULE_BASE + PLANTED_GLOBAL_BASE_OFFSET, data=bytes(0x100))
    snap = DiscoverySnapshots(
        player_before=empty, player_after=empty, global_before=empty_g, global_after=empty_g
    )
    result = derive_layout(
        snap, module=PLANTED_MODULE, module_base=PLANTED_MODULE_BASE, manifest=_manifest()
    )
    assert not result.ok
    assert result.player_anchor is None
    assert result.global_anchor is None
    assert set(result.unresolved) >= {"char_id", "health", "move_id", "pos_x", "frame_counter"}
