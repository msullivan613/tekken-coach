"""C4h: derive the Tekken 8 player layout from behavior, with no seeded within-struct offsets.

C4d/C4e seed the field offsets and the pointer chain and derive only the static ``base_offset``. On
build 5.02.01 those seeds are stale, so C4h removes the dependence: it locates the entity struct on
the enumerated heap by *behavior*, derives every field offset + the stride + Jin's id as outputs,
and reverse-scans the static data for a pointer path that survives a reallocation.

The planted world (:func:`planted_heap_world`) exposes the transform-component heap as enumerable
regions plus a *reallocated* variant. Every offset the scan reports is planted at a value the probe
manifest does not carry, so a passing derivation proves the scan **found** it — most pointedly Jin's
id, which is discovered as the P1 counterpart, never seeded (community data says 6; here it is 1).
the scan reports whatever is actually there).
"""

from __future__ import annotations

from pathlib import Path

from tekken_coach.reader.decode import resolve_anchor
from tekken_coach.reader.discovery.heapscan import (
    DeriveInputs,
    ReversePath,
    build_pointer_index,
    confirm_across_realloc,
    confirm_entity_layout,
    derive_layout_scan,
    entity_candidates,
    locate_entity_layout,
    reverse_pointer_paths,
)
from tekken_coach.reader.discovery.manifest import ProbeManifest, load_probe_manifest
from tekken_coach.reader.discovery.pe import parse_module_image
from tekken_coach.reader.discovery.scanners import Region
from tekken_coach.reader.memory_source import MemorySource
from tekken_coach.reader.offsets import POSITION_COMPONENT, OffsetTable, load_offset_table
from tests.fixtures.reader.planted_chain import (
    JIN,
    KAZUYA,
    MODULE,
    MODULE_BASE,
    P1_BASE,
    STRIDE,
    HeapCaptures,
    planted_heap_idle,
    planted_heap_world,
)
from tests.fixtures.reader.state_map import calibrated_state_map


def _manifest() -> ProbeManifest:
    return load_probe_manifest(Path("assets/offsets/probe-manifest.json"))


def _seed() -> OffsetTable:
    return load_offset_table(Path("assets/offsets/2.01.01.json"))


def _inputs(caps: HeapCaptures) -> DeriveInputs:
    return DeriveInputs(
        before=caps.before, during=caps.during, after=caps.after, realloc=caps.realloc
    )


def _heap_buffers(source: MemorySource) -> list[Region]:
    return [Region(base=r.base, data=source.read(r.base, r.size)) for r in source.regions()]


# --- Phase 2: locate the entity struct by behavior ----------------------------------------------


def test_entity_candidates_pair_kazuya_with_a_similar_struct_at_a_stride() -> None:
    m = _manifest()
    assert m.derive_scan is not None
    caps = planted_heap_world()
    candidates = entity_candidates(caps.before, manifest=m, spec=m.derive_scan)
    # The zero-immune similarity discriminator collapses the many spurious (zeroed, Kazuya) pairs to
    # the one real pair: Jin's char_id one stride below Kazuya's, both structs sharing non-zero.
    assert [(c.p1_char, c.p2_char, c.stride, c.jin_id) for c in candidates] == [
        (P1_BASE + 0x168, P1_BASE + STRIDE + 0x168, STRIDE, JIN)
    ]


def test_locate_entity_layout_derives_jin_id_stride_and_move_id() -> None:
    m = _manifest()
    assert m.derive_scan is not None
    caps = planted_heap_world()
    layout = locate_entity_layout(caps.before, caps.during, manifest=m, spec=m.derive_scan)
    assert layout is not None
    assert layout.jin_id == JIN and layout.jin_id != m.kazuya_char_id  # discovered, not seeded
    assert layout.kazuya_id == KAZUYA
    assert layout.stride == STRIDE
    assert layout.p1_char == P1_BASE + 0x168
    # move_id is the acting-correlated field the window revealed; its offset is derived, not seeded.
    assert layout.move_id_addr == P1_BASE + 0x528
    assert layout.behavior.accepted


def test_confirm_fails_closed_when_the_acting_player_never_moved() -> None:
    m = _manifest()
    assert m.derive_scan is not None
    caps = planted_heap_idle()  # the struct is there, but move_id never changes
    candidates = entity_candidates(caps.before, manifest=m, spec=m.derive_scan)
    assert candidates, "the struct is still structurally present at round start"
    layout = confirm_entity_layout(
        caps.before, caps.during, candidates, manifest=m, spec=m.derive_scan
    )
    assert layout is None  # a frozen struct is not accepted, however plausible it looks


# --- Phase 3: reverse pointer scan for a static, reallocation-surviving path ---------------------


def test_reverse_scan_finds_a_static_path_to_the_heap_struct() -> None:
    m = _manifest()
    assert m.derive_scan is not None
    caps = planted_heap_world()
    image = parse_module_image(lambda rva, n: caps.before.read(MODULE_BASE + rva, n))
    heap = _heap_buffers(caps.before)
    index = build_pointer_index(
        caps.before, module_base=MODULE_BASE, image=image, heap=heap, max_entries=100_000
    )
    target = P1_BASE + 0x168  # the derived char_id address
    ranges = [
        (MODULE_BASE + s.rva, MODULE_BASE + s.rva + s.virtual_size) for s in image.data_sections()
    ]
    paths = reverse_pointer_paths(
        index, target=target, module_base=MODULE_BASE, data_ranges=ranges, spec=m.derive_scan
    )
    assert paths, "a static slot must reach the struct through the heap chain"
    anchor = paths[0].anchor(MODULE)
    assert resolve_anchor(caps.before, anchor) == target  # the derived path actually lands on it


def test_only_a_path_that_survives_a_reallocation_is_kept() -> None:
    m = _manifest()
    caps = planted_heap_world()
    image = parse_module_image(lambda rva, n: caps.before.read(MODULE_BASE + rva, n))
    heap = _heap_buffers(caps.before)
    assert m.derive_scan is not None
    index = build_pointer_index(
        caps.before, module_base=MODULE_BASE, image=image, heap=heap, max_entries=100_000
    )
    target = P1_BASE + 0x168
    ranges = [
        (MODULE_BASE + s.rva, MODULE_BASE + s.rva + s.virtual_size) for s in image.data_sections()
    ]
    real = reverse_pointer_paths(
        index, target=target, module_base=MODULE_BASE, data_ranges=ranges, spec=m.derive_scan
    )
    # A fabricated static path resolves in the FIRST capture only by luck; it will not reach the
    # reallocated struct in the second, so the round-reset confirmation must drop it.
    from tests.fixtures.reader.planted_chain import GLOBAL_BASE_OFFSET

    bogus = ReversePath(base_offset=GLOBAL_BASE_OFFSET, offsets=(0x0,))
    layout = locate_entity_layout(caps.before, caps.during, manifest=m, spec=m.derive_scan)
    assert layout is not None
    survivors = confirm_across_realloc(
        [*real, bogus],
        caps.before,
        caps.realloc,
        module=MODULE,
        target_before=target,
        layout=layout,
        manifest=m,
    )
    assert real[0] in survivors
    assert bogus not in survivors


# --- Phase 4: field derivation from the real base, then the whole pipeline -----------------------


def test_derive_layout_scan_resolves_the_full_layout() -> None:
    m = _manifest()
    caps = planted_heap_world()
    result = derive_layout_scan(
        _inputs(caps),
        module=MODULE,
        module_base=MODULE_BASE,
        manifest=m,
        seed=_seed(),
        state_map=calibrated_state_map(),
    )
    assert not result.unresolved, result.notes
    assert result.stride == STRIDE
    assert result.player_char_ids == (JIN, KAZUYA)
    # The anchor resolves to the pointer target (the struct base), reached by a DERIVED chain — no
    # seeded pointer_path. char_id sits at the derived offset, not offset 0.
    assert result.player_anchor is not None
    struct_base = resolve_anchor(caps.before, result.player_anchor)
    fields = {f.name: f for f in result.fields if f.scope == "player"}
    assert struct_base + fields["char_id"].offset == P1_BASE + 0x168
    assert struct_base + fields["move_id"].offset == P1_BASE + 0x528
    # damage_taken is DERIVED from the landed jab (0 -> >0), not seeded.
    assert struct_base + fields["damage_taken"].offset == P1_BASE + 0x1260
    assert fields["damage_taken"].confidence.value == "high"
    assert result.max_health == 200  # computed health = round_start_health - damage_taken
    # position moved out to the transform component.
    assert POSITION_COMPONENT in result.components
    assert "pos_x" not in fields


def test_derive_writes_no_table_when_nothing_behaves() -> None:
    m = _manifest()
    caps = planted_heap_idle()
    result = derive_layout_scan(
        _inputs(caps),
        module=MODULE,
        module_base=MODULE_BASE,
        manifest=m,
        seed=_seed(),
        state_map=calibrated_state_map(),
    )
    assert result.player_anchor is None  # fail closed
    assert "char_id" in result.unresolved
    assert any("BEHAVED like the acting player" in n for n in result.notes)
