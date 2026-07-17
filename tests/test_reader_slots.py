"""Brief #11 Stage 1: the pointer-slot classifier, and the deref sweep's pure core.

The classifier's whole job is to separate a plausible heap pointer from the noise a struct is full
of, so the tests plant exactly the confusable cases: a null slot, a misaligned value, a
plausible-but-unmapped value, and a real one. The live enumeration is a thin shell
(``# pragma: no cover``); everything decidable is here.
"""

from __future__ import annotations

import pytest

from tekken_coach.reader.decode import resolve_component, unpack_scalar
from tekken_coach.reader.memory_source import FakeMemorySource, MemoryRegion
from tekken_coach.reader.probe import (
    PollRate,
    SlotPath,
    WatchPoint,
    assemble_row,
    block_span,
    build_read_plan,
    heartbeat_line,
    parse_watch_behind,
    slice_point,
)
from tekken_coach.reader.slots import (
    MIN_POINTER_VALUE,
    RegionIndex,
    classify_slots,
    format_slot_table,
    is_plausible_pointer,
    pointer_candidates,
)

# A planted "heap": two mapped regions the classifier will accept pointers into.
HEAP_A = MemoryRegion(base=0x2_0000_0000, size=0x1000)
HEAP_B = MemoryRegion(base=0x3_0000_0000, size=0x1000)
REGIONS = RegionIndex([HEAP_A, HEAP_B])

# The four confusable slot values, in the order the struct below plants them.
NULL_SLOT = 0
MISALIGNED = HEAP_A.base + 4  # inside a region, but not 8-byte aligned: not a pointer slot value
UNMAPPED = 0x9_0000_0000  # aligned and pointer-sized, but no region maps it
REAL_P1 = HEAP_A.base + 0x40
REAL_P2 = HEAP_B.base + 0x40  # the same slot in P2's struct -> a DIFFERENT object (per-player)
SHARED = HEAP_A.base + 0x80  # both players point at the same object: engine-global, not per-player


def _struct(values: list[int]) -> bytes:
    """A struct block of 8-byte slots holding ``values``."""
    return b"".join(v.to_bytes(8, "little") for v in values)


# Slot layout (by offset) both players share:
#   0x00 null | 0x08 misaligned | 0x10 unmapped | 0x18 real (per-player) | 0x20 shared
P1_BLOCK = _struct([NULL_SLOT, MISALIGNED, UNMAPPED, REAL_P1, SHARED])
P2_BLOCK = _struct([NULL_SLOT, MISALIGNED, UNMAPPED, REAL_P2, SHARED])


class TestRegionIndex:
    def test_contains_only_inside_a_committed_region(self) -> None:
        assert REGIONS.contains(HEAP_A.base)
        assert REGIONS.contains(HEAP_A.base + 0x800)
        assert not REGIONS.contains(HEAP_A.base - 8), "below every region"
        assert not REGIONS.contains(HEAP_A.end), "one past the end is not in the region"
        assert not REGIONS.contains(0x9_0000_0000), "the gap between regions is unmapped"

    def test_a_read_straddling_the_region_end_is_not_contained(self) -> None:
        # The last 8 bytes fit; 16 bytes from there run off the mapping.
        assert REGIONS.contains(HEAP_A.end - 8, 8)
        assert not REGIONS.contains(HEAP_A.end - 8, 16)

    def test_room_at_bounds_a_stage_2_sweep(self) -> None:
        assert REGIONS.room_at(HEAP_A.base + 0x40) == 0x1000 - 0x40
        assert REGIONS.room_at(UNMAPPED) == 0, "unmapped has no room"

    def test_empty_region_map_contains_nothing(self) -> None:
        assert not RegionIndex([]).contains(HEAP_A.base)


class TestPlausibility:
    def test_the_four_confusable_cases(self) -> None:
        assert not is_plausible_pointer(NULL_SLOT, REGIONS), "null is not a pointer"
        assert not is_plausible_pointer(MISALIGNED, REGIONS), "mapped but misaligned"
        assert not is_plausible_pointer(UNMAPPED, REGIONS), "aligned but no region maps it"
        assert is_plausible_pointer(REAL_P1, REGIONS), "aligned, non-null, mapped"

    def test_a_small_aligned_integer_is_not_a_pointer(self) -> None:
        # A count or a flags word is often 8-aligned; the null page is never mapped, so the region
        # check would reject it anyway — this just keeps the bisect off the hot path.
        assert not is_plausible_pointer(MIN_POINTER_VALUE - 8, REGIONS)

    def test_pointer_candidates_are_struct_relative_and_aligned(self) -> None:
        found = list(pointer_candidates(P1_BLOCK, start=0x100))
        assert [off for off, _ in found] == [0x100, 0x108, 0x110, 0x118, 0x120]
        assert found[3] == (0x118, REAL_P1)

    def test_pointer_candidates_rejects_an_unaligned_start(self) -> None:
        with pytest.raises(ValueError, match="8-byte aligned"):
            list(pointer_candidates(P1_BLOCK, start=0x4))

    def test_a_trailing_partial_slot_is_not_yielded(self) -> None:
        assert list(pointer_candidates(b"\x01" * 7)) == []


class TestClassifySlots:
    def test_only_plausible_slots_are_reported(self) -> None:
        findings = classify_slots([[P1_BLOCK, P2_BLOCK]], REGIONS)
        assert [f.offset for f in findings] == [0x18, 0x20], "null/misaligned/unmapped are dropped"

    def test_the_per_player_slot_outranks_the_shared_one(self) -> None:
        findings = classify_slots([[P1_BLOCK, P2_BLOCK]], REGIONS)
        per_player, shared = findings
        assert per_player.offset == 0x18
        assert per_player.chase, "per-player + both + stable is the whole signature"
        assert per_player.values == (REAL_P1, REAL_P2)

        assert shared.offset == 0x20
        assert shared.both, "both players hold a readable address"
        assert not shared.per_player, "...but the same one — engine-global, not this player's input"
        assert not shared.chase

    def test_a_churning_slot_is_reported_but_not_chased(self) -> None:
        moved = _struct([NULL_SLOT, MISALIGNED, UNMAPPED, HEAP_A.base + 0x60, SHARED])
        findings = classify_slots([[P1_BLOCK, P2_BLOCK], [moved, P2_BLOCK]], REGIONS)
        churner = next(f for f in findings if f.offset == 0x18)
        assert not churner.stable, "P1's value moved between polls"
        assert churner.per_player
        assert not churner.chase, "a component anchor does not churn"
        assert churner.values == (HEAP_A.base + 0x60, REAL_P2), "reports the last value seen"

    def test_a_slot_plausible_for_one_player_only_is_kept_but_not_chased(self) -> None:
        half = _struct([NULL_SLOT, MISALIGNED, UNMAPPED, NULL_SLOT, SHARED])
        findings = classify_slots([[P1_BLOCK, half]], REGIONS)
        one_side = next(f for f in findings if f.offset == 0x18)
        assert one_side.plausible == (True, False)
        assert not one_side.both, "the structs are symmetric; a real component resolves for both"
        assert not one_side.chase

    def test_stability_needs_more_than_one_poll_but_one_poll_still_classifies(self) -> None:
        findings = classify_slots([[P1_BLOCK, P2_BLOCK]], REGIONS)
        assert all(f.stable for f in findings), "a single poll cannot show churn"

    def test_no_samples_is_no_findings(self) -> None:
        assert classify_slots([], REGIONS) == []

    def test_ragged_samples_are_rejected_rather_than_silently_misread(self) -> None:
        with pytest.raises(ValueError, match="same width"):
            classify_slots([[P1_BLOCK, P2_BLOCK[:16]]], REGIONS)
        with pytest.raises(ValueError, match="same number of players"):
            classify_slots([[P1_BLOCK, P2_BLOCK], [P1_BLOCK]], REGIONS)


class TestSlotTable:
    def test_the_table_names_the_chase_slot_and_offers_the_stage_2_command(self) -> None:
        findings = classify_slots([[P1_BLOCK, P2_BLOCK]], REGIONS)
        out = "\n".join(format_slot_table(findings, REGIONS))
        assert "1 worth chasing" in out
        assert "CHASE" in out and "shared" in out
        assert '--watch-behind "0x18:0x0-0x100:u8"' in out, "Stage 1's output is Stage 2's input"
        assert "debug/behind-1.jsonl" in out, "a distinct record name per run (#10 lost a log)"

    def test_a_clean_table_recommends_the_controller_path_instead_of_a_guess(self) -> None:
        findings = classify_slots([[P1_BLOCK, P1_BLOCK]], REGIONS)  # every slot shared
        out = "\n".join(format_slot_table(findings, REGIONS))
        assert "No slot is per-player + stable" in out
        assert "PlayerController" in out
        assert "--watch-behind" not in out, "never propose a Stage 2 pass with nothing to chase"

    def test_top_truncates_but_says_so(self) -> None:
        findings = classify_slots([[P1_BLOCK, P2_BLOCK]], REGIONS)
        out = "\n".join(format_slot_table(findings, REGIONS, top=1))
        assert "and 1 more" in out


class TestParseWatchBehind:
    def test_a_one_hop_slot_sweep(self) -> None:
        points = parse_watch_behind("0x38:0x0-0x4:u8")
        assert [p.name for p in points] == ["@0x38+0x0", "@0x38+0x1", "@0x38+0x2", "@0x38+0x3"]
        assert all(p.slot == SlotPath(slot_offset=0x38) for p in points)

    def test_a_multi_hop_slot_matches_transforms_shape(self) -> None:
        # transform is slot_offset 0x20, pointer_path [8] — the precedent this brief runs on.
        (point,) = parse_watch_behind("0x20/8:0x1c:u32")
        assert point.slot == SlotPath(slot_offset=0x20, pointer_path=(8,))
        assert point.name == "@0x20/0x8+0x1c"
        assert point.offset == 0x1C
        assert point.kind == "u32"

    def test_the_slot_path_converts_to_the_offset_tables_own_anchor(self) -> None:
        # A hit is a data edit, not a schema change: what found it is what reads it.
        component = SlotPath(slot_offset=0x20, pointer_path=(8,)).to_component()
        assert component.slot_offset == 0x20
        assert component.pointer_path == [8]

    def test_several_slots_in_one_spec(self) -> None:
        points = parse_watch_behind("0x38:0x0-0x2:u8, 0x40:0x0-0x2:u8")
        assert [p.name for p in points] == ["@0x38+0x0", "@0x38+0x1", "@0x40+0x0", "@0x40+0x1"]

    @pytest.mark.parametrize(
        ("spec", "message"),
        [
            ("0x38:0x0-0x100", "must be SLOT"),
            ("0x38:0x0-0x100:u9", "unknown kind"),
            ("zz:0x0-0x100:u8", "not a valid slot offset"),
            ("-8:0x0-0x100:u8", "non-negative"),
            ("0x38:0x100-0x0:u8", "END must be greater"),
            ("", "empty"),
        ],
    )
    def test_a_malformed_spec_is_an_actionable_error_not_a_traceback(
        self, spec: str, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            parse_watch_behind(spec)


class TestBlockRead:
    """The perf fix (#10's sweep ran at 4.7 Hz doing one syscall per offset)."""

    def test_a_sweep_becomes_one_read_per_object(self) -> None:
        points = parse_watch_behind("0x38:0x0-0x100:u8")
        assert len(points) == 256
        plans = build_read_plan(points)
        assert len(plans) == 1, "256 offsets behind one slot = ONE block read, not 256"
        assert (plans[0].start, plans[0].size) == (0x0, 0x100)

    def test_points_group_by_the_object_they_read_from(self) -> None:
        points = parse_watch_behind("0x38:0x0-0x8:u8,0x40:0x0-0x8:u8")
        plans = build_read_plan(points)
        assert [p.slot for p in plans] == [SlotPath(0x38), SlotPath(0x40)]
        assert len(plans) == 2, "two objects = two reads per player per poll"

    def test_block_span_covers_the_last_points_full_width(self) -> None:
        points = [
            WatchPoint(name="a", offset=0x10, kind="u8"),
            WatchPoint(name="b", offset=0x20, kind="u32"),
        ]
        assert block_span(points) == (0x10, 0x14), "0x20+4 - 0x10; not 0x10"

    def test_block_span_of_nothing_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="zero watch points"):
            block_span([])

    def test_slicing_a_block_matches_reading_the_scalar_directly(self) -> None:
        block = bytes(range(32))
        point = WatchPoint(name="x", offset=0x14, kind="u32")
        assert slice_point(block, 0x10, point) == unpack_scalar(block[0x4:0x8], "u32")

    def test_bool8_folds_to_int_so_the_jsonl_stays_numeric(self) -> None:
        point = WatchPoint(name="b", offset=0, kind="bool8")
        assert slice_point(b"\x01", 0, point) == 1
        assert isinstance(slice_point(b"\x01", 0, point), int)

    def test_f32_stays_a_float(self) -> None:
        import struct

        point = WatchPoint(name="f", offset=0, kind="f32")
        assert slice_point(struct.pack("<f", 1.5), 0, point) == pytest.approx(1.5)

    def test_assemble_row_keeps_the_column_order_the_header_promised(self) -> None:
        # Interleave two slots so grouping-by-object necessarily reorders the reads.
        points = [
            WatchPoint(name="a0", offset=0, kind="u8", slot=SlotPath(0x38)),
            WatchPoint(name="b0", offset=0, kind="u8", slot=SlotPath(0x40)),
            WatchPoint(name="a1", offset=1, kind="u8", slot=SlotPath(0x38)),
        ]
        plans = build_read_plan(points)
        row = assemble_row(plans, [b"\xaa\xab", b"\xbb"], len(points))
        assert row == (0xAA, 0xBB, 0xAB), "a0, b0, a1 — not grouped order"

    def test_a_missing_block_is_an_error_not_a_zero_that_reads_as_data(self) -> None:
        plans = build_read_plan([WatchPoint(name="a", offset=0, kind="u8", slot=SlotPath(0x38))])
        with pytest.raises(ValueError):
            assemble_row(plans, [], 1)


class TestSweepBehindAPointer:
    """The end-to-end deref: a planted component, reached the way the decoder reaches transform."""

    def test_the_sweep_reads_the_object_behind_the_slot(self) -> None:
        player_base = 0x1_0000_0000
        component = HEAP_A.base
        source = FakeMemorySource(
            snapshots=[
                {
                    player_base + 0x38: component.to_bytes(8, "little"),
                    component + 0x0: bytes([6, 2, 0, 0]),
                }
            ],
            module_bases={"game.exe": 0x1000},
            advance_on=0,
        )
        (point,) = parse_watch_behind("0x38:0x0-0x1:u8")
        assert point.slot is not None
        landing = resolve_component(source, player_base, point.slot.to_component())
        assert landing == component, "the deref is resolve_component's, not a reimplementation"

        block = source.read(landing, 4)
        assert slice_point(block, 0, point) == 6

    def test_a_two_hop_chain_resolves_like_transform(self) -> None:
        player_base = 0x1_0000_0000
        outer, inner = HEAP_A.base, HEAP_B.base
        source = FakeMemorySource(
            snapshots=[
                {
                    player_base + 0x20: outer.to_bytes(8, "little"),
                    outer + 0x8: inner.to_bytes(8, "little"),
                    inner: bytes([9]),
                }
            ],
            module_bases={"game.exe": 0x1000},
            advance_on=0,
        )
        (point,) = parse_watch_behind("0x20/8:0x0:u8")
        assert point.slot is not None
        assert resolve_component(source, player_base, point.slot.to_component()) == inner


class TestPollRate:
    def test_the_rate_is_reported_against_10s_baseline(self) -> None:
        rate = PollRate(polls=100, elapsed=5.0)
        assert rate.hz == pytest.approx(20.0)
        assert "20.0 Hz" in rate.summary()
        assert "4.7 Hz" in rate.summary(), "the baseline this brief has to beat"

    def test_no_elapsed_time_is_zero_not_a_division_by_zero(self) -> None:
        assert PollRate().hz == 0.0

    def test_the_heartbeat_surfaces_the_rate_during_the_pass(self) -> None:
        # A sweep too slow to catch a tap must be visible while the user can still stop and retry.
        assert "@ 4.7 Hz" in heartbeat_line(1.0, 3, 5376, 4.7)
        assert "Hz" not in heartbeat_line(1.0, 3, 5376), "omitted when unknown"


class TestStageTwoRehearsal:
    """A dress rehearsal: plant a KNOWN encoding behind a slot and make the real analyzer find it.

    This is what makes a live *negative* worth reporting. If the Stage 2 pass comes back clean, the
    only honest conclusion is "input is not behind these slots" — but that conclusion is worthless
    if the plumbing (deref -> block read -> change record -> name -> analyzer) was quietly broken.
    So: synthesize the log the live sweep would produce if the field *were* there, at the user's
    known 1.15x tempo drift and a late start, and assert ``analyze-input`` recovers it above the
    floor. A failure here means the tool is broken; a clean live run against a passing rehearsal
    means the memory is.
    """

    SCALE = 1.15  # the user's measured drift: a human reading a checklist runs slow
    START = 3.7  # ...and starts the script after the probe. best_alignment must fit both back out.
    DIRS = {"u": 8, "d": 2, "b": 4, "f": 6, "u/f": 9, "d/f": 3, "d/b": 1, "u/b": 7}  # numpad
    BTNS = {"1": 1, "2": 2, "3": 4, "4": 8, "1+2": 3}  # a 1,2,3,4 bitmask

    def _log(self) -> list[str]:
        from tekken_coach.reader.input_probe import PROTOCOL, step_windows
        from tekken_coach.reader.probe import PollSample, change_records

        windows = step_windows(PROTOCOL, self.START, self.SCALE)

        def truth(t: float) -> tuple[int, int]:
            for w in windows:
                # Only a *hold* is a press; the rest windows between steps are hands-off neutral.
                if w.kind == "hold" and w.step is not None and w.t0 <= t < w.t1:
                    base = w.step.label.replace(" (again)", "")
                    return (self.DIRS.get(base, 5), self.BTNS.get(base, 0))
            return (5, 0)

        samples = []
        t = 0.0
        while t < 85.0:
            d, b = truth(t)
            # P2 is the standing dummy: never moves. Its stillness is the acting_only discriminator.
            samples.append(PollSample(t=t, rows=((d, b, 77), (5, 0, 77))))
            t += 0.05
        return [r.to_jsonl() for r in change_records(samples, self.NAMES)]

    # dir, buttons, and a decoy that holds a constant — the analyzer must reject the decoy.
    NAMES = ["@0x18+0x1c", "@0x18+0x1d", "@0x18+0x40"]

    def test_the_analyzer_recovers_a_planted_hit_behind_a_pointer_slot(self) -> None:
        from tekken_coach.reader.input_probe import (
            MIN_PLAUSIBLE,
            best_alignment,
            load_observation,
            rank_for_role,
        )

        obs = load_observation(self._log())
        start, scale = best_alignment(obs, acting_player=1)
        assert scale == pytest.approx(self.SCALE, abs=0.05), (
            "the tempo drift is fitted, not assumed"
        )
        assert start == pytest.approx(self.START, abs=1.0), "the late start is fitted too"

        for role, name, encoding in [
            ("dir", "@0x18+0x1c", self.DIRS),
            ("button", "@0x18+0x1d", self.BTNS),
        ]:
            ranked = rank_for_role(obs, role=role, start=start, scale=scale, acting_player=1)
            best = ranked[0]
            assert best.name == name, f"{role}: ranked the wrong offset"
            assert best.score > MIN_PLAUSIBLE, f"{role}: a real hit must clear the floor"
            # values_by_step is the deliverable: it answers the encoding question outright.
            for label, value in best.values_by_step.items():
                assert value == encoding.get(
                    label.replace(" (again)", ""), 5 if role == "dir" else 0
                )

    def test_the_constant_decoy_never_clears_the_floor(self) -> None:
        from tekken_coach.reader.input_probe import (
            MIN_PLAUSIBLE,
            best_alignment,
            load_observation,
            rank_for_role,
        )

        obs = load_observation(self._log())
        start, scale = best_alignment(obs, acting_player=1)
        for role in ("dir", "button"):
            ranked = rank_for_role(obs, role=role, start=start, scale=scale, acting_player=1)
            decoy = next(c for c in ranked if c.name == "@0x18+0x40")
            assert decoy.score < MIN_PLAUSIBLE, "a field that never reacts is not input"

    def test_the_behind_names_sort_in_address_order_not_lexically(self) -> None:
        from tekken_coach.reader.input_probe import _name_sort_key

        names = ["@0x18+0x1c", "@0x18+0x100", "@0x18+0x2", "@0x20/0x8+0x0"]
        assert sorted(names, key=_name_sort_key) == [
            "@0x18+0x2",
            "@0x18+0x1c",
            "@0x18+0x100",  # lexically this would sort before +0x1c
            "@0x20/0x8+0x0",
        ]
