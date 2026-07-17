"""The pure observation core behind ``probe-state`` (docs/02 §8 state-map calibration).

The live half is a ``while True`` loop reading a live process (``# pragma: no cover``). Everything
worth testing was lifted out of it into :mod:`tekken_coach.reader.probe`: the change-record stream
(the ``--record`` JSONL) and the draft-skeleton builder (the ``--emit-skeleton`` output). These
exercise that core against a scripted :class:`~tekken_coach.reader.memory_source.FakeMemorySource`
and against hand-built samples, and prove the emitted skeleton loads back via ``load_state_map``.

Clean-room note (docs/02 §5 rule 2): every raw value here is an arbitrary stand-in the *test*
supplies; no flag meaning is sourced from any community enum. The skeleton the tool emits carries
empty flag lists — a human fills them.
"""

from __future__ import annotations

import json
from pathlib import Path

from tekken_coach.reader.decode import decode_frame
from tekken_coach.reader.memory_source import FakeMemorySource
from tekken_coach.reader.offsets import EncodedStateSpec, load_state_map
from tekken_coach.reader.probe import (
    WIDE_SWEEP_COLUMNS,
    ChangeRecord,
    PollSample,
    build_skeleton,
    change_records,
    distinct_values,
    due_for_beat,
    heartbeat_line,
    is_wide_sweep,
    parse_watch,
)
from tekken_coach.schemas import ActionState

# The watched set probe-state computes: context fields then the encoded-state words (sorted).
NAMES = [
    "move_id",
    "move_frame",
    "counter_state",
    "complex_move_state",
    "recovery_state",
    "simple_move_state",
    "stun_type",
    "throw_tech_state",
]
ENCODED = [
    "complex_move_state",
    "recovery_state",
    "simple_move_state",
    "stun_type",
    "throw_tech_state",
]


def _sample(t: float, p1: tuple[int | float, ...], p2: tuple[int | float, ...]) -> PollSample:
    return PollSample(t=t, rows=(p1, p2))


def test_change_records_emit_only_on_change_per_player() -> None:
    # P1 changes at t=0 (first sight) and t=2; P2 changes at t=0 and t=1. A tuple that holds across
    # a poll produces no row — a performed-and-held state reads as one event, not a flood.
    a = (100, 0, 0, 0, 0, 0, 0, 0)
    b = (100, 0, 0, 0, 0, 1, 0, 0)  # P1's simple_move_state -> 1
    p2a = (200, 0, 0, 0, 0, 0, 0, 0)
    p2b = (200, 0, 0, 0, 0, 0, 3, 0)  # P2's stun_type -> 3
    samples = [
        _sample(0.0, a, p2a),
        _sample(1.0, a, p2b),  # only P2 changed
        _sample(2.0, b, p2b),  # only P1 changed
        _sample(3.0, b, p2b),  # neither changed -> no rows
    ]
    records = list(change_records(samples, NAMES))

    # First poll: both players are "new" -> one row each. Then one row per subsequent change.
    assert [(r.t, r.player) for r in records] == [
        (0.0, 1),
        (0.0, 2),
        (1.0, 2),
        (2.0, 1),
    ]
    # Player tags are 1-based, times are monotonic, and fields map name -> value in order.
    assert all(r.player in (1, 2) for r in records)
    assert [r.t for r in records] == sorted(r.t for r in records)
    assert records[-1].fields == dict(zip(NAMES, b, strict=True))


def test_change_records_from_a_fake_memory_source() -> None:
    # Drive the change stream from a real FakeMemorySource read loop, proving the pure core composes
    # with the seam the live loop actually uses (the offsets/addresses are arbitrary here).
    base = 0x1000
    offsets = {name: base + 8 * i for i, name in enumerate(NAMES)}
    advance = 0x900  # a distinct "frame counter" address that ticks the fake forward

    def image(values: tuple[int, ...]) -> dict[int, bytes]:
        img = {
            addr: v.to_bytes(4, "little") for addr, v in zip(offsets.values(), values, strict=True)
        }
        img[advance] = (0).to_bytes(4, "little")
        return img

    v0 = (7, 0, 0, 0, 0, 0, 0, 0)
    v1 = (7, 0, 0, 0, 0, 2, 0, 0)
    source = FakeMemorySource([image(v0), image(v1)], module_bases={"m": 0}, advance_on=advance)

    def read_row(t: float) -> PollSample:
        source.read(advance, 4)  # tick to the next snapshot, mirroring a live poll
        row = tuple(int.from_bytes(source.read(offsets[n], 4), "little") for n in NAMES)
        return PollSample(t=t, rows=(row, row))

    samples = [read_row(0.0), read_row(1.0)]
    records = list(change_records(samples, NAMES))
    # v0 then v1: each player emits at t=0 (new) and t=1 (simple_move_state changed) -> 4 rows.
    assert [(r.t, r.player, r.fields["simple_move_state"]) for r in records] == [
        (0.0, 1, 0),
        (0.0, 2, 0),
        (1.0, 1, 2),
        (1.0, 2, 2),
    ]


def test_to_jsonl_round_trips_and_rounds_time() -> None:
    record = ChangeRecord(t=12.4449, player=1, fields={"move_id": 133, "stun_type": 3})
    obj = json.loads(record.to_jsonl())
    assert obj == {"t": 12.44, "player": 1, "fields": {"move_id": 133, "stun_type": 3}}


def test_distinct_values_lists_exactly_the_observed_values_per_encoded_field() -> None:
    records = [
        ChangeRecord(
            t=0.0, player=1, fields={"move_id": 1, "simple_move_state": 0, "stun_type": 0}
        ),
        ChangeRecord(
            t=1.0, player=1, fields={"move_id": 9, "simple_move_state": 1, "stun_type": 0}
        ),
        ChangeRecord(
            t=2.0, player=2, fields={"move_id": 9, "simple_move_state": 5, "stun_type": 3}
        ),
    ]
    result = distinct_values(records, ENCODED)
    # Only encoded fields appear; move_id (context) is watched for correlation, not skeletonized.
    assert set(result) == set(ENCODED)
    assert result["simple_move_state"] == {"0": [], "1": [], "5": []}
    assert result["stun_type"] == {"0": [], "3": []}
    # A field never seen contributes an empty map (valid, just nothing observed for it).
    assert result["throw_tech_state"] == {}
    # Every value maps to an empty flag list — the tool emits values, a human emits meanings.
    assert all(flags == [] for codes in result.values() for flags in codes.values())


def test_skeleton_loads_through_load_state_map_uncalibrated(tmp_path: Path) -> None:
    records = [
        ChangeRecord(t=0.0, player=1, fields={"simple_move_state": 0, "stun_type": 0}),
        ChangeRecord(t=1.0, player=2, fields={"simple_move_state": 2, "stun_type": 3}),
    ]
    skeleton = build_skeleton(records, ENCODED)
    assert skeleton["calibrated"] is False
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(skeleton), encoding="utf-8")

    spec = load_state_map(path)
    assert isinstance(spec, EncodedStateSpec)
    assert spec.calibrated is False
    assert spec.flags["simple_move_state"] == {"0": [], "2": []}
    assert spec.flags["stun_type"] == {"0": [], "3": []}


def test_parse_watch_parses_hex_and_decimal_pairs() -> None:
    from tekken_coach.reader.probe import parse_watch

    points = parse_watch("0x434:u32, 0x670:u32 , 1360:i32")
    assert [(p.name, p.offset, p.kind) for p in points] == [
        ("@0x434", 0x434, "u32"),
        ("@0x670", 0x670, "u32"),
        ("@0x550", 1360, "i32"),  # decimal offset, named by its hex
    ]


def test_parse_watch_expands_a_range_stepped_by_kind_width() -> None:
    from tekken_coach.reader.probe import parse_watch

    # A START-END:KIND term sweeps [START, END) stepped by the kind's byte width — for locating an
    # unknown field (e.g. match_phase) in a struct region.
    pts = parse_watch("0x10-0x20:u32")
    assert [(p.name, p.offset) for p in pts] == [
        ("@0x10", 0x10),
        ("@0x14", 0x14),
        ("@0x18", 0x18),
        ("@0x1c", 0x1C),
    ]  # END is exclusive, stride 4 for u32
    # u16 steps by 2; a single value and a range compose in one spec.
    pts2 = parse_watch("0x8:u16, 0x40-0x44:u16")
    assert [p.offset for p in pts2] == [0x8, 0x40, 0x42]


def test_parse_watch_rejects_a_backwards_or_absurd_range() -> None:
    import pytest

    from tekken_coach.reader.probe import parse_watch

    with pytest.raises(ValueError, match="END must be greater"):
        parse_watch("0x20-0x10:u32")
    with pytest.raises(ValueError, match="expands to"):
        parse_watch("0x0-0x100000:u8")  # 1 MiB of u8 points -> refused


def test_parse_watch_rejects_malformed_specs() -> None:
    import pytest

    from tekken_coach.reader.probe import parse_watch

    for bad, needle in [
        ("0x434", "OFFSET:KIND"),  # no colon
        ("0xzz:u32", "not a valid offset"),  # bad number
        ("0x434:u33", "unknown kind"),  # bad kind
        ("-4:u32", "START-END"),  # leading '-' reads as a (malformed) range, and is rejected
        ("", "empty"),  # nothing to watch
        ("  ,  ", "empty"),  # only separators
    ]:
        with pytest.raises(ValueError, match=needle):
            parse_watch(bad)


def test_watch_points_flow_through_change_records_with_float_values() -> None:
    # A watch run may target an f32 candidate (e.g. to confirm an offset is a float, not a counter);
    # the pipeline must carry the float faithfully, not truncate it to int.
    from tekken_coach.reader.probe import build_skeleton

    records = list(
        change_records(
            [
                _sample(0.0, (0, 1.5), (0, 1.5)),
                _sample(1.0, (0, 2.5), (0, 1.5)),  # only P1's float column changed
            ],
            ["@0x550", "@0x370"],
        )
    )
    assert [(r.t, r.player, r.fields["@0x370"]) for r in records] == [
        (0.0, 1, 1.5),
        (0.0, 2, 1.5),
        (1.0, 1, 2.5),
    ]
    skel = build_skeleton(records, ["@0x370"])
    assert skel["flags"] == {"@0x370": {"1.5": [], "2.5": []}}  # floats keyed as strings


def test_a_hand_filled_skeleton_value_reaches_player_frame(tmp_path: Path) -> None:
    # Round-trip proof the calibrated path is wired: fill ONE value with a real flag by hand (as the
    # owner would post-landing), load it into the offset table, decode a frame, and assert the flag
    # surfaces on PlayerFrame. This ships no filled map — the edit lives only in the test.
    from tests.test_reader_decode_encoded import _encoded_table, _source

    records = [ChangeRecord(t=0.0, player=1, fields={"stun_type": 2})]
    flags = distinct_values(records, ["stun_type"])
    flags["stun_type"]["2"] = ["hit_stun"]  # the human's judgment, applied by hand
    path = tmp_path / "filled.json"
    path.write_text(json.dumps({"calibrated": True, "flags": flags}), encoding="utf-8")

    spec = load_state_map(path)
    table = _encoded_table()
    filled = table.model_copy(
        update={"state_codes": table.state_codes.model_copy(update={"encoded_state": spec})}
    )
    p1 = decode_frame(_source(filled, p1_state={"stun_type": 2}), filled).players[0]
    assert p1.hit_stun is True
    assert p1.action_state is ActionState.hitstun


def test_an_unannotated_skeleton_field_stays_neutral(tmp_path: Path) -> None:
    # Guardrail: a skeleton with observed values but no human annotation stays calibrated:false and
    # contributes no flags — every state decodes to neutral (valid structure, empty semantics).
    from tests.test_reader_decode_encoded import _encoded_table, _source

    records = [ChangeRecord(t=0.0, player=1, fields={"stun_type": 2})]
    skeleton = build_skeleton(records, ["stun_type"])
    path = tmp_path / "unannotated.json"
    path.write_text(json.dumps(skeleton), encoding="utf-8")

    spec = load_state_map(path)
    assert spec.calibrated is False
    table = _encoded_table()
    unannotated = table.model_copy(
        update={"state_codes": table.state_codes.model_copy(update={"encoded_state": spec})}
    )
    p1 = decode_frame(_source(unannotated, p1_state={"stun_type": 2}), unannotated).players[0]
    assert p1.action_state is ActionState.neutral
    assert p1.hit_stun is False


# --- wide-sweep console output (brief #10 follow-up) ---------------------------------------------


def test_is_wide_sweep_trips_past_the_column_budget() -> None:
    # A whole-struct sweep prints ~20 chars per column per change: at 5376 offsets that is a ~100 KB
    # console line ~20x/s, which renders slower than the game runs and wrecks the pass being
    # recorded. Past the budget the probe must switch to a heartbeat.
    assert not is_wide_sweep([f"@0x{i:x}" for i in range(WIDE_SWEEP_COLUMNS)])
    assert is_wide_sweep([f"@0x{i:x}" for i in range(WIDE_SWEEP_COLUMNS + 1)])
    # The sweep this exists for: the whole known player struct, byte by byte.
    assert is_wide_sweep([p.name for p in parse_watch("0x100-0x1600:u8")])


def test_due_for_beat_fires_first_then_rate_limits() -> None:
    assert due_for_beat(None, 0.0)  # the first change always beats, so the user sees it is alive
    assert not due_for_beat(4.0, 4.5, every=1.0)
    assert due_for_beat(4.0, 5.0, every=1.0)


def test_heartbeat_line_shows_the_clock_the_checklist_is_written_in() -> None:
    line = heartbeat_line(12.5, changes=480, points=5376)
    assert line.strip().startswith("12.50")  # the probe's t == the input-protocol checklist's t
    assert "5376 offsets" in line
    assert "480 changes" in line
