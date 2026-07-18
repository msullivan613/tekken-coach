"""Validation-table builder for the moveset-datamine spike (#15).

Turns a set of cancels + ground-truth ``move_id → notation`` into a hit/miss table. Used by the
test on the synthetic fixture, and runnable on a REAL extract:

    python -m tests.spikes.moveset_datamine.validate bryan_cancels.json

where the JSON is ``{"neutral_move_id": <int>, "cancels": [{"source_move_id", "dest_move_id",
"command"}, ...], "input_sequences": {"<idx>": "<motion>"}, "alphabet": {"<code>": "<token>"}}``.
The alphabet is the calibrated direction map (solved from the known anchors). See AGENT-REPORT.md.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from .decoder import Cancel, DirectionAlphabet, join_moves


@dataclass(frozen=True)
class Row:
    move_id: int
    expected: str
    got: str | None
    status: str  # "HIT" | "MISS" | "COLLISION" | "UNRESOLVED"


def build_validation_table(
    cancels: list[Cancel],
    ground_truth: dict[int, str],
    *,
    neutral_move_id: int,
    alphabet: DirectionAlphabet,
    input_sequences: dict[int, str],
) -> list[Row]:
    result = join_moves(
        cancels,
        neutral_move_id=neutral_move_id,
        alphabet=alphabet,
        input_sequences=input_sequences,
    )
    rows: list[Row] = []
    for move_id in sorted(ground_truth):
        expected = ground_truth[move_id]
        if move_id in result.notation:
            got = result.notation[move_id]
            rows.append(Row(move_id, expected, got, "HIT" if got == expected else "MISS"))
        elif move_id in result.collisions:
            rows.append(Row(move_id, expected, "|".join(result.collisions[move_id]), "COLLISION"))
        else:
            rows.append(Row(move_id, expected, None, "UNRESOLVED"))
    return rows


def format_table(rows: list[Row]) -> str:
    hits = sum(1 for r in rows if r.status == "HIT")
    lines = [f"{'move_id':>8}  {'expected':<10}  {'got':<12}  status", "  " + "-" * 46]
    for r in rows:
        lines.append(f"{r.move_id:>8}  {r.expected:<10}  {str(r.got):<12}  {r.status}")
    lines.append("  " + "-" * 46)
    lines.append(f"  {hits}/{len(rows)} hits")
    return "\n".join(lines)


def _main(path: str) -> int:  # pragma: no cover - runs on a real extract, not in the gate
    with open(path) as fh:
        blob = json.load(fh)
    alphabet = DirectionAlphabet({int(k): v for k, v in blob.get("alphabet", {}).items()})
    input_sequences = {int(k): v for k, v in blob.get("input_sequences", {}).items()}
    cancels = [
        Cancel(c["source_move_id"], c["dest_move_id"], c["command"]) for c in blob["cancels"]
    ]
    rows = build_validation_table(
        cancels,
        {int(k): v for k, v in blob["ground_truth"].items()},
        neutral_move_id=blob["neutral_move_id"],
        alphabet=alphabet,
        input_sequences=input_sequences,
    )
    print(format_table(rows))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main(sys.argv[1]))
