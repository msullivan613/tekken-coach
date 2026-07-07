# 03 — Data Schemas

> Settles summary §9 #3: *concrete schema for the per-frame state record and the segmented
> interaction record.* This spec defines the three record types that flow through the pipeline
> and the on-disk session log format that is the contract seam ([00](00-architecture.md) §3).

All records are JSON-serializable. Python-side they are `@dataclass`es (or pydantic models);
on disk they are JSON. IDs use lowercase snake_case. Times are integer **game frames** unless a
field name ends in `_ms`.

---

## 1. `FrameRecord` — one per game frame (reader → segmenter)

The raw, uninterpreted state of one frame. Produced by [02](02-memory-reader.md), consumed by
[04](04-segmenter.md). Kept deliberately flat and cheap; ~thousands per match.

```jsonc
{
  "frame": 128472,            // int: game global frame counter (monotonic within a match)
  "match_state": "in_round",  // enum: pre_round | in_round | round_over | match_over | replay | menu
  "round": 2,                 // int: 1-based round number
  "timer_ms": 41200,          // int: round clock remaining, ms
  "players": [ <PlayerFrame>, <PlayerFrame> ]  // exactly 2; index 0 = P1, index 1 = P2
}
```

### `PlayerFrame`

```jsonc
{
  "char_id": 12,             // int: character ID (→ name via move map, 05)
  "move_id": 2145,           // int: current move/animation ID (→ name via move map, 05)
  "move_frame": 7,           // int: frames elapsed within the current move (0 = just started)
  "action_state": "attack",  // enum: neutral | attack | recovery | blockstun | hitstun
                             //       | stagger | throw_tech_window | thrown | airborne
                             //       | knockdown | wakeup | sidestep | crouch
  "health": 142,             // int: current HP
  "pos": [1.42, 0.0, -0.31], // [x,y,z] floats (game units)
  "facing": 1,               // int: +1 faces right, -1 faces left (P-relative sign for distance)
  "block_stun": false,       // bool: in block recovery this frame
  "hit_stun": false,         // bool: in hit recovery this frame
  "counter_state": "none",   // enum: none | counter_hit | punish_counter  (defender's hit type)
  "throw_active": false,     // bool: executing/attempting a throw
  "airborne": false,         // bool: feet off ground (juggle-eligible)
  "juggle": false,           // bool: in an active juggle combo
  "heat": {                  // Heat system state
    "active": true,
    "timer_ms": 3100,
    "engager_used": true     // has this player spent their Heat engager this round
  },
  "rage": true,              // bool: Rage available
  "input": {                 // may be null if inputs are not resolvable this frame
    "dir": 6,                // numpad notation direction (1-9), 5 = neutral
    "buttons": ["2"]         // pressed attack buttons this frame: subset of 1,2,3,4 (+ combos)
  }
}
```

**Notes**
- `action_state` is a *thin* normalization the reader can derive cheaply from memory flags. The
  segmenter does the real interpretation; it does not trust `action_state` alone (see
  [04](04-segmenter.md) §4 on why raw flags around stagger/tech are ambiguous).
- `move_frame` is essential: it distinguishes "a new move started" from "same move, next frame,"
  which is how the segmenter detects move boundaries without name lookups.
- `input` may be `null` in clean/replay capture if inputs aren't exposed during playback; the
  segmenter must not require it (it enriches labeling, e.g. "user pressed jab", but interactions
  are derivable from state transitions alone).
- Distance is **not stored** per frame (derivable from `pos`); the segmenter computes it.

---

## 2. `Interaction` — one per segmented exchange (segmenter → xref)

A discrete, bounded exchange with an attacker, a defender, an outcome, and follow-up. Produced by
[04](04-segmenter.md). This is the "unglamorous middle layer" the summary §6 calls the real core.

```jsonc
{
  "id": "m3-r2-i017",        // stable id: match-round-interaction
  "match_id": "2026-07-07T20:14:03Z#3",
  "round": 2,
  "start_frame": 128410,
  "end_frame": 128498,
  "attacker": 1,             // player index who initiated (1 = P2 here)
  "defender": 0,             // the other player
  "attacker_move_id": 2145,
  "context": {
    "distance": 1.6,         // float at interaction start
    "attacker_heat": true,
    "defender_heat": false,
    "attacker_pressure": true, // attacker already had frame advantage entering (from prior interaction)
    "wall": "none",          // none | near | splat  (position context)
    "defender_health_frac": 0.71
  },
  "defender_reaction": "blocked",  // see enum below
  "observed_advantage": -13,       // int frames: measured advantage to attacker on this contact
                                   //   (negative = attacker disadvantaged / punishable); null if N/A
  "outcome": "no_punish",          // see enum below
  "follow_up": {                   // what the defender did in their action window after reaction
    "move_id": null,               // move the defender did, or null for nothing. The authoritative "did nothing" signal is result=="none"; consumers key on result, not move_id (a reader may also report 0 for "no move" — treat 0 and null as equivalent).
    "result": "none",              // none | whiffed | hit | blocked | got_counter_hit | traded
    "reaction_frames": null        // frames until defender acted, if they acted
  },
  "notes": []                      // segmenter diagnostics (e.g. "gap-tolerated:2 dropped frames")
}
```

### `defender_reaction` enum
`blocked` · `hit` · `counter_hit` · `whiff_punished` (defender blocked/evaded then hit back) ·
`evaded` (sidestep/backdash made it whiff) · `parried` · `thrown` · `throw_broke` ·
`traded` · `interrupted` (defender's own move beat it)

### `outcome` enum (from the *user's* coaching perspective, filled after we know which side is the user)
`no_punish` (punishable, defender did nothing) · `punished` · `bad_punish` (punished but suboptimal) ·
`respected_true` (respected a real gap — correct) · `respected_false` (respected a fake gap — could have acted) ·
`challenged_true` (mashed into a true string — got hit) · `challenged_false` (correctly challenged a gap) ·
`ate_low` · `ate_mid` · `mashed_into_ch` · `neutral` (nothing coachable)

> The `outcome` here is the **segmenter's best structural guess**; the *authoritative*
> punishability/gap judgments come from the frame-data xref (§3), which may confirm or correct it.

---

## 3. `LabeledInteraction` — xref output (xref → session log)

An `Interaction` plus ground-truth annotations from [05](05-frame-data-and-move-map.md). Same
shape as `Interaction` with an added `labels` block and resolved human-readable names:

```jsonc
{
  ...allInteractionFields,
  "attacker_move_name": "df+2",
  "attacker_char_name": "Kazuya",
  "defender_char_name": "Jin",
  "labels": {
    "frame_data_matched": true,      // did we resolve this move in the frame-data table
    "on_block": -13,                 // ground-truth on-block advantage for the move
    "was_punishable": true,          // on_block ≤ the defender's fastest punisher startup
    "punish_window": 3,              // frames of slack (fastest punisher startup − |on_block|... see 05)
    "correct_punish": "f,F+2 (i15)", // recommended punish for defender's character at this range
    "user_punished_correctly": false,
    "in_string": false,              // was this contact part of a multi-hit string
    "string_gap": null,              // {duckable|interruptible|true|null} for string situations
    "gap_size": null,                // frames of the gap, if any
    "duckable_high_hit": null,       // index of a high in this string the user blocked STANDING that is duck-punishable (a missed duck-punish); null otherwise (05 §4.1)
    "duck_punish": null,             // recommended punish after ducking that high, e.g. "df+1 (i13)"
    "move_property": "mid",          // high | mid | low | throw | unblockable
    "is_knowledge_check": true,      // did this trip a rubric pattern (06)
    "knowledge_check_ids": ["punish_missed"]  // which rubric pattern(s), see 06
  }
}
```

**Nullable-vs-required in `labels`.** The **frame-data-derived** fields — `on_block`,
`was_punishable`, `punish_window`, `correct_punish`, `user_punished_correctly`, `move_property`,
`string_gap`, `gap_size`, `duckable_high_hit`, `duck_punish` — are **null when
`frame_data_matched` is false** (the miss-tolerant path, [05](05-frame-data-and-move-map.md)
§2.3/§4.1); they carry values only on a match. The **structural/rubric** fields are always present:
`frame_data_matched`, `in_string`, `is_knowledge_check` (bool), and `knowledge_check_ids` (list,
possibly empty). Rule of thumb: *if it depends on resolving the move in frame data, it is nullable.*

**Xref is a pure function.** `LabeledInteraction = f(Interaction, FrameDataTable, MoveMap, Rubric)`.
No memory access, no LLM. It is fully unit-testable against fixture interactions ([04](04-segmenter.md) §7).

---

## 4. Aggregate: `KnowledgeCheckTally` (optional, coaching-side)

Not persisted by the pipeline; computed by the coaching layer or a pre-pass to rank recurring
issues. Included here so the shape is agreed. Counts `LabeledInteraction`s grouped by
`(knowledge_check_id, attacker_char, attacker_move_id, matchup)` with occurrence counts and
example interaction IDs. This is what turns "you missed this once" into "you missed this 6 times
this session" — the recurrence that makes something a *knowledge check* (summary §1). See
[06](06-coaching-skill.md).

---

## 5. Session event log — the on-disk contract (`.jsonl`)

One session = one JSON Lines file. First line is a **header record**; every subsequent line is one
`LabeledInteraction`. Append-only; flushed at round end ([00](00-architecture.md) §4).

```
sessions/2026-07-07T20-14-03.jsonl
```

**Header (line 1):**
```jsonc
{
  "record": "session_header",
  "schema_version": "1.0.0",
  "created_at": "2026-07-07T20:14:03Z",
  "capture_mode": "clean",         // live | clean  (01)
  "game_version": "2.01.01",       // ties log to the offset/frame-data snapshot used
  "framedata_snapshot": "2026-06-30",
  "user_player": 0,                // which player index is the user (01 §5) — coaching pivots on this
  "user_char": "Jin",
  "matches": [                      // may be empty at open; finalized on close (see note)
    {"match_id": "...#3", "opponent_char": "Kazuya", "result": "loss", "rounds": 3}
  ]
}
```

> **Header write model (append-only reconciliation).** The header is written **once at open** as
> line 1, and the body is strictly **append-only** — the header is never rewritten per match. Since
> every `LabeledInteraction` carries its `match_id`, the `matches` list is a **denormalized
> convenience** derivable from the body. Two allowed strategies, both preserving line-1-is-header:
> (a) leave `matches` as written at open (possibly empty) and let consumers derive the match list
> from the body; or (b) **rewrite line 1 once on session close** to finalize `matches`. The default
> is (b). Per-match in-place header rewrites are **not** done. Header finalization is a session-
> orchestration concern ([07](07-output-and-cli.md)); the writer may expose a `finalize(matches)`
> hook for it. `MatchSummary.result` is intentionally an open string until the reader establishes
> the real value set (`win`/`loss`/disconnect/…); do not invent an enum prematurely.

**Body (lines 2..N):** one `LabeledInteraction` per line (§3).

**Why JSONL:** append-friendly (round-end flush without rewriting), streamable by the coaching
layer, trivially diffable, and readable by both the Claude Code Skill (reads the file) and the API
backend (streams it) with no schema negotiation. `schema_version` gates compatibility; the coaching
layer refuses logs with an unknown major version.

---

## 6. Schema versioning & compatibility

- `schema_version` is semver. **Major** bump = breaking field change; **minor** = additive.
- The reader and segmenter emit the current version. **Compatibility is gated on major only:** a
  consumer accepts any log whose **major** matches its own, regardless of minor. Within a matching
  major, minor differences are non-breaking in **both** directions — a newer log's additive fields
  are ignored on load, and an older log simply omits newer (optional) fields. A **major** mismatch
  is rejected. (An earlier draft said "minor ≤ its own"; that was wrong — it contradicts additive
  tolerance — and is corrected here.)
- `game_version` + `framedata_snapshot` in the header make every log **reproducible**: you can
  re-run xref/coaching against the exact data set that produced it, which matters across Season
  patches ([05](05-frame-data-and-move-map.md)).
