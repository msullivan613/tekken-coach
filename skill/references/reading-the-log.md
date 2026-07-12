# Reading the event log

The session log is a **JSON Lines** file (`.jsonl`): one JSON object per line.

- **Line 1** is the `session_header` (`"record": "session_header"`).
- **Every line after** is one `LabeledInteraction` — a segmented, frame-data-labeled
  exchange.

Parse it line by line. The full schemas live in
`src/tekken_coach/schemas.py` (`SessionHeader`, `LabeledInteraction`, `Labels`);
this file is the field guide for coaching.

## The header (line 1)

```jsonc
{
  "record": "session_header",
  "schema_version": "1.2.0",
  "capture_mode": "clean",       // "live" = immediate, "clean" = delayed/replay
  "user_player": 0,              // 0 = P1, 1 = P2 — THE PIVOT: who the user is
  "user_char": "Kazuya",
  "matches": [                    // may be empty at open; finalized on close
    {"match_id": "...#1", "opponent_char": "Paul", "result": "loss", "rounds": 3}
  ]
}
```

- **`user_player` is the pivot for everything.** Coaching is about *the user*. In
  most interactions the user is the **defender** — the one blocking, failing to
  punish, or eating a mix — while the opponent is the `attacker`. Read every
  interaction from the user's side.
- `user_char`, `matches[].opponent_char`, and `matches[].result` give you the
  header line of the report. If `matches` is empty, derive the matchup and result
  from the interaction stream (each interaction carries `attacker_char_name` /
  `defender_char_name` and its `match_id`).
- `capture_mode` → "immediate" (`live`) vs "delayed" (`clean`) in the report header.

## Each interaction (lines 2..N)

The fields that matter for coaching:

```jsonc
{
  "id": "m1-r2-i017",            // stable id — CITE this as your example
  "match_id": "...#1",
  "round": 2,
  "attacker": 1, "defender": 0, // player indices; compare to header.user_player
  "attacker_move_id": 2145,
  "attacker_move_name": "df+2",  // human-readable, already resolved
  "attacker_char_name": "Kazuya",
  "defender_char_name": "Paul",
  "defender_reaction": "blocked",
  "follow_up": { "result": "none" },   // what the user did after: none | whiffed |
                                       //   hit | blocked | got_counter_hit | traded
  "labels": { ... }              // the ground truth — see below
}
```

### `labels` — the ground truth (already computed)

This is the frame-data xref's output. **Do not recompute any of it.**

- `is_knowledge_check` (bool) and `knowledge_check_ids` (list of strings) — whether
  this interaction tripped a rubric pattern, and which. **This is your primary
  filter:** the interactions to coach are the ones with `is_knowledge_check: true`.
  See `rubric.md` for what each id means.
- `frame_data_matched` — if `false`, the move wasn't in the frame-data table and the
  rest of the labels are `null`. These never trip a knowledge check; skip them.
- `on_block` — the move's on-block advantage (negative = punishable).
- `was_punishable`, `punish_window`, `correct_punish` — the punish ground truth.
  `correct_punish` is the exact input to recommend for `punish_missed`.
- `in_string`, `string_gap` (`duckable`/`interruptible`/`true`), `gap_size` — string
  timing, for `respected_fake_gap` / `challenged_true_string`.
- `duckable_high_hit`, `duck_punish` — for `standing_duckable_high`: the hit to duck
  and the punish after ducking it.
- `move_property` (`high`/`mid`/`low`/`throw`/`unblockable`) — for `ate_low`/`ate_mid`.

### Aggregating

Walk the stream, keep only `is_knowledge_check: true` interactions, and group them
by `(knowledge_check_id, attacker_char_name, attacker_move_id, matchup)` where
matchup is `"<attacker> vs <defender>"`. Count each group and remember one `id` per
group as your example. The count is what promotes a fluke to a coachable knowledge
check (see `rubric.md`).

## The shared assets (move-map + frame data)

The move map and frame-data snapshot live in the repo's **`assets/`** directory,
sibling to this `skill/` bundle — i.e. `../assets/` relative to `skill/`, or
`<repo>/assets/`:

- `assets/movemap/` — `move_id` → notation → `framedata_key`, per character.
- `assets/framedata/current/` — the frame-data snapshot (on-block, hit levels,
  string gaps, curated punishers).
- `assets/punishers/` — curated per-character punisher profiles.

You **do not need** these to write the report — `attacker_move_name`,
`correct_punish`, `duck_punish`, and the frame numbers are already resolved into
each interaction's `labels`. Consult `assets/` only to enrich phrasing (e.g. to
name a punish the log didn't spell out, or to describe a string's later hits). Both
this Skill and the `--coach api` backend resolve the same `assets/` directory, so
the two paths stay consistent. (There is deliberately **no** `skill/assets` symlink
— it doesn't survive this project's Windows/WSL checkout; the path above is the
contract instead.)
