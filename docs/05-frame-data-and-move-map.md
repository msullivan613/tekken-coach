# 05 — Frame Data & Move Map

> Settles summary §9 #5: *source of truth and update process for the move-ID map and frame data
> (ingest cadence, handling of Season patches).* This is the primary maintenance sink the summary
> §7 flags. This spec covers two distinct-but-linked assets and the cross-reference that consumes
> them.

## 1. Two assets, two different problems

| Asset | Question it answers | Source of truth | Volatility |
|---|---|---|---|
| **Move map** | move_id 2145 = "df+2" for Kazuya | game-derived + community DBs | shifts on patches; **can't be fully derived at runtime** |
| **Frame data** | "df+2" is −13 on block, mid, i15 | community sites (Wavu / TekkenDocs) | shifts on balance patches |

They are separate because the move **ID→name** binding is a property of the game build (memory),
while the **name→properties** binding is a property of the balance patch (community data). A patch
can change either or both ([02](02-memory-reader.md) §4).

## 2. The move map (`assets/movemap/`)

### 2.1 Why it can't be purely runtime-derived
The Tekken 8 TekkenBot fork we build on ([02](02-memory-reader.md)) **stopped parsing the move
list from memory** and shifted to a shipped database, because the in-memory move list became
unreliable to parse. We inherit that: the move map is a **maintained asset**, seeded from
community data and the fork's `assets/database/`, not something we can regenerate from a clean
memory read alone.

### 2.2 Structure
One file per character, keyed by move ID:
```
assets/movemap/
├── index.json          // char_id → char name → movemap file, + game_version stamp
├── kazuya.json
├── jin.json
└── ...
```
```jsonc
// kazuya.json
{
  "char_id": 12,
  "char_name": "Kazuya",
  "game_version": "2.01.01",
  "moves": {
    "2145": { "notation": "df+2", "aliases": ["down-forward 2"], "framedata_key": "df+2" }
  }
}
```
`framedata_key` is the join key into the frame-data table (§3) and is kept explicit because
notation strings and frame-data table keys don't always match character-to-character.

### 2.3 Seeding & maintenance
- **Seed** from the fork's `assets/database/frame_data` + `opponent_moves` and from community
  notation lists, per character.
- **Coverage is scoped** (summary §8): only the user's played matchups need a *complete* map for
  v1. Unmapped IDs are not fatal — they resolve to `move_id:<n>` and xref marks
  `frame_data_matched:false`, so an unknown move degrades to "unlabeled interaction," not a crash.
- The map is the labor-intensive grind (summary §7); §5 below defines the cadence that keeps it
  from silently rotting.

## 3. The frame-data table (`assets/framedata/`)

### 3.1 Sources of truth
Two community sources, both offering machine-readable data:
- **Wavu Wiki** (`wavu.wiki`) — community frame-data wiki; pages expose JSON via a `?_format=json`
  request. Primary source.
- **TekkenDocs** (`tekkendocs.com`) — serves frame data as JSON (e.g. `/api/.../framedata`
  patterns). Secondary / cross-check.

> **Important distinction found in research:** `wank.wavu.wiki` is a *different* service — it
> serves **player ratings and a `/api/replays` endpoint**, *not* move frame data. It is relevant
> to clean-capture replay selection ([01](01-capture-modes.md) §4.2), **not** to this asset. Don't
> conflate the two Wavu hosts.

Exact endpoint shapes must be re-verified at implementation time (community sites change); treat
the URLs above as starting points, not a stable API contract.

### 3.2 Structure — a versioned snapshot
Frame data is ingested into a **local snapshot**, not fetched live (reproducibility, offline
operation, and rate-limit friendliness — Wavu asks that clients keep ≤1 request in flight):
```
assets/framedata/
├── snapshot-2026-06-30/
│   ├── manifest.json     // source URLs, fetch date, game/patch version, per-char checksums
│   ├── kazuya.json
│   └── ...
└── current -> snapshot-2026-06-30/
```
```jsonc
// per move
{
  "key": "df+2",
  "startup": 15,
  "on_block": -13,
  "on_hit": 45,          // or a launch marker
  "on_ch": "+launch",
  "block_frames": -13,
  "hit_level": "mid",     // high | mid | low | throw | unblockable
  "properties": ["homing", "heat_engager"],
  "recovery": 30,
  "notes": "…",
  "heat": { "on_block": -8 }   // Heat-state overrides where the move differs in Heat (04 §4.6)
}
```

### 3.3 Ingest tooling
A `fetch-framedata` command:
1. Pulls each scoped character from the primary source (respecting rate limits: serial requests).
2. Normalizes into the schema above; records source URL + checksum in `manifest.json`.
3. Diffs against `current`; prints what changed (surfaces balance-patch deltas for review).
4. Writes a new `snapshot-<date>/` and, on approval, repoints `current`.
Ingestion is **manual-triggered**, not automatic, so a source outage or format change can't
silently corrupt the table.

## 4. The cross-reference (`tekken_coach.framedata`)

Consumes `Interaction`s ([03](03-data-schemas.md) §2) + move map + frame-data snapshot + rubric,
produces `LabeledInteraction`s ([03](03-data-schemas.md) §3). Pure function, no I/O beyond the
loaded snapshot ([04](04-segmenter.md) §5, [00](00-architecture.md) §3).

### 4.1 Core computations
- **Resolve names:** `attacker_move_id` → `framedata_key` → move record. Miss ⇒
  `frame_data_matched:false`, minimal labels, `is_knowledge_check:false` (can't judge what we
  can't identify).
- **Punishability:** `was_punishable = on_block ≤ −(defender_fastest_punisher_startup)`, using the
  **defender character's** fastest relevant punisher at that range/stance. `punish_window`,
  `correct_punish`, and `user_punished_correctly` follow. This is where the segmenter's *observed*
  advantage is cross-checked against the *canonical* `on_block` (see §4.2).
- **String gaps:** for in-string interactions ([04](04-segmenter.md) §4.2), compute the frame gap
  between hits from per-hit data → `string_gap ∈ {true|interruptible|duckable|null}`, `gap_size`.
- **Heat selection:** if the interaction was in Heat, use the move's `heat` overrides.
- **Knowledge-check tagging:** run the rubric patterns ([06](06-coaching-skill.md)) and set
  `is_knowledge_check` / `knowledge_check_ids`.

### 4.2 Observed vs. canonical reconciliation
The segmenter gives `observed_advantage`; the table gives canonical `on_block`. Rules:
- If they agree (within tolerance) → high confidence; use canonical for labels.
- If they disagree → prefer canonical for the *answer* (punish recommendation) but keep
  `observed_advantage` in the record and add a note. Persistent disagreement across many
  interactions is a **signal the frame-data snapshot is stale** (post-patch) — a maintenance alarm
  (§5), not a per-interaction error.
- If `observed_advantage` is null (dropped frames) → rely on canonical only.

## 5. Patch-handling cadence (ties §02 and §05 together)

A Season/balance patch is a single event with **two** data consequences and one code-free
response:

```
game patch drops
   ├─ memory offsets may shift ──► run update-offsets (02 §4) → new assets/offsets/<ver>.json
   ├─ move IDs may shift       ──► re-seed / re-verify assets/movemap for scoped chars
   └─ frame data may shift     ──► run fetch-framedata → new snapshot-<date>/ → review diff → repoint `current`
Then: bump header stamps (game_version, framedata_snapshot) so new logs are reproducible (03 §6).
```

**Cadence**
- **On every patch** (event-driven): the reader fails closed on an unknown version
  ([02](02-memory-reader.md)), which is the forcing function to run the three steps above.
- **Between patches** (drift guard): the §4.2 observed-vs-canonical alarm and the move-map
  miss-rate (`frame_data_matched:false` frequency) are monitored; a rising miss rate means the map
  needs re-seeding even without a version bump.
- **Snapshots are immutable and dated**; `current` is a pointer. Old logs remain reproducible
  against the snapshot they were captured with.

## 6. Failure & degradation posture
- Unknown move ID → unlabeled interaction, not a crash (§2.3).
- Missing frame-data record → `frame_data_matched:false`; interaction still logged.
- Stale snapshot suspected → drift alarm surfaced to the user, capture continues.
- The pipeline **never blocks capture** on a data-quality problem; it degrades and flags.

## Sources
- Wavu Wiki (frame data): <https://wavu.wiki>
- TekkenDocs (frame data JSON): <https://tekkendocs.com>
- Wavu Wank (ratings/replays API — *not* frame data): <https://wank.wavu.wiki/api>
- Fork whose DB approach we inherit: <https://github.com/dcep93/TekkenBot>
