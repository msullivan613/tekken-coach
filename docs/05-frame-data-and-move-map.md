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
**Primary (git-pinned CSV):**
- **`pbruvoll/tekkendocs`** — the source repo behind `tekkendocs.com`. Its
  `data/wavuConvertedCsv/<char>/*.csv` files are **Wavu-derived frame data, already normalized to
  CSV, in version control**. This is our primary ingest source because it is:
  - **machine-readable and stable** — plain CSV, no scraping, no rate limits;
  - **reproducible** — we **pin a commit SHA**, which is a stricter, better snapshot key than a
    fetch date (§3.2);
  - **rich enough for our checks** — the `Hit level` column carries **per-hit string levels** as
    comma-separated values (e.g. Bryan `1,2,1` → `h, h, m`), which maps directly onto the string
    `hits[]` sequence (§3.2) and feeds `standing_duckable_high` ([06](06-coaching-skill.md) §4.1).
    Columns observed: `Command; Hit level; Damage; Start up frame; Block frame; Hit frame;
    Counter hit frame; Notes; Tags; Transitions; Name; Recovery; …; Wavu id; Character id`.
  - **Licensing (data vs code):** the repo's **code is restrictively licensed** (do **not** copy or
    vendor the app code), but the **data may be used with attribution**. Attribute
    **tekkendocs.com and rbnorway.org** in `NOTICE`/`THIRD_PARTY_LICENSES`. Same split as the reader
    ([02](02-memory-reader.md) §5): we take the *data*, not the code.
  - **Consistency caveat:** coverage/field completeness varies across characters (blank cells,
    occasional missing per-hit startup). Our loader is already miss-tolerant
    ([§2.3](#22-structure), §4.1) and we cross-check against the live sources + okizeme below.

**Cross-check / fallback (live — Wavu Cargo API):**
- **Wavu Wiki** (`wavu.wiki`) is the upstream authority the CSVs derive from. It is a **MediaWiki +
  Cargo** install queried structurally at **`https://wavu.wiki/w/api.php`** with
  `action=cargoquery&format=json` against the **`Move`** table. The importer we're mirroring
  (`pbruvoll/tekkendocs` `utils/wavu-importer/.../wavu_reader.py`) requests fields:
  `id, name, input, alias, alt, num, parent, image, video, target, damage, reach, tracksLeft,
  tracksRight, startup, recv, tot, crush, block, hit, ch, notes`, scoped `WHERE id LIKE '<Char>%'`,
  `limit 500`, ordered by `id`.
  - **How per-hit string levels arise:** hits are stored as separate `Move` rows linked by
    `parent`/`num`; the importer concatenates a string's hits and joins their **`target`** (hit
    level) into the comma list that becomes the CSV `Hit level` column (`h, h, m`). Same data as the
    CSV — the CSV just has this reconstruction pre-done. `block`/`hit`/`ch` need HTML-tag stripping
    and pipe-delimited normalization (the importer does this too).
  - **Use it for:** reconciling suspicious CSV values, and as a **direct-ingest fallback** if the
    CSV repo goes stale. If we query it, set a **descriptive `User-Agent`** and keep requests
    **serial** (the importer sets neither, but MediaWiki etiquette expects both).
- **TekkenDocs** (`tekkendocs.com`) — the rendered site, for quick human spot-checks.

> **Important distinction found in research:** `wank.wavu.wiki` is a *different* service — it
> serves **player ratings and a `/api/replays` endpoint**, *not* move frame data. It is relevant
> to clean-capture replay selection ([01](01-capture-modes.md) §4.2), **not** to this asset. Don't
> conflate the two Wavu hosts.

**Reference-only (not an ingest source):**
- **okizeme.gg** (`okizeme.gg`) — an excellent Tekken 8 data platform and the best *human*
  cross-check, especially for **per-hit string levels and duck/punish info** (exactly what the
  `standing_duckable_high` check needs, [06](06-coaching-skill.md) §4.1). But it is **client-rendered
  with no documented public API** — the page fetches data from an internal endpoint. Programmatic use
  would require reverse-engineering that endpoint or scraping the rendered DOM, which is a
  ToS/stability risk and not a stable contract; so we use it to **verify/curate the snapshot by hand**,
  not as an automated ingest. Revisit if okizeme ever publishes an API.
- Community API `theneosloth/tekken-api` is **Tekken 7 only and unlicensed** — not usable for T8.

Exact endpoint shapes must be re-verified at implementation time (community sites change); treat
the URLs above as starting points, not a stable API contract.

### 3.2 Structure — a versioned snapshot
Frame data is ingested into a **local snapshot**, not fetched live (reproducibility, offline
operation, and rate-limit friendliness — Wavu asks that clients keep ≤1 request in flight):
```
assets/framedata/
├── snapshot-2026-06-30/
│   ├── manifest.json     // pinned source commit SHA (pbruvoll/tekkendocs), game/patch version,
│   │                     //   per-char checksums, attribution (tekkendocs.com, rbnorway.org)
│   ├── kazuya.json
│   └── ...
└── current -> snapshot-2026-06-30/
```
```jsonc
// per move
{
  "key": "df+2",
  "startup": 15,          // parsed lower-bound int; the raw token (e.g. "i15~16") is kept in startup_raw
  "on_block": -13,        // canonical on-block advantage — the CSV "Block frame" column
  "on_hit": "+45",        // stored as a raw string: carries launch/knockdown markers, e.g. "+32a (+24)"
  "on_ch": "+launch",     // raw string, same reason (a clean leading int is still parseable)
  "hit_level": "mid",     // best-effort MoveProperty (high|mid|low|throw|unblockable); raw token in hit_level_raw
  "properties": ["homing", "heat_engager"],
  "recovery": 30,
  "notes": "…",
  "heat": { "on_block": -8 }   // Heat-state overrides where the move differs in Heat (04 §4.6)
}
```

**Strings** additionally carry a per-hit sequence and, where a mid-string high can be ducked for a
punish, a `duck_punish` marker:
```jsonc
// per string, e.g. Paul "df+1,1,2" (mid → high → mid)
{
  "key": "df+1,1,2",
  "startup": 14,
  "hits": [
    { "hit_level": "mid",  "startup": 14 },
    { "hit_level": "high", "startup": null },   // the CSV occasionally omits a per-hit startup
    { "hit_level": "mid",  "startup": 22 }
  ],
  "duck_punish": { "after_hit": 2, "answer": "df+1 (i13)" }  // duck the high (hit 2), punish before hit 3
}
```
`hits[].hit_level` is populated directly from the primary CSV's `Hit level` column, split on
commas (§3.1); `hits[].startup` comes from the per-hit `Start up frame` column and is what §4.1
uses to compute string gaps (so a string carries per-hit startup, not just per-hit level).
`duck_punish.answer` is **not** in the CSV — it is derived (a high mid-string that
whiffs on crouch, leaving a punish window) and hand-curated against okizeme.gg (§3.1) for the
scoped matchups; absent ⇒ no `standing_duckable_high` flag, which is a safe miss ([§4.1](#41-core-computations)).

**Normalization notes (how the loader shapes the CSV into the above).**
- **`hit_level` is a best-effort enum with a raw passthrough.** The CSV vocabulary is richer than
  the five-value set (`m!`, `sm`, `sl`, `sp`, `th(h)`, `*`/`!` power/break markers, case variants).
  Recognizable heights map to `MoveProperty`; anything without a clean height mapping is `null`, and
  the exact source token is always preserved in `hit_level_raw` (per-hit: `hits[].hit_level_raw`).
  `hit_level_raw` is authoritative for edge tokens; the enum mappings can be refined during curation.
- **Frame cells that carry annotations keep a `*_raw`.** `startup`/`startup_raw`,
  `on_block`/`block_raw`, `recovery`/`recovery_raw`: the parsed leading integer is exposed where it
  is clean, and the raw cell (`"i15~16"`, `"-13~-8"`, `"r31"`, a stance code) is preserved beside it.
  `on_hit`/`on_ch` are stored **only** as raw strings because they routinely carry launch markers.
- **`on_block` is the single block-advantage field** (sourced from the CSV `Block frame` column);
  there is no separate `block_frames` — an earlier draft listed both with identical values, which was
  redundant and is removed here. `labels.on_block` ([03](03-data-schemas.md) §3) reads this field.
- **`current` is a symlink where the filesystem supports one** (`current -> snapshot-<date>/`), with
  a **plain-text pointer file** (contents = the snapshot dir name) as the fallback on filesystems
  without symlinks (notably Windows without privilege); the loader and `promote` step read both forms.

### 3.3 Ingest tooling
A `fetch-framedata` command:
1. Fetches the scoped characters' `data/wavuConvertedCsv/<char>/*.csv` from `pbruvoll/tekkendocs`
   **at a pinned commit SHA** (raw file fetch or a shallow checkout of that ref) — no scraping, no
   rate-limit dance.
2. Parses the CSV and **normalizes into the schema above** (§3.2): splits the `Hit level` column
   into the string `hits[]` sequence, maps block/hit/CH frames, carries `Notes`/`Tags`. Records the
   **pinned SHA + attribution + per-char checksums** in `manifest.json`.
3. Diffs against `current`; prints what changed (surfaces balance-patch deltas for review).
4. Writes a new `snapshot-<date>/` and, on approval, repoints `current`.
Ingestion is **manual-triggered**, not automatic, and pinned to a specific commit, so an upstream
edit or format change can't silently corrupt the table — a snapshot is only adopted after the diff
is reviewed. The live Wavu/TekkenDocs sources and okizeme.gg (§3.1) are used to reconcile
suspicious values and to hand-curate `duck_punish` answers, not as the bulk feed.

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
  (This is a *timing* property — distinct from the *height* check below.)
- **Duckable highs:** cross-reference per-hit `hit_level` (§3.2 string `hits`) against the
  segmenter's per-hit block/duck record ([04](04-segmenter.md) §4.2). If the user **blocked a high
  standing** that the frame data marks `duck_punish`-able, set `labels.duckable_high_hit` (the hit
  index) and `labels.duck_punish` (the answer). If the user ducked it (the high whiffed), no flag —
  that's the correct play, not a knowledge check. Feeds `standing_duckable_high` ([06](06-coaching-skill.md) §4.1).
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
- Primary CSV data (git-pinned, Wavu-derived): <https://github.com/pbruvoll/tekkendocs> —
  `data/wavuConvertedCsv/<char>/*.csv`. Data usable with attribution (tekkendocs.com, rbnorway.org);
  repo code is restrictively licensed — do not vendor it.
- Wavu Wiki (frame data): <https://wavu.wiki> — Cargo API: <https://wavu.wiki/w/api.php> (`action=cargoquery`, `Move` table)
- Wavu importer reference (query shape + string reconstruction): `pbruvoll/tekkendocs` `utils/wavu-importer/src/wavu/wavu_reader.py`
- TekkenDocs (frame data JSON): <https://tekkendocs.com>
- Wavu Wank (ratings/replays API — *not* frame data): <https://wank.wavu.wiki/api>
- okizeme.gg (manual cross-check reference; no public API): <https://okizeme.gg>
- Fork whose DB approach we inherit: <https://github.com/dcep93/TekkenBot>
