# 02 — Memory Reader

> Settles summary §9 #1: *which TekkenBot fork / memory-reading approach to build on, and its
> current Tekken 8 offset-maintenance story.*

## 1. Decision: build on `dcep93/TekkenBot`

Of the TekkenBot lineage, the live Tekken 8 branch is **[`dcep93/TekkenBot`](https://github.com/dcep93/TekkenBot)**
(a.k.a. "TekkenBot420"). The ancestral projects
([`WAZAAAAA0/TekkenBot`](https://github.com/WAZAAAAA0/TekkenBot) and its many forks) target
**Tekken 7**; their `TekkenGameState.py` memory layout is the shared root but the offsets are T7.

We adopt `dcep93/TekkenBot` as the **reference for memory layout and offset-maintenance
tooling**, and port a **read-only subset** of it into `tekken_coach.reader`. We do not fork it
wholesale — see §5.

### What it gives us
- **Python**, matching our stack decision.
- A working Tekken 8 memory-reading `GameState` equivalent (the `Entry.py`-documented fields).
- A **post-patch offset re-discovery script**, `update_memory_address.py`: open practice mode as
  **P1 Jin vs P2 Kazuya** and run it; it re-derives the new address locations automatically
  (community reports ~10 min). This is exactly the "auto-address-update script as prior art"
  the summary §7 anticipated.
- A GitHub-Actions build pipeline for redistributing after a patch.

### What it warns us about (important)
- The fork **stopped parsing the move list out of memory** and moved to a shipped database
  (`assets/database/frame_data`, supplemented by `assets/database/opponent_moves`), because the
  in-memory move list became unreliable/unavailable to parse. **We inherit this reality:** the
  move-ID → name map is a **maintained asset**, not something we can fully derive at runtime. This
  is why [05](05-frame-data-and-move-map.md) treats the move map as a first-class data-ingest
  problem, and it confirms the summary §7 cost-sink warning.
- Its guidance to **wipe the frame-data folder after a patch** to avoid stale data tells us patch
  handling must invalidate cached per-move data, not just offsets.

## 2. What the reader reads (and does not)

The reader resolves a base address for each player struct plus global match state, then reads a
fixed set of fields per frame. The concrete field list and types are the `FrameRecord` schema in
[03](03-data-schemas.md); this spec covers *how* they are obtained, not their shape.

Categories read:
- **Global/match:** frame counter, match state, round state/number, timer.
- **Per player (×2):** character ID, current move/animation ID, frame-in-move counter, a
  simple attack/recovery/neutral state, health, position (x/y/z) & facing, block/hit-stun flags,
  throw state, airborne/juggle state, Heat state (active + timer), Rage state, and the current
  **input** (buttons/direction) where available.

Three of those read differently than the naive layout suggests, and §3 explains how:
**health** is computed from `damage_taken`; the stun/throw/airborne **flags** are decoded from a few
encoded state words rather than read as booleans (§8); and **position** lives in a separate transform
component, not in the player struct.

**Read-only, always.** The reader opens the process for read access, reads, and closes. It never
calls a write primitive and never synthesizes input. The entire input/bot half of the ancestral
TekkenBot is **removed from the port, not merely unused** ([00](00-architecture.md) §7), so there
is no code path that could write to the game. This is the anti-cheat posture from summary §4/§5
expressed in code structure.

## 3. Offset model & the address table

Offsets are **not** hard-coded in source. They live as data under `assets/offsets/`, one file per
**game version**:

```
assets/offsets/
├── index.json               # game-version → offset-file, with a detected-version marker
├── 2.00.00.json
├── 2.01.01.json
└── ...
```

Each offset file is a versioned table: module base anchoring, the player-struct stride, and the
field offsets within it, plus a `discovered_at` timestamp and the `update-offsets` run notes. The
reader:

1. Detects the running game version (executable/product version, or a memory signature).
2. Loads the matching offset file; if none matches, refuses to attach and prints the
   re-discovery instructions (§4). It **does not guess** with a stale table — a wrong offset
   silently produces garbage `FrameRecord`s, which is worse than not running.

### Anchoring strategy
Prefer **module-base + static offset**, optionally followed by a **pointer chain**, and store an
**AOB/pattern signature** alongside it — never an absolute address, because signatures survive minor
relocations better. Our clean-room `update-offsets` (§4, §5) regenerates all three.

**Tekken 8 forces the pointer chain.** The per-player entity struct is **heap-allocated and
reallocates** on every character change and round (confirmed live: an address found by heap scanning
went stale after a character swap). No module-relative window reaches it. The reader therefore
anchors the player struct as:

```
module_base + base_offset   ->  deref +o1 -> deref +o2 -> ... ->  player-struct base
```

`base_offset` (the static slot holding the root pointer) shifts every build; the **chain offsets and
the within-struct field offsets are far more stable**. So `update-offsets --base-scan` re-derives
only `base_offset`, by scanning the module's static data for a slot whose chain lands on a struct
that satisfies the **known field layout** — a plausible `char_id`, a plausible `move_id`,
`damage_taken == 0` at round start, and P1/P2 resolving to Jin and Kazuya. That mutual two-struct
consistency is the acceptance test. Around the accepted slot it captures an **AOB signature** (the
pointer bytes wildcarded) and stores it in the table so a subsequent run re-finds the slot directly.

`decode.resolve_anchor` already follows a multi-level `pointer_path`, so the reader consumes such an
anchor with no code change; the chain is re-resolved on every read, which is precisely what makes it
survive reallocation.

**The global/match struct is behind its own static pointer too**, so it gets the same treatment —
sweep the static data, follow a seeded chain shape, validate the landing. What differs is the
**oracle**. The player struct has a *structural* signature (a plausible char id next to a plausible
move id next to a zero damage counter). A frame counter is just a `u32`; nothing about one instant
identifies it. So the global oracle is **behavioral**, read across the same two snapshots the
position scan uses:

| field | behavior across the two snapshots | required? |
|---|---|---|
| `frame_counter` | strictly increases, by a plausible delta | yes |
| `round` | holds constant, in 1..k | yes |
| `timer_ms` | strictly decreases (a round clock counts down) | no — practice mode freezes it |
| `match_phase` | reads as a small code | no |

Crucially, the within-struct offsets are seeded as an **unassigned list**: we know those offsets
carry match state, not which is which. The scan assigns offset → field *by behavior*, so a reordering
in the source data cannot silently mislabel the frame counter. `frame_counter` + `round` together are
the acceptance: a ticking `u32` is common in any game process, but a ticking `u32` beside a steady
round number at a known match-state offset is not.

The **chain shape** is likewise seeded as a *list of hypotheses*, tried longest-first. The layout
source records the frame counter as one long offset run and does not say where the pointer chain ends
and the field offsets begin. Listing several candidate splits is safe precisely because the oracle
rejects the wrong ones — a wrong chain dereferences into nothing, or lands somewhere with no ticking
counter beside a plausible round.

### Position is not in the player struct

A full-struct scan across a walking snapshot pair finds **no moving float triple** anywhere in the
entity struct (confirmed live), and the fork's layout data has no position field either. Position
lives in a separate Unreal **transform component**, reached by a pointer stored inside the entity
struct. `PlayerStruct` therefore carries an optional named `components` map, each a
`ComponentAnchor` resolved *relative to that player's base*:

```
address = deref(player_base + slot_offset)   # the component object
for o in pointer_path: address = deref(address + o)   # further hops (an Unreal actor -> scene component)
# fields are read at address + field.offset
```

`update-offsets` locates it the same way it locates everything else: the entity's own pointer slots
are the candidates (filtered to those holding the *same* pointer in both snapshots — a component does
not reallocate while the player takes a step), a moving `(x,y,z)` float triple inside the pointee is
the oracle, and **P2 resolving through the identical path to a different, plausible coordinate** is
the acceptance. That is the same two-struct mutual-consistency argument the player oracle rests on.

## 4. Patch handling (the maintenance story)

A Season/balance patch can shift **both** offsets **and** move data (summary §7). The runbook:

1. Game updates → reader detects an unknown version → **fails closed** with instructions.
2. User opens **practice mode, P1 Jin vs P2 Kazuya**, at **round start** (full health, no damage
   taken), and runs the clean-room `update-offsets --base-scan` command (§5 — a re-implementation of
   the fork's re-discovery *technique*, not its script). It re-derives the player anchor
   (`base_offset` + pointer chain + AOB signature) and the global/match anchor, and writes a
   candidate `assets/offsets/<version>.json` keyed to the detected exe version. The tool prompts the
   user to walk P1 and press a button between two snapshots; that single action does triple duty —
   the position delta locates the transform component, the frame-counter delta identifies the global
   struct, and an in-struct scan for round-start max HP locates `health` (or falls back to
   `damage_taken`). (`update-offsets` without `--base-scan` is the older heap value-scan; it cannot
   reach a struct behind a pointer chain.)
3. **Invalidate move/frame data** for the new version (mirror the fork's "wipe frame_data"
   guidance) — see [05](05-frame-data-and-move-map.md) for the data side of the same patch event.
4. **Verify the state map** (§8). It is *not* per-version and is normally carried across a patch
   untouched — but if the report says `NOT CALIBRATED`, or a state decodes wrong, run the §8
   observation protocol. This is a one-time cost at bring-up, not a per-patch one.
5. Re-run the reader's **self-check** (§6). Green → capture is usable again.

What the base scan proves vs. what still needs a human. It **derives**: the player anchor, the
stride, `char_id`, `move_id`, `health` (as `max_health - damage_taken`), `pos_{x,y,z}` (via the
transform component), the **global/match anchor**, and `frame_counter` / `round` (and `timer_ms`
where the clock is running). It **seeds** the encoded state-word offsets from the layout facts —
where `stun_type` lives is knowable, but no round-start oracle can prove it, because nobody is in
stun at round start. It leaves **carried from the previous table**: heat, rage, input, facing, the
`state_codes` maps, and `match_phase`'s offset. All of that is named in the diagnostic report, tiered
by confidence, so calibration is focused rather than a hunt.

Two things the tool refuses to guess. If P2 turns out **not** to sit at a constant stride from P1 (a
two-level `p2_data_offset`), it reports the P1 anchor and **writes no table** — expressing that needs
a per-player-anchor schema change, not an invented stride. And the **meanings** of the encoded state
values are never inferred: they come from the §8 observation protocol, and until they do the reader
decodes every state as `neutral` and says so loudly.

This is the primary ongoing maintenance sink and is expected to recur every Season/patch. It is
deliberately a **data + tooling** operation, not a source-code edit.

## 5. Port scope — what we take vs. leave

| Take (read-only) | Leave |
|---|---|
| `GameState`/`Entry.py` memory-layout knowledge | The bot / decision / input-injection layer |
| The field-offset map + chain shape from its `memory_address.ini` **data** | Its shipped `base_offset` (stale, per-build — we re-derive it) |
| The Jin-vs-Kazuya re-discovery *technique*, clean-room as `update-offsets` | The fork's `update_memory_address.py` script text (GUI/overlay too) |
| Process-attach + read primitives | The fork's real-time frame-data display |
| Offset table format (adapted to `assets/offsets/`) | Its match against our schema is re-mapped, not copied |

### Licensing (resolved — not legal advice)

The lineage splits into two legally distinct pieces:

- **`WAZAAAAA0/TekkenBot` root — MIT License, © 2017 roguelike2d.** MIT permits use, modification,
  and redistribution; it is irrevocable for the code it covered. Any code in `dcep93/TekkenBot`
  that is unchanged from or derived from this MIT root can be ported, **provided we retain the MIT
  copyright notice and permission text** (ship a `NOTICE`/`THIRD_PARTY_LICENSES` crediting
  roguelike2d, 2017).
- **`dcep93/TekkenBot` fork — no LICENSE file (confirmed 404).** Net-new code added in the fork is
  therefore **"all rights reserved" by default** — GitHub's ToS lets other users view and fork it
  on GitHub, but not copy it into our separately-distributed project. So we do **not** copy the
  fork's net-new *source*.

What this means concretely for the port:

1. **Offsets, addresses, pointer-chain shapes, and AOB signatures are facts/data, not copyrightable
   expression** — we can use the *knowledge* of the T8 memory layout the fork surfaces (into our own
   `assets/offsets/`, §3) freely. Concretely, the within-struct field map and chain shape seeded into
   `assets/offsets/probe-manifest.json` (`base_scan`) come from the fork's committed
   `memory_address.ini` **data file**, credited in `NOTICE`.
2. The **method** of `update_memory_address.py` (open Jin vs Kazuya in practice, scan to re-derive
   addresses) is an idea/technique, not protected — we **clean-room re-implement** it as
   `update-offsets` rather than copying the fork's script text. Its script is deliberately **not
   read**: our candidate-generate-and-validate base scan (sweep static data slots, follow the seed
   chain, accept on the field-layout oracle) is written from the technique's description, not ported.
3. **MIT-root code** we port directly, with attribution.
4. We do **not** pursue a license grant from the fork author. The repo has been **inactive for
   ~2 years**, so an issue/PR is unlikely to be answered — and, more importantly, we don't need
   one: rules 1–3 already give us everything required (MIT-root code + T8 layout facts + a
   clean-room `update-offsets`).

This is engineering-practical guidance, not legal advice; if distribution terms matter to you,
have counsel confirm before shipping.

### The fork is a frozen reference, not a live dependency

Because `dcep93/TekkenBot` has been inactive for ~2 years, treat it as a **point-in-time reference
snapshot**, not a maintained upstream:

- Its shipped **T8 offsets are probably already stale** for current Season patches, and its
  `assets/database` move data likewise. Expect the very first `update-offsets` run (§4) and a fresh
  `fetch-framedata` ([05](05-frame-data-and-move-map.md)) to be **required at bring-up**, not just
  after the *next* patch.
- **We own offset maintenance from day one.** The summary §7 framed this as an ongoing burden; the
  fork's inactivity means there is no upstream absorbing patches for us even initially. This is
  already the design — §3 keeps offsets as our own versioned `assets/offsets/` data, and §4 is our
  own runbook — but it should be understood as day-one work, not a later cost.
- The value we take from the fork is therefore the **layout knowledge and the re-discovery
  technique** (both durable across its inactivity), not any expectation of current addresses.

## 6. Reader self-check (sanity gate)

Before any capture session, and after every offset update, run a deterministic self-check that
attaches during **practice mode** and asserts:
- both character IDs resolve to known characters,
- health reads a plausible max at round start,
- the frame counter monotonically increases,
- a known move (e.g. a jab) produces a stable, non-garbage move ID,
- positions/distance change when the practice dummy is moved.

A failed self-check blocks capture and points at §4. This is the fast signal that a patch broke
the offsets, rather than discovering it as corrupt interactions three matches later.

## 7. Reliability & failure modes

| Failure | Detection | Response |
|---|---|---|
| Unknown game version | version lookup miss | fail closed, print §4 runbook |
| Stale offsets (garbage reads) | self-check §6 | block capture |
| Process not found / closed mid-capture | attach/read error | live: fail silent-closed, surface after match ([01](01-capture-modes.md) §3.2); clean: stop batch, report |
| Dropped frames (poll slower than 60fps) | frame-counter gaps | record gap markers in the stream; segmenter tolerates small gaps ([04](04-segmenter.md)) |
| Anti-cheat / access denied | attach error | report clearly; do not retry-hammer |

## 8. State-map calibration (the one thing no scan can prove)

Our `PlayerFrame` ([03](03-data-schemas.md) §1) asks for booleans — `block_stun`, `hit_stun`,
`airborne`, `juggle`, … — because that is what the segmenter reasons with ([04](04-segmenter.md)
§4.1 distinguishes blockstun from hitstun from stagger *by the specific flag*). **Tekken 8 does not
store them.** It stores a handful of **encoded state words** — `simple_move_state`, `stun_type`,
`complex_move_state`, `throw_tech_state`, `recovery_state` — whose integer values each denote a
whole situation.

So the reader needs two separate pieces of knowledge, and they have very different provenance:

1. **Where** the state words live. Offsets — facts/data (§5), seeded into the probe manifest,
   written into the offset table by `update-offsets`.
2. **What their values mean.** `stun_type == 3` is *stagger* only if someone observed it. No
   two-snapshot oracle at the Jin-vs-Kazuya round-start setup can derive this, because at round start
   nobody is in stun, staggered, thrown, or airborne — the setup deliberately excludes every state we
   want to name.

Piece 2 lives in `assets/offsets/state-map.json`: a `field → raw value → [semantic flags]` map,
validated at load (an unknown flag name is rejected, not silently dropped at 60fps). The decoder
reads every mapped word, **unions** the flags across fields, and folds the union into the
`PlayerFrame`. Unioning is what lets overlapping axes compose — `stun_type=hit_stun` together with
`complex_move_state=airborne+juggle` is a juggle combo — without the map enumerating the product.

It is kept **separate from the per-version offset tables** on purpose: `update-offsets` rewrites the
*addresses* every build, but the value semantics are calibrated once and carried forward.

### It ships uncalibrated, and says so

The checked-in map is an empty skeleton (`"calibrated": false`). An unmapped raw value contributes no
flags, so as shipped the reader **runs** but every `action_state` decodes to `neutral` and every stun
flag is false: structurally valid, semantically empty. `update-offsets` prints
`encoded state map: NOT CALIBRATED` and the calibration runbook. This is deliberate — a *guessed*
map would produce a reader that looks like it works and mislabels every interaction, which is exactly
the failure mode §3 refuses stale offsets to avoid.

### The observation protocol

The raw words ride out on `PlayerFrame.raw_state` ([03](03-data-schemas.md) §1), so this is a matter
of watching values change while you cause each state:

```
python -m tekken_coach.reader.commands probe-state
```

It streams a line whenever any watched word changes, for both players. In practice mode (P1 you, P2
the dummy), perform each state in turn and record the value:

| do this | expect a distinct value in | maps to |
|---|---|---|
| stand still | `simple_move_state` | `neutral` |
| throw a jab | `simple_move_state` | `attack`, then `recovery` |
| hold down-back | `simple_move_state` | `crouch` |
| set the dummy to attack; **block** it | `stun_type` | `block_stun` |
| set the dummy to attack; **get hit** | `stun_type` | `hit_stun` |
| take a stagger-on-block move | `stun_type` | `stagger` |
| jump | `complex_move_state` | `airborne` |
| get juggled | `complex_move_state` | `airborne` + `juggle` |
| get knocked down; then wake up | `complex_move_state` | `knockdown`, `wakeup` |
| sidestep | `complex_move_state` | `sidestep` |
| throw the dummy / get thrown / break a throw | `throw_tech_state` | `throw_active` / `thrown` / `throw_tech` |

Write the observed integers into `state-map.json`, set `"calibrated": true`, and re-run `doctor`.
A `capture`d fixture is the cross-check: its `raw_state` shows exactly which values are still
unmapped.

> **Licensing note — an open choice, deliberately not made here.** The fork's source presumably
> contains these enum values. Under §5 rule 1 the *values* would be facts/data, like the offsets — but
> we have not read them, and the clean-room posture we took for `update-offsets` (§5 rule 2) argues
> for deriving the map by observation instead, which the protocol above does at the cost of ~15
> minutes in practice mode. **We ship the observation path and no fork-derived values.** If a
> reviewer prefers to seed the map from the fork's enums as facts, that is a defensible reading of
> rule 1 — but it should be an explicit decision recorded here, not something that happened silently.

## Sources
- TekkenBot Tekken 8 fork: <https://github.com/dcep93/TekkenBot>
- Ancestral TekkenBot (T7) and layout root: <https://github.com/WAZAAAAA0/TekkenBot>
