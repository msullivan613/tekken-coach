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
field offsets within it, plus a `discovered_at` timestamp and the `update_memory_address.py`
run notes. The reader:

1. Detects the running game version (executable/product version, or a memory signature).
2. Loads the matching offset file; if none matches, refuses to attach and prints the
   re-discovery instructions (§4). It **does not guess** with a stale table — a wrong offset
   silently produces garbage `FrameRecord`s, which is worse than not running.

### Anchoring strategy
Prefer **module-base + static offset** and, where the ancestral tool uses them, **AOB/pattern
signatures** over absolute addresses, because signatures survive minor relocations better. The
ported `update_memory_address.py` is the tool that regenerates both.

## 4. Patch handling (the maintenance story)

A Season/balance patch can shift **both** offsets **and** move data (summary §7). The runbook:

1. Game updates → reader detects an unknown version → **fails closed** with instructions.
2. User opens **practice mode, P1 Jin vs P2 Kazuya**, runs the ported `update-offsets` command
   (our wrapper around `update_memory_address.py`). It writes a new `assets/offsets/<version>.json`.
3. **Invalidate move/frame data** for the new version (mirror the fork's "wipe frame_data"
   guidance) — see [05](05-frame-data-and-move-map.md) for the data side of the same patch event.
4. Re-run the reader's **self-check** (§6). Green → capture is usable again.

This is the primary ongoing maintenance sink and is expected to recur every Season/patch. It is
deliberately a **data + tooling** operation, not a source-code edit.

## 5. Port scope — what we take vs. leave

| Take (read-only) | Leave |
|---|---|
| `GameState`/`Entry.py` memory-layout knowledge | The bot / decision / input-injection layer |
| `update_memory_address.py` (as `update-offsets`) | Any GUI/overlay from the fork |
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

1. **Offsets, addresses, and AOB signatures are facts/data, not copyrightable expression** — we
   can use the *knowledge* of the T8 memory layout the fork surfaces (into our own
   `assets/offsets/`, §3) freely.
2. The **method** of `update_memory_address.py` (open Jin vs Kazuya in practice, scan to re-derive
   addresses) is an idea/technique, not protected — we **clean-room re-implement** it as
   `update-offsets` rather than copying the fork's script text.
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

## Sources
- TekkenBot Tekken 8 fork: <https://github.com/dcep93/TekkenBot>
- Ancestral TekkenBot (T7) and layout root: <https://github.com/WAZAAAAA0/TekkenBot>
