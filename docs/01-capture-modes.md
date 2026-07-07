# 01 — Capture Modes

> **Decision (summary §9):** both modes ship in v1. They feed the identical downstream pipeline
> and differ only in *when the reader is attached* and *how immediate* coaching is.

## 1. The two modes

| | **Live capture** | **Clean capture** |
|---|---|---|
| Reader attached during | the ranked match itself | offline replay playback only |
| Ranked session footprint | reader attached, silent, read-only | **zero** — nothing attached during ranked |
| Coaching fires | at match end | at end of the replay-review session |
| Feedback latency | immediate | delayed (each replay plays back in ~real time) |
| Anti-cheat profile | lower than a live-advice bot, but non-zero | pure-offline zero |

Both modes produce byte-identical `FrameRecord` streams ([03](03-data-schemas.md)); the segmenter
and everything downstream cannot tell which mode produced a log. The only differences are the
*trigger* that starts/stops capture and the *coaching cadence*.

## 2. Mode is user-configured, and defaults to clean

- The mode is a config setting and a CLI flag (`--mode live|clean`, [07](07-output-and-cli.md)).
- **Default is `clean`.** Rationale: the summary is explicit that the online line (§3.2) and
  anti-cheat concentration in online play (§4) are the sharp edges. The safer mode is the default;
  live is opt-in with eyes open. The reader never attaches to an online session unless the user
  has explicitly selected `live`.
- A first-run notice explains the trade-off before the user can pick `live`.

## 3. Live capture

### 3.1 Lifecycle
```
idle ──(user starts app in live mode)──► armed
armed ──(reader detects match-active game state)──► recording
recording ──(reader detects match-over game state)──► flush → COACH → armed
```
- "Armed" means attached to the process but only watching round/match state flags; it begins
  buffering `FrameRecord`s when a match goes active and stops when it ends.
- The **coaching trigger** is the match-over transition. Coaching for match *N* runs during the
  downtime before match *N+1* (post-match screens, rematch/queue). This is the downtime slot the
  latency constraint reserves (summary §3.1).
- Absolutely nothing is rendered during the match. Terminal output is written only after the
  match-over transition. This is a hard invariant, not a UI preference (summary §3.2).

### 3.2 Invariants
- Read-only: no memory writes, no input injection, ever ([02](02-memory-reader.md)).
- No mid-match surface of any kind (no overlay, no sound, no window change).
- If the reader loses the process or offsets go stale mid-match, it fails **silent and closed**:
  stop buffering, log an internal error, surface it only after the match ends.

## 4. Clean capture

Tekken 8 has **no replay-data file export** (summary §6), so "offline analysis" still means
reading memory — just from the game replaying a match rather than playing one live. The reader
attaches to the game during **replay playback in an offline mode**, which never touches an online
session.

### 4.1 Lifecycle
```
idle ──► user queues a set of saved replays to review
      ──► for each replay: play it back offline with reader attached → segment → append to log
      ──► at end of the batch: COACH over the whole session log
```

### 4.2 Replay selection — the Wavu replays API
Research surfaced a useful aid: `wank.wavu.wiki` exposes an `/api/replays` endpoint returning a
player's recent replays ordered by `battle_at` (this is *separate* from frame-data sources; see
[05](05-frame-data-and-move-map.md)). v1 does **not** depend on it, but it is the natural way to
later present "here are your last N ranked matches — pick which to review." For v1, replay
selection is manual (the user starts playback in-game; the tool detects playback-active state the
same way it detects match-active state). The API integration is a documented v1.x enhancement.

### 4.3 Playback detection
The reader distinguishes *live match* from *replay playback* by the game's mode/state flags in
memory (the same round/match-state fields it already reads — [03](03-data-schemas.md)). In clean
mode the reader **only** buffers when the state indicates offline replay playback, and refuses to
buffer if it detects an online match state (defense in depth against misconfiguration).

## 5. Shared requirements

- The segmenter, xref, session store, and coaching layer are **mode-agnostic**. No `if mode ==`
  branching past the reader/trigger layer.
- Each session log records its capture mode in the header record ([03](03-data-schemas.md) §5) so
  coaching can note delayed vs. immediate context, and so fixtures remember their origin.
- Player identity: the log must mark **which player is the user** (P1/P2). In live capture the
  user's side is known from config/practice; in clean capture from the replay's perspective
  metadata. Getting this wrong inverts all coaching, so it is validated (a mismatch between
  configured character and observed character on the user's side is a hard error).
