# 07 — Output & CLI

> Settles summary §9 #6. **Decision:** terminal output in v1; a desktop viewer is the end goal but
> **v2** work. This spec covers the v1 command-line surface and how coaching is presented between
> matches.

## 1. v1 surface: a terminal application

One Python CLI, `tekken-coach`, orchestrates the pipeline ([00](00-architecture.md)) and renders
coaching to the terminal. No overlay, no window, no desktop app in v1.

### 1.1 Commands
```
tekken-coach live                 # live capture (01): arm, record per match, coach at match end
tekken-coach clean [replays...]   # clean capture (01): play back replays offline, coach at session end
tekken-coach coach <session.jsonl># re-run coaching on an existing event log (no capture)
tekken-coach update-offsets       # post-patch offset re-discovery (02 §4)
tekken-coach fetch-framedata       # ingest a new frame-data snapshot (05 §3.3)
tekken-coach doctor                # reader self-check + data-freshness report (02 §6, 05 §5)
```

### 1.2 Key flags
| Flag | Applies to | Meaning |
|---|---|---|
| `--mode live\|clean` | capture | overrides the configured default (default `clean`, [01](01-capture-modes.md) §2) |
| `--coach skill\|api` | capture, `coach` | which backend ([06](06-coaching-skill.md)); default `skill` |
| `--user p1\|p2` | capture | which player is the user ([01](01-capture-modes.md) §5) |
| `--char <name>` | capture | the user's character, validated against reads |
| `--out <path>` | capture | where to write the session `.jsonl` (default `sessions/<timestamp>.jsonl`) |

Config file (`~/.config/tekken-coach/config.toml`) holds durable defaults (mode, coach backend,
user side/char, paths) so the common case is a bare `tekken-coach live` / `clean`.

## 2. The default (Skill) flow vs. the API flow

Because the default coaching backend is a Claude Code Skill ([06](06-coaching-skill.md)), the two
flows differ only at the last step:

**`--coach skill` (default):** capture runs, writes `sessions/<ts>.jsonl`, and prints:
```
✔ Session recorded: sessions/2026-07-07T20-14-03.jsonl  (3 matches, 128 interactions)
→ Coach it in Claude Code:  open this repo and run the tekken-coach skill on that file.
```
The terminal shows *that a log is ready*; the coaching prose is produced in Claude Code on the
user's subscription (zero marginal cost). This keeps v1 fully functional with no API key.

**`--coach api`:** capture runs, then the app calls the Claude API itself and prints the coaching
report directly to the terminal (§3). Requires the user's API credential ([06](06-coaching-skill.md) §3).

## 3. Terminal coaching report

When coaching is rendered in-terminal (`--coach api`, or `tekken-coach coach` against a log), it
follows `references/output-format.md` ([06](06-coaching-skill.md) §5): a short, ranked,
between-matches report. Rendering guidance:

- Plain, readable text; light use of color/box-drawing for the section headers and the ranked
  list. Degrade gracefully when stdout is not a TTY (no ANSI) so logs pipe cleanly.
- **Timing respects the capture mode** ([01](01-capture-modes.md)): in live mode the report prints
  in the post-match downtime; nothing is ever printed mid-match (hard invariant, summary §3.2). In
  clean mode the report prints at end-of-session after the replay batch.
- Keep it scannable in the seconds between matches — the report is deliberately short.

## 4. Errors & operational output
- Reader/offset problems surface here with the §02 runbook (unknown game version → run
  `update-offsets`; failed self-check → `doctor`).
- Data-freshness warnings (stale frame-data snapshot, rising move-map miss rate,
  [05](05-frame-data-and-move-map.md) §5/§6) print as non-fatal notices; capture continues.
- In **live** mode, capture-time errors are held and shown only after the match ends
  ([01](01-capture-modes.md) §3.2).

## 5. v2: the desktop viewer (out of scope, noted for direction)
The end goal is a desktop application to browse sessions, replay interaction timelines, and read
coaching in a richer surface. It is **v2** and deliberately not built now. It is cheap to add later
because:
- the session `.jsonl` files are a stable, self-describing archive
  ([03](03-data-schemas.md) §5/§6) the viewer can read directly;
- the coaching layer already has a file-based contract, so the viewer can invoke either backend;
- nothing in v1 assumes a terminal is the *only* consumer — the terminal renderer is one view over
  the event log, not the source of truth.

v1 should therefore avoid baking presentation logic into the pipeline: the pipeline emits data
([03](03-data-schemas.md)); the CLI is just the first renderer of it.
