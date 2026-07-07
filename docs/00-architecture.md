# 00 — Architecture

## 1. One-paragraph description

A Python side-car process reads Tekken 8's game state out of process memory, one record per
frame. A **segmenter** turns that raw frame stream into discrete, labeled **interactions**
(e.g. "opponent threw df+2, user blocked at −13, user did nothing"). A **frame-data
cross-reference** annotates each interaction against ground truth (was it punishable, was the
gap real, what was the correct answer). The annotated interactions for a match are written to a
**session event-log file**. Between matches, a **coaching layer** — by default a Claude Code
Skill, optionally the Claude API — reads that log plus the matchup rubric and prints coaching to
the terminal.

## 2. Pipeline diagram

```
                        ┌─────────────────────────── capture mode selects source ──────────────────────────┐
                        │                                                                                    │
  Tekken 8 process  ──► │  [live: attached during ranked]        [clean: attached during replay playback]   │
  (game memory)         │                                                                                    │
                        └────────────────────────────────────────┬───────────────────────────────────────────┘
                                                                  │
                                                                  ▼
   ┌──────────┐   frame records   ┌────────────┐   interactions   ┌──────────────────┐   labeled events
   │  READER  │ ────────────────► │ SEGMENTER  │ ───────────────► │ FRAME-DATA XREF  │ ─────────────────┐
   │ (02)     │  (schema §03)     │ (04)       │  (schema §03)    │ (05)             │                  │
   └──────────┘                   └────────────┘                  └──────────────────┘                  │
        ▲                                                                                                │
        │ move-ID map + memory offsets (05)                                                              ▼
        │                                                                                    ┌────────────────────┐
   ┌────┴───────────────┐                                                                    │  SESSION EVENT LOG │
   │ offset re-discovery│                                                                    │  (.jsonl file, 03) │
   │ after patches (02) │                                                                    └─────────┬──────────┘
   └────────────────────┘                                                                              │
                                                                                                       ▼
                                                          ┌──────────────────────────────────────────────────────┐
                                                          │  COACHING LAYER (06)                                  │
                                                          │  default: Claude Code Skill (rides subscription)     │
                                                          │  optional: Claude API backend (pay-per-token)        │
                                                          │  rubric + move-map + frame-data as skill resources   │
                                                          └────────────────────────────┬─────────────────────────┘
                                                                                        │
                                                                                        ▼
                                                                            ┌──────────────────────┐
                                                                            │  TERMINAL OUTPUT (07)│
                                                                            │  (desktop app = v2)  │
                                                                            └──────────────────────┘
```

## 3. Module boundaries

| Module | Package (proposed) | Responsibility | Never does |
|--------|--------------------|----------------|------------|
| Reader | `tekken_coach.reader` | Attach to process, resolve offsets, emit per-frame `FrameRecord`s | Interpret meaning; write memory; touch input |
| Segmenter | `tekken_coach.segment` | Frame stream → `Interaction`s via a state machine | Read memory; know frame data |
| Frame-data xref | `tekken_coach.framedata` | Annotate `Interaction`s with punishability/gap/correct-answer labels | Read memory; call the LLM |
| Session store | `tekken_coach.session` | Buffer, persist, and load the `.jsonl` event log | Analyze content |
| Coaching | `tekken_coach.coach` | Render event log → coaching via a backend (Skill export / API) | Read memory; segment |
| CLI / output | `tekken_coach.cli` | Mode selection, orchestration, terminal rendering | Business logic |
| Data (assets) | `assets/` | move-ID map, frame-data snapshot, offset table, rubric | Code |

**The contract seam.** Everything left of the session event log is the *capture pipeline*
(reader → segmenter → xref). Everything right of it is the *coaching layer*. The `.jsonl` event
log ([03](03-data-schemas.md)) is the frozen interface between them. This seam is what makes the
hybrid LLM decision cheap: the pipeline does not know or care whether a Skill or the API consumes
its output. It also makes the pipeline independently testable against recorded fixtures with no
LLM in the loop.

## 4. Data flow & timing

- The reader polls at the game's frame cadence (60 fps → ~16.6 ms/frame). It is a **producer**
  on a bounded in-memory queue.
- The segmenter is a **streaming consumer**: it maintains a small window of recent frames and a
  per-player state machine, emitting an `Interaction` when one closes. It does not need the whole
  match in memory.
- Frame-data xref is a pure function `Interaction × FrameDataTable → LabeledInteraction`. No I/O
  beyond the loaded frame-data snapshot.
- Interactions are appended to the session buffer and flushed to `.jsonl` at round end (so a
  crash mid-match loses at most one round). The **coaching trigger** fires at match end (live) or
  session end (clean) — see [01](01-capture-modes.md).
- Coaching is the only stage with LLM latency, and it runs entirely in downtime, matching the
  latency constraint in summary §3.1.

## 5. Why this shape

- **The event-log seam** decouples the two halves so the fiddly, deterministic capture work and
  the probabilistic LLM work evolve independently and test differently.
- **Streaming segmentation** keeps memory flat over a long session and means clean-capture replay
  batches process one match at a time.
- **Assets separated from code** ([05](05-frame-data-and-move-map.md)) is what lets a Season patch
  be absorbed by re-running data/offset tooling rather than editing source.

## 6. Repository layout (proposed)

```
tekken-coach/
├── docs/                       # these specs
├── src/tekken_coach/
│   ├── reader/                 # 02
│   ├── segment/                # 04
│   ├── framedata/              # 05
│   ├── session/                # 03 (persistence)
│   ├── coach/                  # 06 (backends)
│   └── cli/                    # 07
├── assets/
│   ├── offsets/                # per-game-version memory offset tables (02)
│   ├── movemap/                # per-character move-ID → name (05)
│   ├── framedata/              # frame-data snapshot (05)
│   └── rubric/                 # knowledge-check pattern definitions (06)
├── skill/                      # the Claude Code Skill bundle (06)
├── tests/
│   └── fixtures/               # recorded frame streams + golden interactions (04)
└── tekken8-coach-solution-summary.md
```

## 7. Out of scope for v1 (noted for boundaries)

- Any real-time / mid-match display (summary §3.1, §3.2). The fast deterministic lookup layer is
  explicitly *not* built.
- Desktop viewer application (v2 — [07](07-output-and-cli.md)).
- All 32 characters. v1 is scoped to the user's most-played matchups (summary §8).
- Bot/auto-play behavior from the TekkenBot ancestry. We take only the reader; the input side is
  deleted, not disabled.
