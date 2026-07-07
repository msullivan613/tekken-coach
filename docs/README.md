# Tekken 8 Coach — Technical Design

This directory holds the technical design for the Tekken 8 AI coaching tool. It is the
implementation-facing counterpart to [`../tekken8-coach-solution-summary.md`](../tekken8-coach-solution-summary.md),
which captures the product vision and the *why* behind the settled decisions. Read the summary
first; these specs assume its conclusions and do not re-argue them.

## Reading order

| # | Spec | What it settles | Summary open question |
|---|------|-----------------|-----------------------|
| 00 | [Architecture](00-architecture.md) | End-to-end pipeline, tech stack, module boundaries, data flow | — |
| 01 | [Capture modes](01-capture-modes.md) | Live vs. clean capture; both ship in v1 | §9 (DECISION: both) |
| 02 | [Memory reader](02-memory-reader.md) | Which prior art to build on; offset-maintenance story | §9 #1 |
| 03 | [Data schemas](03-data-schemas.md) | Per-frame state record and segmented interaction record | §9 #3 |
| 04 | [Segmenter](04-segmenter.md) | Frame-stream → interactions; edge-case handling | §9 #4 |
| 05 | [Frame data & move map](05-frame-data-and-move-map.md) | Source of truth, ingest cadence, patch handling | §9 #5 |
| 06 | [Coaching skill](06-coaching-skill.md) | Skill structure, rubric encoding, LLM backends | §9 #7 |
| 07 | [Output & CLI](07-output-and-cli.md) | Terminal output for v1; desktop app is v2 | §9 #6 (DECISION: terminal) |

## Settled foundations (from the design conversation)

These were decided before the specs were written and are load-bearing throughout:

- **Language: Python.** The entire TekkenBot lineage is Python; forking the memory-offset prior
  art and address-update tooling is far cheaper than porting. See [02](02-memory-reader.md).
- **LLM layer is hybrid, Skill-default.** The coaching layer's stable contract is a labeled
  **event-log file**. v1's default consumer is a **Claude Code Skill** — it runs inside Claude
  Code on the user's existing Pro/Max subscription, at **zero marginal cost per match**. An
  optional **direct Claude API backend** (pay-per-token, one-command automation) sits behind the
  same contract for users who want it. See [06](06-coaching-skill.md).
- **Both capture modes ship in v1.** Live capture (reader attached during ranked) and clean
  capture (reader attached only during offline replay playback). See [01](01-capture-modes.md).
- **Terminal output in v1.** A desktop viewer is the end goal but v2 work. See [07](07-output-and-cli.md).
- **MVP = knowledge checks, scoped narrow.** Ship the user's most-played matchups and the ~10
  gimmicks that beat them first, before mapping all 32 characters. See [06](06-coaching-skill.md).

## Design principles carried from the summary

1. **The reader is read-only, always.** It never writes game memory, never injects input. This
   is both a correctness boundary and the core of the anti-cheat risk posture (summary §4).
2. **No real-time advice during live online matches.** During a live ranked match the tool is a
   silent flight recorder; all coaching is delivered between matches/sets (summary §3.2).
3. **The segmenter is the real core.** Raw frames are cheap; turning them into correctly-labeled
   interactions around blockstun/tech/whiff boundaries is where the system earns its keep
   (summary §6, §7). It gets the most detailed spec.
4. **Frame-accurate truth lives in game state, not pixels.** No computer vision anywhere in the
   pipeline (summary §2).
