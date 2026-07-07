# Tekken 8 AI Coaching Tool — Solution Summary

> **Purpose of this document.** This is a decisions-and-rationale summary of a solution
> arrived at through discussion. It is *input* for a later Claude Code session that will
> produce full technical design documentation. It deliberately captures the **why** behind
> each decision so the design phase does not re-litigate settled questions. Where something
> is genuinely still open, it is called out under "Open Questions."

---

## 1. Product Vision

A "side-car" coaching tool for Tekken 8. The user plays **online ranked** matches against
random opponents. As each match concludes, a companion process reviews what happened and
surfaces observations and improvement suggestions — with particular emphasis on identifying
**"knowledge checks"**: recurring situations where the user is losing to a specific piece of
matchup knowledge they don't yet have (a punishable move they aren't punishing, a string they
keep respecting or keep challenging incorrectly, a mix-up they keep guessing wrong, etc.).

Coaching is delivered **between matches**, not during them (see §3).

## 2. Why Not Video / Computer Vision

The intuitive approach — point an AI at the match footage — was evaluated and rejected:

- **Volume.** 2 hours of 60fps footage is ~432,000 frames. Feeding frames as images is
  cost-prohibitive at any real sampling rate, and current Claude/LLM models have **no native
  video input** (text + images only).
- **Resolution.** Even setting cost aside, compressed video frames + motion blur cannot
  reliably resolve the frame-perfect detail that matters in Tekken (startup frames, punish
  windows, frame advantage). CV-on-frames pays a fortune to feed the model imagery that is
  structurally incapable of showing the thing being analyzed.

**Key insight that reframes the whole project:** in a fighting game the frame-accurate detail
lives in **inputs and game state**, not in pixels. So the correct data source is the game's
memory, not the rendered image.

## 3. Two Hard Constraints That Shaped the Design

### 3.1 Latency — an LLM cannot coach mid-exchange
Tekken decisions happen in tens of milliseconds at 60fps. An LLM round-trip is hundreds of ms
to seconds. Therefore:
- The LLM's natural slot is the **downtime** — between rounds and between matches — where a
  few seconds exist to digest an event log and return sharp notes. This is also where LLM
  reasoning is strongest (aggregating habits, prioritizing what matters).
- Anything that must be **instant** (e.g., an on-screen "that was -13, punish is X") is **not
  an LLM job**. If ever built, it is a fast, deterministic, local lookup layer (TekkenBot-style),
  computed from tables with no model in the loop. Out of scope for the online use case below,
  but noted for architectural completeness.

### 3.2 The online line — no live advice during human matches
Two things flip the moment a memory-reading side-car is advising the user *during* a live
online match against another human:
1. **Risk lane.** Live online play is where anti-cheat and manual reports concentrate. A
   real-time info overlay is functionally indistinguishable from the auto-punish scripts those
   systems target, even if it never touches an input.
2. **Fairness.** Real-time computer-generated advice that helps win a live match is assistance
   the opponent did not agree to play against — regardless of the user's (entirely legitimate)
   intent. The *format* (live help vs. a human) is what crosses the line, not the motive.

**Resolution:** the tool never advises during a live online match. It acts as a **flight
recorder** — capturing silently — and the LLM coaches **after the match / between sets** from
the log. This preserves rich analysis while feeding no real-time edge.

## 4. Anti-Cheat Risk (Context, Not Legal Advice)

For **offline, read-only** use the risk is low, for structural reasons:
- Anti-cheat detection is overwhelmingly scoped to **competitive online** play, not offline modes.
- Tekken 8 in particular has notably light anti-cheat (community-reported; the director stated
  no Denuvo), with much enforcement being manual/report-based.

Caveats to carry into design:
- The **capture-timing choice (§5)** determines whether the reader is ever attached during
  online play at all.
- Whether reading process memory conflicts with the EULA/ToS is a **terms** question distinct
  from ban mechanics; not legal advice. Design should keep the reader **read-only** (never
  writes/injects) and detached from any online session unless the "live capture" option is
  explicitly chosen with eyes open.

## 5. Capture Timing — One Decision Left to the User

Both options feed the identical downstream pipeline; they differ only in immediacy vs. online
footprint. **Design should support this as a configurable mode.**

| Option | How | Pro | Con |
|---|---|---|---|
| **Live capture** | Reader attached during the ranked match, silent, coaching fires at match end | Immediate feedback | Reader is attached during online play (lower profile than a live-advice bot — read-only, no inputs, nothing shown mid-match — but not the pure-offline zero) |
| **Clean capture** | Nothing attached during ranked; afterward, batch-play saved replays offline with the reader, coach at end of session | Ranked session completely untouched | Feedback is delayed; each replay plays back in ~real time |

## 6. Architecture / Pipeline

Data source: a **live-process memory reader** in the TekkenBot lineage. Note: this reads memory
from the running game process (during play or during replay playback); Tekken 8 has **no
built-in replay-data export**, so there is no offline replay-*file* parser — playback + memory
read is the mechanism.

Stages:

1. **Reader** — emits per-frame game state: both players' move/animation IDs, distance/position,
   health, Heat state, frame advantage, round state.
2. **Segmenter** — turns the raw per-frame state stream into discrete, labeled **interactions**:
   e.g., "opponent threw [move], user blocked, it was -13, user pressed jab, user got
   counter-hit." (Raw state → discrete events with outcomes.) *This unglamorous middle layer is
   the real core of the system.*
3. **Frame-data cross-reference** — labels each interaction against ground truth: was it
   punishable, did the user punish, was the gap real, etc. Frame data is available as JSON from
   community sources (e.g. TekkenDocs / Wavu).
4. **LLM layer (the Skill)** — reads the labeled events + matchup knowledge + rubric and writes
   the coaching output.

The **Claude Skill** is where the domain edge concentrates and should hold:
- the move-ID → readable-name map,
- frame-data access,
- the **rubric** describing what a "knowledge check" is, expressed as **detectable patterns**,
- the output format for observations/suggestions.

## 7. Known Cost Sinks & Maintenance Burden

Flagged early so the design accounts for them rather than discovering them late:

- **Move-ID → name mapping**, per character: ~32 characters × hundreds of moves each. Season
  patches shift **both** the memory offsets **and** the move data. This is the primary grind and
  ongoing maintenance sink. Community TekkenBot data softens it but does not eliminate it.
- **Memory-offset breakage on patches** — the reader needs a re-discovery step after game
  updates (existing TekkenBot forks ship auto-address-update scripts as prior art).
- **Segmentation edge cases** — correctly deriving interaction outcomes is fiddly around
  blockstun, stagger, tech states, sidestep, and whiff-vs-block boundaries.

## 8. MVP Scoping — Knowledge Checks First

Do **not** build the full coach at once. Knowledge checks are the MVP because they are
enumerable, matchup-specific, and rule-checkable — exactly what this architecture is good at.
Each is: *detect the pattern in the event log → look up the frame-data answer → state the counter.*

Starter set of detectable patterns:
- A punishable move the user keeps blocking and not punishing → "that's -13, you can launch it."
- A string gap the user keeps respecting, or a fake gap they keep challenging → "duck after hit 2"
  / "that's a true string, stop pressing."
- A low-that-launches or a mid-disguised-as-low the user keeps eating.
- A plus-on-block move the user keeps mashing into a counter-hit.

**Scope narrow:** ship for the user's most-played matchups and the ~10 gimmicks that beat them
most, *before* mapping all 32 characters. Prove it narrow, then widen.

## 9. Open Questions for the Design Phase

- Which specific TekkenBot fork / memory-reading approach to build on (or fork), and its current
  Tekken 8 offset-maintenance story.
- Capture-timing default (§5) and whether both modes ship in v1.
- Concrete schema for the per-frame state record and for the segmented "interaction" record.
- Segmentation algorithm for the edge cases in §7.
- Source of truth and update process for the move-ID map and frame data (ingest cadence,
  handling of Season patches).
- Output surface: how/where coaching is presented between matches (overlay, desktop window,
  terminal, file).
- Skill structure and the exact rubric encoding for knowledge-check patterns.
