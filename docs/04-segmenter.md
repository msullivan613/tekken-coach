# 04 — Segmenter

> Settles summary §9 #4: *the segmentation algorithm and its edge cases.* The summary (§6, §7)
> names this "the unglamorous middle layer [that] is the real core of the system," so this is the
> most detailed spec. Input: a stream of `FrameRecord`s ([03](03-data-schemas.md) §1). Output: a
> stream of `Interaction`s ([03](03-data-schemas.md) §2).

## 1. What an "interaction" is

An **interaction** is one bounded attacker→defender exchange: someone commits an attack that
makes contact (or is evaded), the defender reacts, and the exchange resolves back toward neutral,
including the **defender's immediate follow-up** (their chance to punish / their mash). We segment
at the granularity of *"a thing that has a right answer"* — because that is exactly what a
knowledge check evaluates (summary §8).

Neutral movement, whiffs into nothing, and dead time between exchanges are **not** interactions
(they produce no `Interaction`; at most they set context like distance for the next one).

## 2. Core model: a per-exchange state machine over the frame stream

The segmenter is a streaming consumer holding: a short ring buffer of recent frames (for
look-back), the current open interaction (if any), and a small amount of carried context
(who had advantage leaving the last interaction → `context.attacker_pressure`).

```
        ┌─────────┐
        │ NEUTRAL │◄───────────────────────────────────────────────┐
        └────┬────┘                                                 │
             │ attacker enters an attack move within threat range   │ resolve() emits Interaction
             ▼                                                      │
        ┌─────────┐  contact detected     ┌──────────┐  defender    │
        │ COMMIT  │──────────────────────►│ CONTACT  │  window ends  │
        └────┬────┘  (block/hit/evade)    └────┬─────┘──────────────►│
             │ move recovers, no contact       │ track defender      │
             │ (pure whiff, no punish)         │ follow-up + result  │
             └────────────► NEUTRAL            ▼                     │
                                          ┌──────────┐               │
                                          │ FOLLOWUP │───────────────┘
                                          └──────────┘
```

**Transitions**
- **NEUTRAL → COMMIT:** a player's `move_id` changes to an *attacking* move (per move map, or per
  `action_state == attack`) while distance ≤ a range threshold that makes contact plausible. That
  player becomes `attacker`.
- **COMMIT → CONTACT:** the defender's state shows contact — a transition into `blockstun`,
  `hitstun`, `stagger`, `thrown`, or a registered `counter_state`. Classify `defender_reaction`
  ([03](03-data-schemas.md) §2 enum) from *which* transition.
- **COMMIT → NEUTRAL (whiff):** the attacker's move reaches recovery with no defender contact and
  the defender never entered a punish attempt → discard as non-interaction (unless the defender
  whiff-punishes — that flips to CONTACT with `whiff_punished`).
- **CONTACT → FOLLOWUP:** contact resolved; open the defender's **action window** — the frames
  from when the defender first becomes actionable until they either act or return to neutral.
- **FOLLOWUP → NEUTRAL (resolve):** the defender acts (record `follow_up.move_id/result/reaction_frames`)
  or the window elapses with no action (`follow_up = nothing`). Emit the `Interaction`.

## 3. Deriving the four things that matter

For each interaction the segmenter must produce, from state transitions alone:

1. **Who attacked / what move.** `attacker`, `attacker_move_id` — from the COMMIT transition.
   Use the *first* attacking move of the exchange; multi-hit strings are handled in §4.
2. **How the defender reacted.** `defender_reaction` — from the CONTACT transition kind.
3. **Observed advantage.** `observed_advantage` — count frames between when the **attacker**
   becomes actionable again and when the **defender** becomes actionable again after contact:
   `defender_actionable_frame − attacker_actionable_frame`. Negative ⇒ attacker is at
   disadvantage ⇒ punishable. This is the *measured* value; [05](05-frame-data-and-move-map.md)
   supplies the *canonical* one and they cross-check (§6 there).
4. **What the defender did with it.** `follow_up` + a structural `outcome` guess
   ([03](03-data-schemas.md) §2). E.g. defender was +something and pressed nothing on a
   punishable move → `no_punish`.

"Actionable" = first frame a player exits all stun/recovery states and could input a move
(`action_state` back to `neutral`/`crouch`/`sidestep` and stun flags clear).

## 4. Edge cases (the summary §7 fiddly boundaries)

These are the situations that make naive segmentation wrong. Each has an explicit rule and a
fixture in `tests/fixtures/` ([00](00-architecture.md) §6).

### 4.1 Blockstun vs. hitstun vs. stagger
All three are "defender is stuck," but they mean opposite things for coaching (blocked = maybe
punish; hit = you got hit; stagger = a mid you should have blocked low, or vice-versa).
- Distinguish by the **specific memory flag** (`block_stun`, `hit_stun`, `action_state==stagger`),
  not by "defender can't act." The reader exposes them separately precisely for this
  ([03](03-data-schemas.md) §1).
- **Stagger** (e.g. blocked a move that forces a stagger, or got hit by a stagger-on-normal-hit):
  treat as its own reaction; the defender's "actionable" frame is when stagger ends, and any
  extra disadvantage is part of `observed_advantage`.

### 4.2 Multi-hit strings — segment per *string*, annotate per *hit*
A string like Kazaya's `df+1,2` or a long Jin string is **one interaction** whose
`attacker_move_id` is the string's entry, but whose labeling ([05](05-frame-data-and-move-map.md))
needs per-hit gap data.
- The segmenter keeps the interaction **open across consecutive hits of the same string** (move
  IDs that chain within blockstun without the defender becoming actionable between them).
- It records the **hit index at which the defender's state changed** (blocked hit 1, got hit by
  hit 2 ⇒ they pressed in a gap, or the gap was real and they ate it). This feeds
  `string_gap` / `gap_size` labeling and the "duck after hit 2 / stop pressing" knowledge checks.
- It records, per hit, whether the defender **blocked it standing** vs **ducked/evaded it** — a
  ducked high whiffs and breaks the string there, so a high that appears in the *blocked* sequence
  means the defender stood on it. Combined with per-hit `hit_level` from frame data
  ([05](05-frame-data-and-move-map.md) §3.2), this lets xref flag a **duckable high the user blocked
  standing** (the `standing_duckable_high` check, [06](06-coaching-skill.md) §4.1) — e.g. Paul's
  `df+1,1,2` (mid→high→mid), where ducking hit 2 gives a punish before hit 3.
- Boundary rule: the string interaction **closes** when the defender becomes actionable *between*
  hits (they interrupted, or there was a real gap they acted in) or when the string fully recovers.

### 4.3 Tech states (throw breaks, ground tech, recovery)
- **Throw break window:** `throw_active` on attacker + defender in `throw_tech_window` → if the
  defender breaks, `defender_reaction = throw_broke`; if not, `thrown`. Coaching cares whether the
  user broke throws they could break (a knowledge check candidate).
- **Ground/okizeme tech:** on knockdown, the follow-up window is the **wakeup**, not the current
  standing action window. When the defender is in `knockdown`/`wakeup`, extend FOLLOWUP to their
  wakeup-actionable frame so we correctly attribute "got hit on wakeup" vs "the mixup after."

### 4.4 Sidestep / backdash / whiff
- If the defender's `action_state == sidestep` (or moves out of range) during COMMIT and the
  attacker's move recovers with no contact → `defender_reaction = evaded`. If the defender then
  punishes the whiff → `whiff_punished` and the whiff-recovery frames feed `observed_advantage`.
- A **pure neutral whiff** with no defender involvement is discarded (§2). Distinguish "whiffed
  because spaced out" (not an interaction) from "whiffed because sidestepped" (an interaction —
  the user *did* something right, and it's coachable positively).

### 4.5 Counter-hit / punish-counter
`counter_state` on the *defender's* hit (from [03](03-data-schemas.md) `PlayerFrame`) marks
`counter_hit` / `punish_counter`. This is central to the "plus-on-block move the user keeps
mashing into a counter-hit" knowledge check: the interaction records that the user's `follow_up`
move `got_counter_hit`.

### 4.6 Heat transitions & Heat-engager
Heat activation mid-string changes frame advantage (many moves are plus in Heat). The interaction
records `context.attacker_heat`/`defender_heat` at start and detects a Heat activation **within**
the interaction (note it). Frame-data xref must select the **Heat-state-appropriate** advantage
value ([05](05-frame-data-and-move-map.md)).

### 4.7 Dropped frames / poll gaps
If the reader missed frames (frame-counter gap, [02](02-memory-reader.md) §7), the segmenter:
- tolerates gaps up to a small threshold (e.g. ≤3 frames) by interpolating state continuity, and
  records `notes: ["gap-tolerated:N"]`;
- if a gap is large enough to make advantage-counting unreliable, it still emits the interaction
  but marks `observed_advantage: null` and lets xref fall back to canonical frame data only.

### 4.8 Round/match boundaries
Round-over/match-over `match_state` closes any open interaction as-is (marking it truncated in
`notes`) so a round-ending hit is still recorded. No interaction spans a round boundary.

## 5. What the segmenter does *not* decide

- It does **not** know whether −13 is punishable — that is defender-character-dependent and lives
  in xref ([05](05-frame-data-and-move-map.md)). The segmenter reports *observed* frames and a
  *structural* outcome guess; xref supplies truth and may override the guess.
- It does **not** know move names — only IDs. Name resolution is xref/move-map.
- It does **not** know which player is the user — it segments symmetrically; the `outcome`
  field's user-perspective values are finalized once `user_player` (header, [03](03-data-schemas.md) §5)
  is applied.

## 6. Determinism & the `outcome` guess

The segmenter is **fully deterministic**: same `FrameRecord` stream → same `Interaction`s. This
is what makes fixture testing possible. Its `outcome` is an explicit *guess* that xref confirms or
corrects, so a segmenter mistake is recoverable downstream rather than silently authoritative.

## 7. Testing strategy (this is where correctness is won)

- **Recorded fixtures.** Capture real `FrameRecord` streams (both modes) for known scenarios —
  a blocked −13, a duckable string, a true string, a throw the user could break, a whiff punish,
  a Heat-plus mixup — and freeze them under `tests/fixtures/`.
- **Golden interactions.** Each fixture has a hand-verified expected `Interaction` list.
  Regression = the segmenter reproduces the goldens frame-for-frame.
- **Edge-case fixtures are mandatory** for every §4 case before that case is considered handled.
- **Property tests:** interactions never overlap; every interaction lies within one round; every
  emitted interaction has an attacker ≠ defender and start_frame < end_frame.
- Because the segmenter needs no LLM and no live game, the whole suite runs in CI on the recorded
  streams. New Season patch data can be regression-checked by re-recording fixtures.
