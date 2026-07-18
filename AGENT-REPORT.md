# Agent report — #14: match live candidates by startup, not the (unreliable) on-block

**Branch:** `feat/live-match-by-startup` (off `main` @ 139f329, the #13 tip) — **not merged.**
**Commit:** `b172f2d`. Gates: ruff check ✓, ruff format ✓, mypy (58 files) ✓, pytest **733 passed**.

## The bug (from the brief's trace)
`map-moves --live` reads **startup** accurately — it's the crisp contact-frame `move_frame` — but
**on-block too negative** for fast/plus moves: the attacker side is measured from the return-to-idle
animation, which lags ~10 frames, while the defender side (leaving block-stun) is accurate. The two
don't cancel, so Bryan standing jab `1` (truth **i10 / +1**) read `(startup=10, on_block=−5)`. The
on_block-primary join (`join_move`) **hard-filters by on-block (±1)**, so the +1 `1` fell outside a
−5 fingerprint's candidate set entirely and stayed unmappable. `recovery_state@0x5b4` is dead — there
is no clean first-actionable field to re-point at, so the measurement can't just be fixed.

## The fix (route around it — no datamine)
Live's only job is `move_id → notation`; the frame data the coach uses comes from **Wavu keyed by the
confirmed notation**, never the live measurement. So an accurate *shortlist* is what's needed, and
**startup gives it**.

Added **`join_move_live`** (movemap_build.py), used only by the live harness:
1. **Startup-primary**, tol **±1**, falling back to **±2** only when ±1 is empty (late-poll high read).
   Moves whose Wavu startup misses the band are ruled out.
2. **On-block never filters** — it's a rough *lower bound* on the truth (reads too negative), so it
   only **soft-ranks**: candidates with Wavu on-block `>= observed − 1` rank ahead of ones that
   contradict the bound, but a startup-match is **never dropped** for failing it. The true +1 `1`
   survives an observed −5.
3. Moves with **no Wavu startup** (later string hits) are still **offered, ranked last**, not dropped.

The log path (`join_move`, used by `movemap_miner` + the #8 audit) and #12/#13 logic are **untouched**.
The live prompt now labels the observed on-block *approximate* and guards the `None` startup/on-block
display fields (later-hit candidates print `i?`).

## Verified locally (offline)
Driving `_prompt_confirm` on the exact failure fingerprint `(startup=10, on_block=−5)` for a Bryan set
`{1:+1/i10, 1,2,4:−5/i10, 1,4,2,4:−5/i10, 2,4:−5/i10, 1,2:−8/no-startup, 3:−4/i16}`:

```
move_id 1695 (from 5 reps): startup≈10 on-block≈-5 (approximate — reads low for fast moves)
  [1] 1  (i10, +1 on block) <- top
  [2] 1,2,4  (i10, -5 on block)
  [3] 1,4,2,4  (i10, -5 on block)
  [4] 2,4  (i10, -5 on block)
  [5] 1,2  (i?, -8 on block)
```

`1` now appears **and ranks top** (before: absent); the i10 decoys are offered to disambiguate; `3`
(i16) is ruled out by startup; the no-startup later hit is offered last without crashing.

## Tests (pure, offline) — in `tests/test_movemap_build.py`
- `test_old_join_excludes_the_true_plus_move_reproducing_the_bug` — old join drops the +1 `1`.
- `test_live_join_includes_the_true_plus_move` — new join keeps `1`, ranks it top, still offers decoys.
- soft-rank-never-drops, ±1 hit, ±2 fallback, no-Wavu-startup-ranked-last, honest no_candidate.
All existing join/consensus/miner/audit/live tests stay green.

## Needs a live user run (validation)
Re-run `map-moves --live --char bryan --user p1`, clean isolated jab ×5 → **`1` appears in the
candidate list** (before it was absent); confirm it. Startup still ~i10; on-block line now labelled
approximate. Report the candidate list.

## Out of scope (unchanged from brief)
Perfecting live on-block (needs a real first-actionable field; `recovery_state@0x5b4` is dead — a
datamine, off the critical path). Reader/offsets, #12 boundary logic, #13 poll-rate — untouched.
