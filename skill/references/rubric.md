# Rubric — the judgment layer

This is the layer rules can't do. The pipeline already ran the **machine rubric**
(the trigger predicates in `tekken_coach.framedata.rubric`) and set
`labels.is_knowledge_check` / `labels.knowledge_check_ids` on every interaction it
fired on. What's left is judgment: **prioritize** across everything that triggered,
**de-duplicate** related habits, decide what's noise, and phrase the fix in the
user's matchup. Do that here.

## What a knowledge check is (and why recurrence matters)

A knowledge check is a **recurring, exploitable gap in the user's Tekken
knowledge** — not a single mistake. The whole point of the pipeline was to prove
these narrowly and deterministically; your job is to weigh them.

- A habit that fired **once or twice** is a fluke. Do not coach it. (The machine
  layer tags a check the moment its trigger fires; it becomes worth coaching only
  once it *recurs*.)
- A habit that fired **three or more times against the same move/string** is a
  genuine knowledge check. That threshold is why "you missed this once" becomes
  "you missed this 6× this set" — recurrence is the signal.

If you compute a tally, group by `(knowledge_check_id, attacker_char,
attacker_move_id, matchup)` and count — that grouping is what separates a real
pattern from scattered noise. (`tekken_coach.framedata.tally.build_tally`
implements exactly this if you want to run it, but you can also just count as you
stream.)

## The knowledge-check patterns

Each `knowledge_check_id` you'll see in `labels.knowledge_check_ids`, what it means,
and the shape of the fix. The frame-data specifics (the `-N`, the exact punish) are
already in `labels` — use them verbatim.

| id | What the user did | The fix to phrase |
|---|---|---|
| `punish_missed` | Blocked a punishable move and did nothing (`was_punishable` true, no punish taken) | "X is `on_block`. Punish with `correct_punish`." |
| `respected_fake_gap` | Blocked an interruptible string gap and stood there — could have interrupted | "There's a gap after that hit — you can interrupt / steal your turn back." |
| `challenged_true_string` | Mashed inside a true (uninterruptible) string and got counter-hit | "That's a true string. Stop pressing — block it and take your turn after." |
| `standing_duckable_high` | Blocked a mid-string high **standing** that was duck-punishable (a missed duck-punish) | "Duck hit K (the high) and punish before the last hit: `duck_punish`." |
| `ate_low` | Kept standing on a low on a known mix | "You keep standing on the low — you have to block low / react to X." |
| `ate_mid` | Kept ducking into a mid on a known mix | "You keep ducking the mid — stand and block / react to X." |
| `mashed_into_plus` | Pressed after a plus-on-block move and got counter-hit | "X is plus on block. Stop mashing after it — wait your turn." |

## How to rank

Score each recurring knowledge check on three axes and surface the top few:

1. **Frequency** — how many times it recurred this session. More = higher.
2. **Round impact** — how much each occurrence costs. A missed launch punish or a
   counter-hit into a combo swings a round; standing on a jab does not. Weight
   `punish_missed` on a launcher, `challenged_true_string` into big damage, and
   `standing_duckable_high` (a whole missed punish) heavily; small chip losses
   lightly. Use `on_block` magnitude, `punish_window`, and whether the punish
   launches as signal.
3. **Learnability** — how fixable it is with one concrete instruction. "Punish
   df+2 with f,F+2" is a clean, drillable fix and ranks up; a diffuse spacing
   problem ranks down. Favor checks with a single, specific input as the answer.

Rank by roughly **frequency × round-impact × learnability**. A twice-a-set launch
punish you're missing can outrank a five-times low you keep eating, because the
launch costs a round each time.

## De-duplicate

Collapse related habits into one line. If the user misses the punish on the same
move at two ranges, or eats the same mix logged under two ids, that's **one**
knowledge check, not two. Don't pad the list with near-duplicates — the value is a
focused few.

## How many to surface, and tone

- Surface **~3** knowledge checks. This is a between-match nudge, not an audit.
  Better three the user will actually fix than ten they'll ignore.
- Then name **one** thing to drill — the single highest-value habit to take into
  the next set.
- Tone: **specific, actionable, matchup-scoped.** Name the exact move, the exact
  input, the exact count. "Kazuya's df+2 is −12 — you blocked it 6× and never
  punished; f,F+2 launches it" beats "watch out for punishable mids." Prove it
  narrow. No filler, no hedging, no generic advice that would apply in any matchup.
