# Output format — the coaching report

One report per session. **Between-match, terminal-friendly, short** — it's read in
the seconds of downtime between matches, so it must be scannable at a glance. This
is the exact shape both coaching backends (the Skill and the `--coach api` backend)
produce; keep to it.

## Shape

```
TEKKEN COACH — <User char> vs <Opponent char>   (<result>, <capture mode> capture)

Top knowledge checks this set:

1. <what happened, with the count>
   Ground truth: <the frame data>
   Fix: <the exact input>
   e.g. <one interaction id>

2. <...>

3. <...>

Drill this next: <the single highest-value habit for the next set>
```

## The pieces

- **Header line** — matchup (`user_char` vs opponent), the result, and the capture
  mode (`live` = immediate, `clean` = delayed/replay). One line.
- **Top knowledge checks (ranked, ~3).** Each entry is four short parts:
  - *What happened*, **with the count** — "You blocked Kazuya's df+2 6× and never
    punished." The count is what turns a note into a knowledge check; always include
    it.
  - *Ground truth* — the frame-data fact from `labels`: "df+2 is −12 on block."
  - *The fix* — the exact input to press, from `labels` (`correct_punish`,
    `duck_punish`, or the concrete action): "Punish with f,F+2 (i15, launches)."
  - *One example* — a single interaction `id` from the log (e.g. `m2-r1-i034`) so
    the user can find the moment. One is enough; don't list them all.
- **One thing to drill** — a single line naming the highest-value habit to take
  into the next set. Pick the one check whose fix will win the most rounds.

## Rules

- **~3 checks, not a dump.** A focused few the user will fix beats an exhaustive
  list they'll skim past. If only one or two genuine (recurring) knowledge checks
  exist, surface only those — don't pad.
- **Counts are mandatory** on each check — "6× this set", "4 rounds in a row".
- **Every fix is a concrete input**, taken from the labels, never generic advice.
- **Short.** No preamble, no sign-off, no restating the rubric. Lead with the
  report. If nothing recurred enough to coach, say so in one line and name the one
  thing worth watching.
- Plain text. Light structure (numbers, short labels) is fine; assume a terminal,
  so no tables or wide layout.
