# 06 — Coaching Skill & LLM Layer

> Settles summary §9 #7: *Skill structure and the exact rubric encoding for knowledge-check
> patterns.* Also realizes the hybrid LLM decision: **a Claude Code Skill is the default v1
> coaching surface; a direct Claude API backend is an optional add-on**, both behind the session
> event-log contract ([03](03-data-schemas.md) §5).

## 1. Why hybrid, Skill-first

A Claude Code **subscription (Pro/Max) does not include programmatic Claude API access** — the
API is billed separately per token via an API key. So:

- **Default backend = Claude Code Skill.** Coaching runs *inside Claude Code*, on the user's
  existing subscription, at **zero marginal cost per match**. The app's job ends at writing the
  `.jsonl` event log; the user invokes the Skill on it in Claude Code.
- **Optional backend = Claude API.** For users who want one-command, headless coaching (no manual
  Claude Code step) and are willing to pay per-token, the same app can call the Claude API
  directly.

Both consume the identical event log and the identical rubric/resources, so the domain content is
authored once ([00](00-architecture.md) §3).

## 2. The Skill bundle (`skill/`)

A Claude Code Skill is a directory with a `SKILL.md` plus resource files it reads on demand. Our
bundle:

```
skill/
├── SKILL.md                 # the coach's instructions + rubric (progressive-disclosure entry point)
├── references/
│   ├── rubric.md            # knowledge-check patterns, expanded (§4)
│   ├── output-format.md     # exact shape of the coaching report (§5)
│   └── reading-the-log.md   # how to parse the .jsonl LabeledInteraction records (03)
└── assets -> ../assets/     # move-map + frame-data snapshot, shared with the pipeline (05)
```

`SKILL.md` frontmatter carries a `name` and a `description` that tells Claude Code *when* to load
it (e.g. "Coach a Tekken 8 ranked session from a tekken-coach event log"). The body:

1. States the coaching goal — **knowledge checks first** (summary §8), recurrence over one-offs.
2. Points at `references/rubric.md` for the detectable patterns (loaded when a session is being
   analyzed, not upfront — progressive disclosure).
3. Points at `references/output-format.md` for the report shape.
4. Instructs: read the header (who is the user, matchup, capture mode), then the
   `LabeledInteraction` stream, then **prioritize by recurrence and severity** before writing.

**The Skill does not re-derive frame data.** The pipeline already labeled every interaction
([03](03-data-schemas.md) §3, [05](05-frame-data-and-move-map.md)). The Skill reasons over
*labeled* events — aggregating habits, ranking what matters, phrasing the fix. It may consult the
move-map/frame-data assets to enrich phrasing (e.g. name the exact punish), but the ground-truth
judgments are already in `labels`.

## 3. The API backend (`tekken_coach.coach`, optional)

When enabled (`--coach api`, [07](07-output-and-cli.md)), the app calls the Claude API itself:

- **SDK:** the official `anthropic` Python package.
- **Model:** `claude-opus-4-8` (default; the coaching reasoning is the quality-sensitive step).
- **Request shape (current API):** adaptive thinking on, effort high — this is aggregation and
  prioritization reasoning, exactly where thinking helps:
  ```python
  client.messages.create(
      model="claude-opus-4-8",
      max_tokens=8000,
      thinking={"type": "adaptive"},
      output_config={"effort": "high"},
      system=[{"type": "text", "text": RUBRIC_AND_INSTRUCTIONS,
               "cache_control": {"type": "ephemeral"}}],  # stable prefix → cache it
      messages=[{"role": "user", "content": event_log_text}],
  )
  ```
- **Prompt caching:** the rubric + move-map context is the large, stable prefix; cache it so
  repeated sessions only pay to process the (small) event log. The event log is the volatile
  suffix and goes after the cached breakpoint.
- **Auth:** requires the user's own `ANTHROPIC_API_KEY` (or an `ant auth login` profile). The app
  never ships a key. If no credential is found, it prints how to set one and falls back to
  directing the user to the Skill path.
- **The system prompt is generated from the same `skill/` sources** — `SKILL.md` +
  `references/rubric.md` + `references/output-format.md` are concatenated into `RUBRIC_AND_INSTRUCTIONS`.
  Single source of truth; the API backend is not a second copy of the domain content.

## 4. Rubric encoding — knowledge checks as detectable patterns

The rubric is the domain edge (summary §6). Each knowledge check is expressed as **a detectable
pattern over `LabeledInteraction`s**, not as prose the model must infer from scratch. Two layers:

### 4.1 Machine layer (in the pipeline, [05](05-frame-data-and-move-map.md) §4.1)
The frame-data xref already sets `labels.is_knowledge_check` and `labels.knowledge_check_ids` by
running these patterns. Each pattern has an **id**, a **trigger** (a predicate over one
interaction's fields), and a **recurrence rule** (how many times across the session before it's
worth coaching). Starter set (from summary §8), as rule specs:

| id | Trigger (per interaction) | Recurrence | The coaching line |
|---|---|---|---|
| `punish_missed` | `labels.was_punishable && outcome == "no_punish"` on the same `attacker_move_id` | ≥3× vs one move | "X is −N. Punish with `correct_punish`." |
| `respected_fake_gap` | `defender_reaction == "blocked"` in-string && `labels.string_gap == "interruptible"` && defender did nothing | ≥3× vs one string | "There's a gap after hit K — you can interrupt." |
| `challenged_true_string` | `follow_up.result == "got_counter_hit"` && `labels.string_gap == "true"` | ≥3× vs one string | "That's a true string. Stop pressing; block it." |
| `standing_duckable_high` | in-string && `labels.duckable_high_hit != null` (user **blocked a high standing** mid-string that could have been ducked for a punish) | ≥3× vs one string | "X is mid→high→mid — duck hit K (the high) and punish before the last hit: `duck_punish`." |
| `ate_low` / `ate_mid` | `defender_reaction == "hit"` && `labels.move_property` low/mid on a known mix | ≥3× vs one move | "You keep standing on the low / ducking the mid — react to X." |
| `mashed_into_plus` | attacker `+on_block`, user's `follow_up` `got_counter_hit` | ≥3× vs one situation | "X is plus. Stop mashing after it; wait your turn." |

`ate_low` / `ate_mid` is one table row but **two distinct pattern ids** (they split on the move's
height, and the coaching line and the tally grouping differ) — the machine layer carries them as
two rules, so the starter set is 7 rules across these 6 rows.

**Two-phase trigger/recurrence split.** Each pattern's **trigger** is a pure predicate over one
interaction (evaluated in the xref, which sets `labels.is_knowledge_check` / `knowledge_check_ids`
when a trigger fires). The **recurrence rule** (the `≥N×` threshold) is applied **session-level in
the `KnowledgeCheckTally`** ([03](03-data-schemas.md) §4) — a per-interaction function can't see
session counts. So an interaction is *tagged* the moment its trigger fires; it becomes a *coached*
knowledge check only once the tally clears its recurrence threshold.

Recurrence is what turns a fluke into a *knowledge check* (summary §1). The
`KnowledgeCheckTally` ([03](03-data-schemas.md) §4) is the structure that counts these.

### 4.2 Judgment layer (in the Skill/LLM, `references/rubric.md`)
The LLM layer does what rules can't: **prioritize** across many triggered checks (which 3 cost the
user the most games?), **de-duplicate** related habits, phrase the fix in the user's matchup
context, and decide what's noise. `references/rubric.md` encodes:
- the *definition* of a knowledge check and why recurrence matters,
- how to rank (frequency × round-impact × how learnable),
- how many to surface (a focused few, not an exhaustive dump),
- tone: specific, actionable, matchup-scoped (summary §8 "prove it narrow").

This split — deterministic detection in the pipeline, judgment in the LLM — is the whole
architecture's thesis (summary §6) applied to the rubric itself.

## 5. Output format

Defined once in `references/output-format.md` and used by both backends. Between-match report,
terminal-friendly ([07](07-output-and-cli.md)):

- **Header:** matchup, result, capture mode (immediate vs delayed).
- **Top knowledge checks (ranked, ~3):** each = *what happened* (with the count: "6× this set"),
  *the ground truth* (frame data), *the fix* (the exact input), and *one example* interaction id.
- **One thing to drill:** the single highest-value habit to take into the next set.
- Deliberately **short** — this is read in the downtime between matches.

## 6. Extensibility (post-MVP)
The full coach (beyond knowledge checks — neutral, movement, conversions) is future work. It slots
in as **more rubric patterns** + richer `references/`, with no pipeline change, because the event
log already carries the raw material. Ship knowledge checks narrow first (summary §8), widen by
adding patterns.
