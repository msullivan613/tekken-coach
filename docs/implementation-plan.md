# Implementation Plan & Orchestration

This file breaks the [specs](README.md) into implementable chunks. Each chunk is sized for a
single focused Claude Code session. They are **sequenced by dependency** and implemented one at a
time; an orchestrator reviews each against the specs before the next begins.

## Dependency graph

```
C0 schemas + skeleton ──┬──► C1 frame data & move map ──► C2 xref + rubric ──┐
                        │                                                    │
                        ├──► C3 segmenter ───────────────────────────────────┤
                        │                                                     ▼
                        └──► C4 memory reader ─────────────────────────► C6 CLI + orchestration
                                                                              ▲
                             C5 coaching (skill + api) ────────────────────────┘
```

**Build order:** C0 → C1 → C2 → C3 → C4 → C5 → C6. C2/C3/C4 all depend only on C0 (plus C1 for
C2) and could be reordered, but this order keeps the offline-testable spine (schemas → data →
xref → segmenter) solid before the environment-dependent reader lands, and coaching/CLI last since
they integrate everything.

## Sizing legend
**S** ≈ a short focused session · **M** ≈ a full session · **L** ≈ a long session, candidate to split.

---

## C0 — Project skeleton, data schemas, session store  ·  **M**
- **Specs:** [00](00-architecture.md) §6, [03](03-data-schemas.md)
- **Depends on:** nothing (foundation; everything imports it)
- **Deliverables:**
  - `pyproject.toml` (package `tekken_coach`, deps, entry point `tekken-coach`), `src/tekken_coach/`
    layout per [00](00-architecture.md) §6, tooling: `ruff`, `pytest`, `mypy` (or `pyright`).
  - `tekken_coach.schemas`: the record types + enums from [03](03-data-schemas.md) — `FrameRecord`,
    `PlayerFrame`, `Interaction`, `LabeledInteraction`, session header, and every enum
    (`action_state`, `defender_reaction`, `outcome`, labels). Pydantic or dataclass; JSON-serializable.
  - `tekken_coach.session`: write/read the `.jsonl` session log ([03](03-data-schemas.md) §5) —
    header line + append `LabeledInteraction` lines, round-end flush, load + iterate.
  - `schema_version` constant + compatibility gate ([03](03-data-schemas.md) §6).
- **Out of scope:** any reader/segmenter/xref logic; no game, no frame data.
- **Acceptance criteria:**
  - Every field/enum in [03](03-data-schemas.md) is present with the spec's names and types.
  - Round-trip test: object → `.jsonl` → object is lossless for all three record types.
  - A log with an unknown **major** `schema_version` is rejected; additive **minor** is tolerated.
  - `mypy`/lint clean; `pytest` green.
- **Test strategy:** pure unit tests, fully offline.

---

## C1 — Frame data & move map (assets + loaders + ingest)  ·  **M**
- **Specs:** [05](05-frame-data-and-move-map.md) §2, §3
- **Depends on:** C0
- **Deliverables:**
  - `assets/movemap/` and `assets/framedata/` formats exactly as [05](05-frame-data-and-move-map.md)
    §2.2 / §3.2, with `index.json` / `manifest.json` and the `current ->` snapshot pointer.
  - A small **committed sample snapshot** covering the ~2 scoped MVP matchups (summary §8) so
    downstream chunks have real data to test against.
  - `tekken_coach.framedata` loaders: load move map + the `current` frame-data snapshot into typed
    structures; miss-tolerant lookups (unknown id → `frame_data_matched:false` path,
    [05](05-frame-data-and-move-map.md) §2.3).
  - `fetch-framedata` ingest command ([05](05-frame-data-and-move-map.md) §3.3): read
    `pbruvoll/tekkendocs` `wavuConvertedCsv/<char>/*.csv` **at a pinned commit SHA** → parse
    (split `Hit level` into `hits[]`) → normalize to the §3.2 schema → diff vs `current` → write
    snapshot (manifest records the pinned SHA + attribution) → repoint on approval.
  - `NOTICE`/`THIRD_PARTY_LICENSES` attributing **tekkendocs.com and rbnorway.org** for the data
    (repo *code* is restrictively licensed — data only, never vendor the app code).
- **Out of scope:** the xref computations (C2); the CLI top-level wiring (C6 — just expose the
  command callable).
- **Acceptance criteria:**
  - Asset files validate against the spec's shapes; `framedata_key` join is explicit.
  - Loader unit tests: known id resolves; unknown id degrades, doesn't raise.
  - `fetch-framedata` writes an immutable dated snapshot and only moves `current` on approval;
    source URLs treated as verify-at-runtime (not hard-coded contracts), per [05](05-frame-data-and-move-map.md) §3.1.
- **Test strategy:** offline against the committed sample; network fetch behind an integration
  test that can be skipped in CI.

---

## C2 — Frame-data cross-reference + rubric machine layer  ·  **M**
- **Specs:** [05](05-frame-data-and-move-map.md) §4, [06](06-coaching-skill.md) §4.1, [03](03-data-schemas.md) §3–§4
- **Depends on:** C0, C1
- **Deliverables:**
  - `tekken_coach.framedata.xref`: pure function `Interaction × MoveMap × FrameData × Rubric →
    LabeledInteraction` ([05](05-frame-data-and-move-map.md) §4.1) — punishability, string-gap,
    Heat selection, observed-vs-canonical reconciliation ([05](05-frame-data-and-move-map.md) §4.2).
  - The **machine-layer rubric** ([06](06-coaching-skill.md) §4.1): the starter pattern set as
    `(id, trigger predicate, recurrence rule)` specs; sets `labels.is_knowledge_check` /
    `knowledge_check_ids`.
  - `KnowledgeCheckTally` aggregation ([03](03-data-schemas.md) §4).
- **Out of scope:** the LLM judgment layer (C5); memory reads.
- **Acceptance criteria:**
  - Pure function — no I/O beyond the loaded snapshot; deterministic.
  - Each starter pattern in [06](06-coaching-skill.md) §4.1 has a fixture proving trigger + recurrence.
  - Observed vs canonical: agreement uses canonical; disagreement keeps observed + note; null-observed
    falls back to canonical (all three branches tested).
  - Unknown move → unlabeled, `is_knowledge_check:false`, no crash.
- **Test strategy:** fixture `Interaction`s → asserted `LabeledInteraction`s; fully offline.

---

## C3 — Segmenter  ·  **L (may split 3a core / 3b edge cases)**
- **Specs:** [04](04-segmenter.md) (all)
- **Depends on:** C0
- **Deliverables:**
  - `tekken_coach.segment`: the streaming per-exchange state machine ([04](04-segmenter.md) §2),
    deriving the four outputs ([04](04-segmenter.md) §3) and the structural `outcome` guess.
  - **3a (core):** NEUTRAL→COMMIT→CONTACT→FOLLOWUP, block/hit/whiff, single-hit punish detection,
    observed-advantage counting.
  - **3b (edge cases):** every case in [04](04-segmenter.md) §4 — blockstun/hitstun/stagger,
    multi-hit strings, tech states, sidestep/whiff, counter-hit, Heat transitions, dropped-frame
    tolerance, round/match boundaries.
  - `tests/fixtures/`: hand-authored `FrameRecord` streams + golden `Interaction`s
    ([04](04-segmenter.md) §7), one per §4 case; property tests (no overlap, within-round, attacker≠defender).
- **Out of scope:** frame-data labeling (that's C2, downstream); reading memory (C4).
- **Acceptance criteria:**
  - Deterministic: same stream → same interactions ([04](04-segmenter.md) §6).
  - Every §4 edge case has a passing fixture before it's considered done.
  - Property tests hold across all fixtures.
- **Test strategy:** hand-authored fixtures (real captures replace/augment them after C4). No game
  needed. **Recommend splitting** 3a and 3b into two review cycles given the edge-case volume.

---

## C4 — Memory reader  ·  **L (environment-dependent)**
- **Specs:** [02](02-memory-reader.md), emits [03](03-data-schemas.md) §1
- **Depends on:** C0
- **Deliverables:**
  - `tekken_coach.reader`: attach read-only, resolve offsets, emit `FrameRecord`s at frame cadence.
    **Read-only invariant** — no write/inject code path exists ([02](02-memory-reader.md) §2, §5).
  - `assets/offsets/` versioned table format + `index.json` + version detection
    ([02](02-memory-reader.md) §3); fail-closed on unknown version.
  - `update-offsets` — **clean-room** re-implementation of the Jin-vs-Kazuya re-discovery technique
    ([02](02-memory-reader.md) §4); do **not** copy the fork's script text (see §5 licensing).
  - Reader self-check / `doctor` ([02](02-memory-reader.md) §6) and the §7 failure-mode handling.
  - `NOTICE`/`THIRD_PARTY_LICENSES` crediting roguelike2d (MIT, 2017) for any MIT-root code ported.
- **Out of scope:** segmentation; coaching; the live/clean *mode* logic (C6 owns triggers — reader
  just exposes attach + a match/replay-state signal).
- **Acceptance criteria:**
  - No memory-write or input-injection primitive anywhere in the module (grep-able).
  - Unknown game version fails closed with the §4 runbook; self-check gates capture.
  - `FrameRecord`s conform to [03](03-data-schemas.md) §1 (validated against C0 schemas).
  - Licensing posture honored: MIT-root code attributed; offsets live as data; `update-offsets` is
    original code.
- **Test strategy:** self-check + a smoke capture require the game (Windows). Where the game isn't
  available to the review, verify the read-only invariant, offset-table handling, version
  fail-closed, and schema conformance by inspection + unit tests on the non-attach logic. Capture a
  couple of **real `FrameRecord` fixtures here** to feed back into C3's suite.

---

## C5 — Coaching layer: Skill bundle + API backend  ·  **M**
- **Specs:** [06](06-coaching-skill.md) (esp. §2, §3, §4.2, §5)
- **Depends on:** C0 (reads `.jsonl`), C1 (assets), C2 (labels the log carries)
- **Deliverables:**
  - `skill/` bundle: `SKILL.md` (frontmatter `name`/`description`; body per [06](06-coaching-skill.md) §2)
    + `references/rubric.md` (judgment layer, [06](06-coaching-skill.md) §4.2) +
    `references/output-format.md` ([06](06-coaching-skill.md) §5) + `references/reading-the-log.md`.
  - `tekken_coach.coach`: the optional **API backend** ([06](06-coaching-skill.md) §3) — `anthropic`
    SDK, `claude-opus-4-8`, adaptive thinking + effort high, prompt-cached stable prefix, log as
    volatile suffix; system prompt **generated from the same `skill/` sources** (single source of truth).
  - Auth handling: use the user's own credential; if none, print guidance + fall back to the Skill path.
- **Out of scope:** the machine rubric (C2, already in the log); capture; CLI wiring (C6).
- **Acceptance criteria:**
  - `SKILL.md` loads/describes correctly as a Claude Code Skill; references are progressive-disclosure.
  - API backend builds its system prompt by concatenating the `skill/` sources — no second copy of
    the domain content.
  - Report matches [06](06-coaching-skill.md) §5 (ranked ~3 checks with counts, ground truth, fix,
    example id, one drill; short).
  - Runs against a sample `.jsonl` end to end (Skill path by hand; API path if a key is present).
- **Test strategy:** the API backend's assembly/caching/auth-fallback are unit-testable with a
  mocked client; a live smoke run needs a key. The Skill is validated by invoking it in Claude Code
  on a sample log.

---

## C6 — CLI, orchestration & capture-mode wiring  ·  **M**
- **Specs:** [07](07-output-and-cli.md), [01](01-capture-modes.md), [00](00-architecture.md) §4
- **Depends on:** C0–C5
- **Deliverables:**
  - `tekken_coach.cli`: the `tekken-coach` entry point + commands `live`, `clean`, `coach`, `doctor`
    (and registration of `update-offsets` / `fetch-framedata` delivered in C4/C1) — [07](07-output-and-cli.md) §1.
  - Capture-mode orchestration ([01](01-capture-modes.md)): arm/record/coach lifecycle for live;
    replay-batch lifecycle for clean; the **no-mid-match-output** invariant; `clean` as default;
    `user_player` validation ([01](01-capture-modes.md) §5).
  - The producer→queue→segmenter→xref→session flow ([00](00-architecture.md) §4); round-end flush.
  - Terminal renderer ([07](07-output-and-cli.md) §3): TTY-aware, degrades without ANSI; timing
    respects capture mode; Skill-path vs API-path output ([07](07-output-and-cli.md) §2).
  - `config.toml` defaults.
- **Out of scope:** re-implementing any subsystem — this wires reviewed components together.
- **Acceptance criteria:**
  - `tekken-coach live` / `clean` run the full pipeline to a written `.jsonl` and the correct
    coaching hand-off for the chosen `--coach` backend.
  - No output is emitted mid-match in live mode (invariant test on the state machine).
  - Mode-agnostic below the trigger layer — no `if mode ==` past reader/trigger ([01](01-capture-modes.md) §5).
- **Test strategy:** drive the pipeline on recorded fixtures (from C3/C4) end to end with a fake
  reader replaying a `FrameRecord` stream — exercises live/clean triggers without the game.

---

## Orchestration protocol (how each chunk is run)

For each chunk, the orchestrator:

1. **Issues a handoff brief** to the implementing Claude Code instance: the chunk's goal, the exact
   spec sections to read, the deliverables, the acceptance criteria, the test requirements, and an
   explicit **"do not"** list (scope boundaries + the licensing constraints for C4).
2. The instance implements on a **feature branch** with tests, and reports what it did.
3. The orchestrator **reviews the diff against the spec and the acceptance criteria** — not just
   "does it run," but does it match the schema names, honor the invariants (read-only reader,
   no-mid-match-output, pure-function xref, deterministic segmenter), and cover the required
   fixtures. Runs the tests where the environment allows.
4. Findings are returned as **blocking vs. nits**; the loop repeats until the chunk meets the bar.
5. On acceptance, the chunk merges and the next begins. Fixtures captured in C4 are fed back to
   strengthen C3's suite.

### Cross-cutting review checklist (applied to every chunk)
- Matches the spec's names/types/enums exactly (the specs are the contract).
- Honors the load-bearing invariants: reader is read-only; nothing renders mid-match; the
  event-log `.jsonl` is the only seam between capture and coaching; xref is pure; segmenter is
  deterministic.
- Tests exist and are meaningful (fixtures for every edge case the spec enumerates), and pass.
- Degrades rather than crashes on unknown move ids / stale data / dropped frames.
- No scope creep into a later chunk; no copied unlicensed fork source.
