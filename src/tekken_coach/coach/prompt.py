"""System-prompt assembly for the coaching layer — the single source of truth (docs/06 §3).

Both coaching backends share one authored body of domain content. The **Skill**
(``skill/``) reads its ``SKILL.md`` + ``references/`` on demand inside Claude Code; the
**API backend** (:mod:`tekken_coach.coach.api`) sends the *same* files as its system prompt.
This module is the seam: it reads the ``skill/`` sources off disk and concatenates them into
``RUBRIC_AND_INSTRUCTIONS``. There is deliberately **no second copy** of the rubric/output
rules in Python (docs/00 §3, docs/06 §3) — change the Skill and the API backend changes with it.

It also resolves the shared ``assets/`` directory (the move-map + frame-data snapshot) so the
Skill and the API backend point at the *same* assets, without a committed ``skill/assets``
symlink (which does not survive this project's Windows/WSL checkout — docs/06 §2).
"""

from __future__ import annotations

from pathlib import Path

from tekken_coach.session.store import read_header

# ``.../src/tekken_coach/coach/prompt.py`` -> repo root is four parents up.
REPO_ROOT = Path(__file__).resolve().parents[3]
SKILL_DIR = REPO_ROOT / "skill"
ASSETS_DIR = REPO_ROOT / "assets"

# The ``skill/`` sources concatenated into the system prompt, in order. docs/06 §3 names the
# first three (``SKILL.md`` + ``references/rubric.md`` + ``references/output-format.md``);
# ``reading-the-log.md`` is included too because the headless API backend has no progressive
# disclosure — it must be told how to parse the ``.jsonl`` up front, whereas the Skill loads
# that file on demand. All four are ``skill/`` sources, so this remains a single source of truth.
SKILL_SOURCES: tuple[str, ...] = (
    "SKILL.md",
    "references/rubric.md",
    "references/output-format.md",
    "references/reading-the-log.md",
)


def _strip_frontmatter(text: str) -> str:
    """Drop a leading ``---``-delimited YAML frontmatter block (Claude Code Skill metadata).

    The frontmatter is loader metadata (``name`` / ``description``), not coaching instructions,
    so it is stripped before the body goes into the API system prompt. Text with no frontmatter
    is returned unchanged.
    """
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                return "\n".join(lines[i + 1 :]).lstrip("\n")
    return text


def build_rubric_and_instructions(skill_dir: Path = SKILL_DIR) -> str:
    """Assemble the coaching system prompt by concatenating the ``skill/`` sources (docs/06 §3).

    This is ``RUBRIC_AND_INSTRUCTIONS`` — the stable, prompt-cacheable system prefix the API
    backend sends. It is built *from the Skill files*, never hand-copied, so the two backends
    can never drift.
    """
    parts: list[str] = []
    for rel in SKILL_SOURCES:
        raw = (skill_dir / rel).read_text(encoding="utf-8")
        body = _strip_frontmatter(raw) if rel == "SKILL.md" else raw
        parts.append(body.strip())
    return "\n\n".join(parts) + "\n"


def read_event_log_text(path: str | Path) -> str:
    """Return a session log's raw ``.jsonl`` text — the volatile user suffix (docs/06 §3).

    Validates the header's ``schema_version`` first (via ``session.store.read_header``, docs/03 §6):
    an unknown-major or missing-header log raises rather than being sent to the model.
    The full file text is the payload both backends consume — identical to what the Skill reads.
    """
    read_header(path)  # version gate; raises IncompatibleSchemaVersionError / ValueError
    return Path(path).read_text(encoding="utf-8")
