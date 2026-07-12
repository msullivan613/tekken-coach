"""Config + flag precedence for the ``tekken-coach`` CLI (docs/07 §1).

Durable defaults live in ``~/.config/tekken-coach/config.toml`` so the common case is a bare
``tekken-coach live`` / ``clean``. Precedence, highest first (docs/07 §1.2):

    CLI flag  >  config.toml  >  built-in default (mode=clean, coach=skill).

A missing config file is not an error — the built-in defaults apply. This module is pure
(no I/O beyond reading the one TOML file) and does not touch the reader or the game.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tekken_coach.schemas import CaptureMode

# Built-in defaults (docs/07 §1: mode=clean, coach=skill; docs/01 §2 the safer mode is default).
DEFAULT_MODE = CaptureMode.clean
DEFAULT_COACH = "skill"
DEFAULT_SESSIONS_DIR = "sessions"
COACH_BACKENDS = ("skill", "api")

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "tekken-coach" / "config.toml"


@dataclass(frozen=True)
class Settings:
    """The resolved capture settings after flag/config/default precedence (docs/07 §1)."""

    mode: CaptureMode
    coach: str  # "skill" | "api"
    user_player: int | None  # 0 (p1) / 1 (p2); None until the user supplies it
    char: str | None  # the user's character; None until supplied (validated against reads, §5)
    out: Path  # where the session .jsonl is written


def _user_to_index(value: str | None) -> int | None:
    """Map a ``p1``/``p2`` string to a 0/1 player index (docs/01 §5). ``None`` passes through."""
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in ("p1", "1"):
        return 0
    if normalized in ("p2", "2"):
        return 1
    raise ValueError(f"invalid user side {value!r}: expected 'p1' or 'p2'")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    """Load the config TOML, or an empty mapping if it is absent (docs/07 §1 — missing is fine)."""
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _default_out(sessions_dir: str) -> Path:
    """The default session path ``<sessions_dir>/<timestamp>.jsonl`` (docs/07 §1.2)."""
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
    return Path(sessions_dir) / f"{stamp}.jsonl"


def resolve_settings(
    *,
    mode: str | None,
    coach: str | None,
    user: str | None,
    char: str | None,
    out: str | None,
    config: Mapping[str, object] | None = None,
) -> Settings:
    """Fold CLI flags over config over built-in defaults into a :class:`Settings` (docs/07 §1.2).

    Every argument is the CLI flag value (``None`` when the user did not pass it). ``config`` is the
    loaded ``config.toml`` mapping (``None``/empty ⇒ built-in defaults only). Invalid values raise
    ``ValueError`` so the CLI can report a clean error rather than capturing with a wrong setting.
    """
    cfg = config or {}

    def pick(flag: object | None, key: str) -> object | None:
        return flag if flag is not None else cfg.get(key)

    mode_value = pick(mode, "mode") or DEFAULT_MODE.value
    if mode_value not in (CaptureMode.live.value, CaptureMode.clean.value):
        raise ValueError(f"invalid mode {mode_value!r}: expected 'live' or 'clean'")

    coach_value = pick(coach, "coach") or DEFAULT_COACH
    if coach_value not in COACH_BACKENDS:
        raise ValueError(f"invalid coach backend {coach_value!r}: expected 'skill' or 'api'")

    user_value = _user_to_index(pick(user, "user"))  # type: ignore[arg-type]
    char_value = pick(char, "char")
    sessions_dir = str(cfg.get("sessions_dir", DEFAULT_SESSIONS_DIR))

    out_path = Path(out) if out is not None else _default_out(sessions_dir)

    return Settings(
        mode=CaptureMode(mode_value),
        coach=str(coach_value),
        user_player=user_value,
        char=str(char_value) if char_value is not None else None,
        out=out_path,
    )
