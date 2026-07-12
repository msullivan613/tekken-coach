"""The optional Claude API coaching backend (``--coach api``, docs/06 §3, docs/07 §2).

The default coaching path is the **Skill** (zero marginal cost, runs in Claude Code). This module
is the optional headless path: given a session ``.jsonl``, it calls the Claude API directly and
returns the between-match report (docs/06 §5), so ``tekken-coach coach --coach api`` can print
coaching without a manual Claude Code step.

Design (docs/06 §3):

* **Single source of truth.** The system prompt comes from ``prompt.build_rubric_and_instructions``
  — assembled from the ``skill/`` files, never a second copy of the domain content.
* **Model & request shape.** ``claude-opus-4-8`` (the plan fixes this — coaching is the
  quality-sensitive step), adaptive thinking on, effort ``high`` (aggregation + prioritization is
  exactly where thinking helps). The rubric/instructions are a **prompt-cached** stable system
  prefix; the (small) event-log text is the volatile user suffix after the cache breakpoint.
* **Auth.** The user's own credential (``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile) —
  the app never ships a key. If no credential is found, we do not crash: we return a result that
  points the user at how to set one and at the Skill path (docs/06 §3).

Offline-testable: the whole thing is driven through an injectable ``client_factory``. Tests pass a
**mocked** client (no network, no ``anthropic`` install needed); the ``anthropic`` package is
imported lazily inside the default factory only, so importing this module never requires it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from tekken_coach.coach.prompt import build_rubric_and_instructions, read_event_log_text

# The coaching reasoning is the quality-sensitive step; the plan fixes this model (docs/06 §3).
MODEL = "claude-opus-4-8"
# The report is short (docs/06 §5); 8000 leaves ample room for adaptive thinking + the report and
# stays under the SDK's non-streaming timeout guard, so a plain create() is fine.
MAX_TOKENS = 8000


class _Messages(Protocol):
    """The single client surface this backend uses: ``client.messages.create(...)``."""

    def create(self, **kwargs: Any) -> Any: ...


class _Client(Protocol):
    """Minimal structural type for the Anthropic client (real or mocked).

    ``messages`` is a read-only property (not a settable attribute) so it is *covariant*: a mock
    whose ``messages`` is a concrete subtype of :class:`_Messages` still satisfies the protocol.
    """

    @property
    def messages(self) -> _Messages: ...


ClientFactory = Callable[[], _Client]

# Backend tags on :class:`CoachResult`.
BACKEND_API = "api"
BACKEND_SKILL_FALLBACK = "skill_fallback"


@dataclass
class CoachResult:
    """The outcome of an API coaching run.

    ``backend`` is ``"api"`` on success or ``"skill_fallback"`` when no usable credential was
    found. On the API path ``report`` holds the coaching text (docs/06 §5); on the fallback path
    ``report`` is ``None`` and ``message`` explains how to authenticate or use the Skill instead.
    """

    backend: str
    report: str | None
    message: str


def _default_client_factory() -> _Client:
    """Construct the real Anthropic client using the user's own credential.

    ``anthropic`` is imported here (not at module scope) so importing :mod:`tekken_coach.coach`
    never requires the optional dependency. A bare ``Anthropic()`` resolves ``ANTHROPIC_API_KEY``
    or an ``ant auth login`` profile; with no credential at all it raises, which
    :func:`coach_session` turns into the graceful Skill-path fallback.
    """
    import anthropic

    return cast("_Client", anthropic.Anthropic())


def _is_auth_error(exc: BaseException) -> bool:
    """Whether ``exc`` is an Anthropic authentication/permission error (bad or missing key).

    Matched by class name + module so this stays importable without ``anthropic`` installed.
    """
    return type(exc).__module__.split(".", 1)[0] == "anthropic" and type(exc).__name__ in {
        "AuthenticationError",
        "PermissionDeniedError",
    }


def _no_credential_message(exc: BaseException) -> str:
    """The user-facing note when ``--coach api`` has no usable credential (docs/06 §3)."""
    return (
        f"No usable Claude API credential for --coach api ({type(exc).__name__}). "
        "Set ANTHROPIC_API_KEY or run `ant auth login`, then retry with --coach api. "
        "Or use the default Skill backend (no API key, no per-match cost): open this repo "
        "in Claude Code and run the tekken-coach skill on the session log."
    )


def _extract_text(response: Any) -> str:
    """Join the ``text`` content blocks of a Messages response, skipping thinking blocks."""
    blocks = getattr(response, "content", None) or []
    texts = [str(getattr(b, "text", "")) for b in blocks if getattr(b, "type", None) == "text"]
    return "".join(texts).strip()


def coach_session(
    log_path: str | Path,
    *,
    client_factory: ClientFactory | None = None,
    model: str = MODEL,
    max_tokens: int = MAX_TOKENS,
) -> CoachResult:
    """Coach one session ``.jsonl`` via the Claude API, or fall back to the Skill path.

    Assembles the system prompt from the ``skill/`` sources, reads (and version-gates) the event
    log, then calls the API with a prompt-cached rubric prefix and the log as the volatile suffix.
    Returns a :class:`CoachResult`: the report on success, or a fallback note if no credential is
    available (never crashes on missing auth — docs/06 §3). Genuine errors (a malformed log, an
    unexpected API failure) still propagate.
    """
    system_text = build_rubric_and_instructions()
    event_log = read_event_log_text(log_path)  # version-gated (docs/03 §6)

    factory = client_factory or _default_client_factory
    try:
        client = factory()
    except Exception as exc:  # any construction failure ⇒ no usable credential ⇒ Skill fallback
        return CoachResult(BACKEND_SKILL_FALLBACK, None, _no_credential_message(exc))

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            # Stable, prompt-cached system prefix (the rubric + instructions).
            system=[{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
            # Volatile user suffix after the cache breakpoint: the (small) event log.
            messages=[{"role": "user", "content": event_log}],
        )
    except Exception as exc:
        if _is_auth_error(exc):
            return CoachResult(BACKEND_SKILL_FALLBACK, None, _no_credential_message(exc))
        raise

    return CoachResult(
        BACKEND_API, _extract_text(response), f"Coached via the Claude API ({model})."
    )
