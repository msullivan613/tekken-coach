"""``map-moves --audit <session.jsonl>`` — observed-vs-canonical drift alarm (brief #8 Layer 2).

The docs/05 §4.2 reconciliation, repurposed as a pure, automatic validator. For each **mapped**
``(char, move_id)`` that actually shows up (blocked) in a session log, it distils the consensus
*observed* on-block from behaviour and compares it to the *canonical* on-block of the notation the
movemap bound it to. A consistent multi-sample gap is the alarm: the id no longer means what the map
says (a **mis-map**), or the snapshot has drifted from the live build (a **stale snapshot**).

Read-only QA — nothing is written, nothing is auto-fixed (a human / Stage-B decides, brief #8 out of
scope). It reuses the exact machinery the builder trusts: header-driven character resolution
(:func:`movemap_miner.resolve_char_ids`) and the same consensus fingerprint
(:func:`movemap_build.build_fingerprint`), so "observed" here means the same thing it means when a
mapping is written.

**Tolerance (``tol = 2``).** A memory poll at ~60 Hz can read an advantage a frame early or late,
and Wavu's published on-block can itself round a frame against the live value; the modal consensus
already absorbs one-off noise, so a ±2 band means a lone single-frame disagreement never fires while
a consistent ≥3-frame gap does. It is deliberately *looser* than the join's ±1 (§6): the join's ±1
is tuned to avoid *admitting* a wrong candidate; here we are tuned to avoid a *false drift alarm* on
a mapping that is fine.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from tekken_coach.framedata.anchors import Anchors
from tekken_coach.framedata.models import CharMoveMap, FrameDataSnapshot
from tekken_coach.framedata.movemap_build import build_fingerprint
from tekken_coach.framedata.movemap_miner import resolve_char_ids
from tekken_coach.schemas import Interaction
from tekken_coach.session.store import LoadedSession

# See the module docstring for the justification. A gap of exactly the tolerance is *not* drift;
# only a strictly larger gap fires.
DEFAULT_DRIFT_TOL = 2

# A gap wider than this reads as a different move entirely (mis-map) rather than a balance tweak; at
# or below it, a consistent small shift reads as a snapshot that lagged a patch. Only a hint.
_MISMAP_GAP = 6

# Finding kinds.
KIND_CONSISTENT = "consistent"
KIND_DRIFT = "drift"
KIND_NO_SIGNAL = "no_signal"  # observed, but no usable blocked consensus to compare


@dataclass(frozen=True)
class ObservedCheck:
    """One mapped ``move_id`` observed in the log, checked against its canonical on-block."""

    char_slug: str
    move_id: int
    notation: str
    framedata_key: str
    observed_on_block: int | None  # modal blocked consensus; None => no usable signal
    canonical_on_block: (
        int | None
    )  # the mapped notation's snapshot on_block; None => broken/no value
    blocked_samples: int
    total_samples: int
    is_anchor: bool
    kind: str  # KIND_CONSISTENT | KIND_DRIFT | KIND_NO_SIGNAL
    delta: int | None  # |observed - canonical|, when both are known

    @property
    def likely(self) -> str | None:
        """For a drift finding, the likelier cause from the gap size (a hint, not a verdict)."""
        if self.kind != KIND_DRIFT or self.delta is None:
            return None
        if self.delta > _MISMAP_GAP:
            return "mis-map (gap too large for a balance tweak — likely a different move)"
        return "stale snapshot (small consistent shift — a patch likely adjusted the frames)"


@dataclass(frozen=True)
class Unobserved:
    """A mapped ``move_id`` that never appears blocked in the log — can't be checked (brief #8)."""

    char_slug: str
    move_id: int
    notation: str
    framedata_key: str
    is_anchor: bool


@dataclass(frozen=True)
class AuditReport:
    """The full drift audit (pure — read-only QA, nothing written)."""

    checks: list[ObservedCheck]
    unobserved: list[Unobserved]
    tol: int

    @property
    def consistent(self) -> list[ObservedCheck]:
        return [c for c in self.checks if c.kind == KIND_CONSISTENT]

    @property
    def drift(self) -> list[ObservedCheck]:
        """Drift findings, ranked most-suspect first: biggest gap, then most samples."""
        drift = [c for c in self.checks if c.kind == KIND_DRIFT]
        return sorted(drift, key=lambda c: (-(c.delta or 0), -c.blocked_samples, c.move_id))

    @property
    def no_signal(self) -> list[ObservedCheck]:
        return [c for c in self.checks if c.kind == KIND_NO_SIGNAL]


def audit_session(
    session: LoadedSession,
    snapshot: FrameDataSnapshot,
    move_maps: dict[str, CharMoveMap],
    anchors: Anchors,
    *,
    only_char: str | None = None,
    tol: int = DEFAULT_DRIFT_TOL,
) -> AuditReport:
    """Compare every mapped, observed ``move_id`` to its canonical on-block (brief #8 Layer 2).

    ``move_maps`` is keyed by ``char_name`` (as :func:`loader.load_move_maps` returns it). Pure and
    read-only: builds no files, changes nothing. ``only_char`` (case-insensitive, on slug) narrows
    the audit to one character.
    """
    names = resolve_char_ids(session.header, session.interactions)
    slug_to_char_id = {name.lower(): char_id for char_id, name in names.items()}

    grouped: dict[tuple[int | None, int], list[Interaction]] = defaultdict(list)
    for interaction in session.interactions:
        grouped[(interaction.attacker_char_id, interaction.attacker_move_id)].append(interaction)

    want = only_char.lower() if only_char else None
    checks: list[ObservedCheck] = []
    unobserved: list[Unobserved] = []

    for move_map in sorted(move_maps.values(), key=lambda m: m.char_name.lower()):
        slug = move_map.char_name.lower()
        if want is not None and slug != want:
            continue
        char_fd = snapshot.get_char(slug)
        char_id = slug_to_char_id.get(slug)
        anchor_keys = anchors.for_char(slug)

        for move_id_str, entry in sorted(move_map.moves.items(), key=lambda kv: int(kv[0])):
            move_id = int(move_id_str)
            is_anchor = move_id in anchor_keys
            move = char_fd.get(entry.framedata_key) if char_fd is not None else None
            canonical = move.on_block if move is not None else None

            items = grouped.get((char_id, move_id)) if char_id is not None else None
            if not items:
                unobserved.append(
                    Unobserved(
                        char_slug=slug,
                        move_id=move_id,
                        notation=entry.notation,
                        framedata_key=entry.framedata_key,
                        is_anchor=is_anchor,
                    )
                )
                continue

            fp = build_fingerprint(char_id if char_id is not None else -1, move_id, items)
            observed = fp.on_block
            if observed is None or canonical is None:
                kind, delta = KIND_NO_SIGNAL, None
            else:
                delta = abs(observed - canonical)
                kind = KIND_DRIFT if delta > tol else KIND_CONSISTENT
            checks.append(
                ObservedCheck(
                    char_slug=slug,
                    move_id=move_id,
                    notation=entry.notation,
                    framedata_key=entry.framedata_key,
                    observed_on_block=observed,
                    canonical_on_block=canonical,
                    blocked_samples=fp.blocked_samples,
                    total_samples=fp.total_samples,
                    is_anchor=is_anchor,
                    kind=kind,
                    delta=delta,
                )
            )

    return AuditReport(checks=checks, unobserved=unobserved, tol=tol)


# ---------------------------------------------------------------------------
# Human-readable rendering (pure; the CLI just prints these lines)
# ---------------------------------------------------------------------------


def format_audit(report: AuditReport) -> list[str]:
    """Render an :class:`AuditReport` as printable lines (brief #8 Layer 2 summary)."""
    lines: list[str] = []
    lines.append(
        f"map-moves --audit (tol ±{report.tol}): "
        f"{len(report.consistent)} consistent, {len(report.drift)} drift, "
        f"{len(report.no_signal)} observed-no-signal, {len(report.unobserved)} unobserved"
    )

    if report.drift:
        lines.append("")
        lines.append("DRIFT — observed on-block disagrees with the mapped notation (ranked):")
        for c in report.drift:
            anchor = " [anchor]" if c.is_anchor else ""
            lines.append(
                f"  [{c.char_slug}] {c.move_id} -> {c.notation}{anchor}: "
                f"observed {c.observed_on_block:+d} vs canonical {c.canonical_on_block:+d} "
                f"(Δ{c.delta}, {c.blocked_samples} sample(s)) — {c.likely}"
            )

    if report.no_signal:
        lines.append("")
        lines.append(
            "observed but not comparable (no blocked consensus, or no canonical on_block):"
        )
        for c in report.no_signal:
            lines.append(
                f"  [{c.char_slug}] {c.move_id} -> {c.notation} ({c.total_samples} sample(s))"
            )

    if report.unobserved:
        lines.append("")
        lines.append(f"mapped but unobserved in this log — can't check ({len(report.unobserved)}):")
        for u in report.unobserved:
            anchor = " [anchor]" if u.is_anchor else ""
            lines.append(f"  [{u.char_slug}] {u.move_id} -> {u.notation}{anchor}")

    if not report.drift:
        lines.append("")
        lines.append(
            "no drift — every mapped, observed move-id agrees with its canonical on-block."
        )
    return lines
