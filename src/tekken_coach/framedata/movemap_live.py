"""The interactive live movemap harness (the C6 ``map-moves --live`` path, brief #6 §B).

Stage B watches the reader's per-frame ``(char_id, move_id, move_frame)`` for the target character
and, when a **new** move-id is seen, captures its observed fingerprint from the exchange that
follows — its **startup** (``move_frame`` at contact) and, when the defender blocked, its
**on-block** advantage. It then shows the ranked Wavu candidates (the Stage-A :func:`join_move`
core) and lets the user confirm the mapping with one keypress, merging incrementally so a Ctrl-C
keeps progress (brief #6 §B).

The decision logic — contact/startup/on-block detection — lives in the pure, unit-tested
:class:`LiveFingerprinter`; only the endless read loop and the keypress prompt are I/O and carry
``# pragma: no cover`` (brief #6 §B: "all decision logic … is unit-tested").

Startup is the discriminator that the log-only miner lacks (:mod:`movemap_miner`), so live capture
is where ambiguous move-ids actually get resolved: a blocked exchange yields on-block *and* startup,
and startup breaks the on-block ties that collide the passive path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tekken_coach.framedata.loader import DEFAULT_FRAMEDATA_DIR, DEFAULT_MOVEMAP_DIR
from tekken_coach.framedata.movemap_build import MoveFingerprint, join_move
from tekken_coach.framedata.movemap_miner import merge_mappings
from tekken_coach.schemas import ActionState, PlayerFrame


@dataclass(frozen=True)
class FrameObservation:
    """The minimal per-frame slice the fingerprinter needs for one attacker's move (brief #6 §B).

    Derived from the two :class:`~tekken_coach.schemas.PlayerFrame`s each poll: the target
    attacker's move + posture and the defender's block/hit-stun. Kept tiny and decoupled from the
    reader so the fingerprinter is pure and testable frame-by-frame.
    """

    attacker_char_id: int
    attacker_move_id: int
    attacker_move_frame: int
    attacker_recovering: bool  # attacker still in its move (attack/recovery), not yet actionable
    defender_block_stun: bool
    defender_hit_stun: bool


@dataclass(frozen=True)
class LiveObservation:
    """A completed observation of one move: fingerprint plus how it made contact (brief #6 §B)."""

    fingerprint: MoveFingerprint
    contacted: bool  # did the move actually connect (block or hit) — else startup is unknown
    blocked: bool  # was the connect a block (=> on_block is meaningful)


def _actionable(state: ActionState) -> bool:
    """True when a player has recovered and can act again (out of attack/recovery/stun)."""
    return state not in (
        ActionState.attack,
        ActionState.recovery,
        ActionState.blockstun,
        ActionState.hitstun,
        ActionState.stagger,
    )


class LiveFingerprinter:
    """Detect one move's startup + on-block from a live frame stream (brief #6 §B, pure).

    Fed :class:`FrameObservation`s in order via :meth:`feed`, it tracks the target attacker's
    current move and returns a :class:`LiveObservation` on the frame the exchange resolves:

    * **contact** — the first frame the defender enters block- or hit-stun while the move is active;
      ``startup`` is the attacker ``move_frame`` at that instant.
    * **on_block** — only when the contact was a block: the advantage is
      ``(defender-actionable frame) - (attacker-actionable frame)`` measured from contact, i.e.
      positive when the attacker recovers first. Emitted once both have recovered.

    A whiff (the move changes away before any contact) resets silently — no observation. The pure
    logic is unit-tested; the live loop that produces the observations is ``# pragma: no cover``.
    """

    def __init__(self, attacker_char_id: int) -> None:
        self._char_id = attacker_char_id
        self._reset()

    def _reset(self) -> None:
        self._move_id: int | None = None
        self._contact_frame: int | None = None  # attacker move_frame at contact
        self._blocked = False
        self._attacker_recovered_at: int | None = None  # frames since contact
        self._defender_recovered_at: int | None = None
        self._since_contact = 0

    def feed(self, obs: FrameObservation) -> LiveObservation | None:
        """Advance the tracker one frame; return a completed observation on the resolving frame."""
        if obs.attacker_char_id != self._char_id:
            self._reset()
            return None

        if obs.attacker_move_id != self._move_id:
            # A new move started (or the first one). Emit nothing; begin tracking it.
            self._reset()
            self._move_id = obs.attacker_move_id

        if self._contact_frame is None:
            if obs.defender_block_stun or obs.defender_hit_stun:
                self._contact_frame = obs.attacker_move_frame
                self._blocked = obs.defender_block_stun
                self._since_contact = 0
                if not self._blocked:
                    # A hit gives startup but not a meaningful on-block reading.
                    return self._emit(obs.attacker_move_id, on_block=None)
            return None

        # Post-contact (blocked): measure when each side becomes actionable again.
        self._since_contact += 1
        if self._attacker_recovered_at is None and not obs.attacker_recovering:
            self._attacker_recovered_at = self._since_contact
        if self._defender_recovered_at is None and not obs.defender_block_stun:
            self._defender_recovered_at = self._since_contact
        if self._attacker_recovered_at is not None and self._defender_recovered_at is not None:
            on_block = self._defender_recovered_at - self._attacker_recovered_at
            return self._emit(obs.attacker_move_id, on_block=on_block)
        return None

    def _emit(self, move_id: int, *, on_block: int | None) -> LiveObservation:
        """Build the observation for the resolved move and reset for the next one."""
        startup = self._contact_frame
        blocked = self._blocked
        contacted = startup is not None
        fingerprint = MoveFingerprint(
            char_id=self._char_id,
            move_id=move_id,
            on_block=on_block,
            startup=startup,
            blocked_samples=1 if blocked and on_block is not None else 0,
            total_samples=1,
        )
        self._reset()
        return LiveObservation(fingerprint=fingerprint, contacted=contacted, blocked=blocked)


def observation_from_frames(attacker: PlayerFrame, defender: PlayerFrame) -> FrameObservation:
    """Project the two player frames into the fingerprinter's per-frame input (brief #6 §B)."""
    return FrameObservation(
        attacker_char_id=attacker.char_id,
        attacker_move_id=attacker.move_id,
        attacker_move_frame=attacker.move_frame,
        attacker_recovering=not _actionable(attacker.action_state),
        defender_block_stun=defender.block_stun or defender.action_state is ActionState.blockstun,
        defender_hit_stun=defender.hit_stun or defender.action_state is ActionState.hitstun,
    )


# ---------------------------------------------------------------------------
# The interactive live harness (I/O; the decision core above is unit-tested)
# ---------------------------------------------------------------------------


def run_live(
    *,
    char: str,
    user_player: int,
    process: str,
    offsets_dir: str,
    movemap_dir: str | Path = DEFAULT_MOVEMAP_DIR,
    framedata_dir: str | Path = DEFAULT_FRAMEDATA_DIR,
    version_override: str | None = None,
    overwrite: bool = False,
    interval: float = 0.05,
) -> int:  # pragma: no cover - endless live loop + keypress prompt; LiveFingerprinter is tested
    """Watch the user's character live, prompt to confirm each new move-id's mapping (brief #6 §B).

    Attaches read-only, decodes both players each poll, feeds the target attacker's frames to a
    :class:`LiveFingerprinter`, and on each newly-observed move-id shows the ranked Wavu candidates
    and asks for a one-key confirm. Each confirm merges immediately (:func:`merge_mappings`), so a
    Ctrl-C keeps every mapping made so far.
    """
    import sys
    import time

    from tekken_coach.framedata.loader import load_current_framedata
    from tekken_coach.reader.decode import MemoryReadError, decode_frame
    from tekken_coach.reader.faults import ReaderError
    from tekken_coach.reader.offsets import select_offset_table
    from tekken_coach.reader.version import detect_running_version
    from tekken_coach.reader.win_source import WinMemorySource

    slug = char.lower()
    snapshot = load_current_framedata(framedata_dir)
    char_fd = snapshot.get_char(slug)
    if char_fd is None:
        print(
            f"error: no frame-data snapshot for {slug!r} — run `fetch-framedata {char}` first.",
            file=sys.stderr,
        )
        return 1
    game_version = snapshot.manifest.game_version or "unknown"

    try:
        source = WinMemorySource(process)
        version = version_override or detect_running_version(process)
        table = select_offset_table(version, offsets_dir)
    except ReaderError as exc:
        from tekken_coach.reader.commands import _report_fault

        return _report_fault(exc)

    print(f"map-moves --live: mapping {char} (P{user_player + 1}) on {version} — Ctrl-C to stop")
    print("perform each move on block; confirm the matched notation with Enter, or 's' to skip.\n")

    fingerprinter: LiveFingerprinter | None = None
    seen: set[int] = set()
    mapped: dict[int, str] = {}

    try:
        while True:
            try:
                frame = decode_frame(source, table)
            except MemoryReadError:
                time.sleep(interval)
                continue
            attacker = frame.players[user_player]
            defender = frame.players[1 - user_player]
            if fingerprinter is None or fingerprinter._char_id != attacker.char_id:
                fingerprinter = LiveFingerprinter(attacker.char_id)
            obs = fingerprinter.feed(observation_from_frames(attacker, defender))
            if obs is not None and obs.contacted and obs.fingerprint.move_id not in seen:
                seen.add(obs.fingerprint.move_id)
                chosen = _prompt_confirm(obs, char_fd)
                if chosen is not None:
                    mapped[obs.fingerprint.move_id] = chosen
                    merge_mappings(
                        slug,
                        char_fd,
                        game_version,
                        [(obs.fingerprint.move_id, chosen)],
                        char_id=attacker.char_id,
                        movemap_dir=movemap_dir,
                        overwrite=overwrite,
                    )
                    print(f"  ✓ wrote {obs.fingerprint.move_id} -> {chosen}\n")
            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\nstopped — {len(mapped)} move-id(s) mapped this session.")
    return 0


def _prompt_confirm(obs: LiveObservation, char_fd: object) -> str | None:  # pragma: no cover - I/O
    """Show ranked candidates for an observed move and read a one-key confirm (brief #6 §B)."""
    from tekken_coach.framedata.models import CharFrameData

    assert isinstance(char_fd, CharFrameData)
    fp = obs.fingerprint
    result = join_move(fp, char_fd)
    detail = f"startup≈{fp.startup}" + (
        f" on_block≈{fp.on_block:+d}" if fp.on_block is not None else " (hit; no on_block)"
    )
    print(f"move_id {fp.move_id}: {detail}")
    if not result.candidates:
        print(f"  no candidate ({result.reason}); skipping.\n")
        return None
    for i, cand in enumerate(result.candidates[:9], start=1):
        tag = " <- top" if i == 1 else ""
        print(f"  [{i}] {cand.framedata_key}  (i{cand.startup}, {cand.on_block:+d} on block){tag}")
    top = result.candidates[0].framedata_key
    answer = input(f"  confirm [{top}]? Enter=yes / 1-9=pick / s=skip: ").strip().lower()
    if answer in ("", "y", "1"):
        return top
    if answer == "s":
        return None
    if answer.isdigit() and 1 <= int(answer) <= len(result.candidates[:9]):
        return result.candidates[int(answer) - 1].framedata_key
    return None
