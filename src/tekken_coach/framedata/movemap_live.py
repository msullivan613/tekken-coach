"""The interactive live movemap harness (the C6 ``map-moves --live`` path, brief #6 §B).

Stage B watches the reader's per-frame ``(char_id, move_id, move_frame)`` for the target character
and, when a **new** move-id is seen, captures its observed fingerprint from the exchange that
follows — its **startup** (``move_frame`` at contact) and, when the defender blocked, its
**on-block** advantage. It then shows the ranked Wavu candidates — matched by *startup*, the
reliable signal (:func:`join_move_live`, brief #14), not by the fuzzy live on-block — and lets the
user confirm the mapping with one keypress, merging incrementally so a Ctrl-C keeps progress (§B).

The decision logic — contact/startup/on-block detection — lives in the pure, unit-tested
:class:`LiveFingerprinter`; only the endless read loop and the keypress prompt are I/O and carry
``# pragma: no cover`` (brief #6 §B: "all decision logic … is unit-tested").

Startup is the discriminator that the log-only miner lacks (:mod:`movemap_miner`), so live capture
is where ambiguous move-ids actually get resolved: a blocked exchange yields on-block *and* startup,
and startup breaks the on-block ties that collide the passive path.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from tekken_coach.framedata.loader import (
    DEFAULT_FRAMEDATA_DIR,
    DEFAULT_MOVEMAP_DIR,
    load_char_move_map,
)
from tekken_coach.framedata.movemap_build import MoveFingerprint, join_move_live
from tekken_coach.framedata.movemap_miner import merge_mappings
from tekken_coach.schemas import ActionState, PlayerFrame

# How many observations of one move-id the live loop gathers before it reduces and prompts. The user
# is told to do each move ~5x on block (brief #13 §B), so a single poll-jitter rep never decides the
# reading: startup is reduced by MIN (its noise is one-sided high) and on-block by the mode (its
# noise is two-sided), which needs a few reps to converge.
DEFAULT_LIVE_REPS = 5

# How many ranked candidates the confirm prompt lists. A character can carry a dozen-plus moves at
# one startup (Bryan has many i15s), so the window is wider than the old 9: startup-primary matching
# only helps if the true move is actually *reachable* in the list, not ranked out of sight (#14).
MAX_CANDIDATES_SHOWN = 15


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
    # Shared per-round game-frame clock (``frames_since_round_start``, mirrored on both players and
    # ticking at 60 fps). When present, on-block advantage is measured in game frames — the precise
    # unit — instead of ~20 Hz poll counts, which are ~3x under-resolved (brief #12 §4). ``None`` if
    # the frame counter is unavailable, in which case the fingerprinter falls back to poll counts.
    frame_clock: int | None = None


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
    """Detect one move's startup + on-block from a live frame stream (brief #6 §B, #12, pure).

    Fed :class:`FrameObservation`s in order via :meth:`feed`, it tracks the target attacker's
    current *attack* and returns a :class:`LiveObservation` on the frame the exchange resolves.

    A move is the span where the **attacker is in an attack** — bounded by neutral, not by raw
    ``move_id`` changes (brief #12). The attacker being **actionable** (``attacker_recovering`` is
    ``False``) *is* neutral, and a defender still in block-stun while the attacker is actionable is
    the *previous* move's lingering block-stun, never a new contact. Concretely:

    * **no tracking on neutral** — a move is "active" only once the attacker is in an attack; an
      actionable attacker never starts a move nor registers a contact (brief #12 §1).
    * **contact** — the first frame the defender enters block-/hit-stun while a move is active;
      ``startup`` is the attacker ``move_frame`` and the move identity is the ``move_id`` *at that
      instant* (a move may carry a 1-frame sub-id on the way in — the contact-frame id is the one
      that matters, brief #12 §3).
    * **on_block** — only when the contact was a block: ``(defender-actionable clock) -
      (attacker-actionable clock)``, positive when the attacker recovers first. The attacker
      returning to neutral is the "attacker recovered" signal — it *finalizes*, never discards, a
      pending measurement (brief #12 §2); the observation is emitted once the defender also leaves
      block-stun. When :attr:`FrameObservation.frame_clock` is present the two clocks are game
      frames (60 fps); otherwise they fall back to poll counts (brief #12 §4).

    A whiff (the attacker returns to neutral before any contact) resets silently — no observation.
    The pure logic is unit-tested; the live loop that produces the observations is
    ``# pragma: no cover``.
    """

    def __init__(self, attacker_char_id: int) -> None:
        self._char_id = attacker_char_id
        self._poll = 0
        self._reset()

    def _reset(self) -> None:
        self._active = False  # is a move currently being tracked (attacker in an attack)?
        self._contact_frame: int | None = None  # attacker move_frame at contact (= startup)
        self._contact_move_id: int | None = None  # move identity sampled at the contact frame
        self._blocked = False
        self._attacker_recovered_clock: int | None = None
        self._defender_recovered_clock: int | None = None

    def feed(self, obs: FrameObservation) -> LiveObservation | None:
        """Advance the tracker one frame; return a completed observation on the resolving frame."""
        self._poll += 1
        if obs.attacker_char_id != self._char_id:
            self._reset()
            return None

        clock = obs.frame_clock if obs.frame_clock is not None else self._poll
        # The attacker is still in a move (attack/recovery), i.e. not neutral/actionable.
        in_attack = obs.attacker_recovering

        if not self._active:
            # Idle: never track or contact on a neutral/actionable attacker (brief #12 §1). Begin
            # tracking only when the attacker is actually in an attack.
            if in_attack:
                self._active = True
            else:
                return None

        if self._contact_frame is None:
            # Pre-contact. If the attacker returns to neutral first, the move whiffed — reset.
            if not in_attack:
                self._reset()
                return None
            if obs.defender_block_stun or obs.defender_hit_stun:
                # Contact: sample startup AND identity here — the id live at contact is the move
                # (any earlier 1-frame sub-id is discarded), brief #12 §3.
                self._contact_frame = obs.attacker_move_frame
                self._contact_move_id = obs.attacker_move_id
                self._blocked = obs.defender_block_stun
                if not self._blocked:
                    # A hit gives startup but not a meaningful on-block reading.
                    return self._emit(on_block=None)
            return None

        # Post-contact (blocked): record when each side becomes actionable again, in clock units.
        # The attacker going neutral finalizes (never discards) the pending measurement (brief #12).
        if self._attacker_recovered_clock is None and not in_attack:
            self._attacker_recovered_clock = clock
        if self._defender_recovered_clock is None and not obs.defender_block_stun:
            self._defender_recovered_clock = clock
        if (
            self._attacker_recovered_clock is not None
            and self._defender_recovered_clock is not None
        ):
            on_block = self._defender_recovered_clock - self._attacker_recovered_clock
            return self._emit(on_block=on_block)
        return None

    def _emit(self, *, on_block: int | None) -> LiveObservation:
        """Build the observation for the resolved move and reset for the next one."""
        startup = self._contact_frame
        move_id = self._contact_move_id
        blocked = self._blocked
        contacted = startup is not None
        assert move_id is not None  # _emit is only reached after a contact set _contact_move_id
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


# ---------------------------------------------------------------------------
# Multi-rep reduction (pure; unit-tested) — brief #13 §B
# ---------------------------------------------------------------------------
#
# A single live rep is ±~1 frame even at 120 Hz, and its two components carry *different-shaped*
# noise, so they must be reduced differently:
#
# * **startup** is detected on the first poll *after* the defender enters block-stun, so an observed
#   startup never lands below the true value — the noise is **one-sided high**. The MIN across reps
#   therefore approaches the truth and never undershoots; a mean/median would sit biased high.
# * **on_block** is the difference of two *independently* late-sampled recovery instants, so its
#   noise is **two-sided** (±). The MODAL value across reps converges on the truth.
#
# The reducer below is pure, so it is unit-tested; the live loop only decides *when* to reduce.


def _reduced_startup(startups: list[int]) -> int | None:
    """MIN startup across reps — detection is always late, so the min approaches truth (#13)."""
    return min(startups) if startups else None


def _reduced_on_block(values: list[int]) -> int | None:
    """MODAL on-block across reps, median-tiebroken (brief #13 §B).

    On-block noise is two-sided, so the mode converges on the true value. A tie between
    equally-frequent values is broken by the (lower) median of those tied modes — the centre of the
    symmetric jitter — rather than discarded as "no consensus": the user has already performed the
    move several times and expects a reading, unlike the log path where a tie is truly ambiguous.
    """
    if not values:
        return None
    counts = Counter(values)
    top = max(counts.values())
    modes = sorted(v for v, c in counts.items() if c == top)
    return modes[(len(modes) - 1) // 2]


def reduce_observations(observations: list[LiveObservation]) -> MoveFingerprint:
    """Reduce repeated observations of one move-id to a consensus fingerprint (brief #13 §B).

    Pure: given per-rep :class:`LiveObservation`s of the *same* move-id, combine them into one
    :class:`MoveFingerprint` whose ``startup`` is the min over contacted reps (one-sided-high noise)
    and whose ``on_block`` is the modal blocked advantage (two-sided noise). ``blocked_samples`` /
    ``total_samples`` carry the rep counts the join already understands. The reduced fingerprint is
    what feeds :func:`join_move_live` (startup-primary, brief #14), so accurate input surfaces the
    true candidate.
    """
    if not observations:
        raise ValueError("reduce_observations needs at least one observation")
    first = observations[0].fingerprint
    startups = [
        o.fingerprint.startup
        for o in observations
        if o.contacted and o.fingerprint.startup is not None
    ]
    blocked_advs = [
        o.fingerprint.on_block
        for o in observations
        if o.blocked and o.fingerprint.on_block is not None
    ]
    return MoveFingerprint(
        char_id=first.char_id,
        move_id=first.move_id,
        on_block=_reduced_on_block(blocked_advs),
        startup=_reduced_startup(startups),
        blocked_samples=len(blocked_advs),
        total_samples=len(observations),
    )


class MoveReducer:
    """Accumulate live observations per move-id and reduce them on demand (brief #13 §B, pure).

    The live loop feeds every contacted observation via :meth:`add`; once a move-id reaches ``reps``
    samples it :meth:`is_ready` and the loop reduces (:meth:`reduce`) and prompts. On Ctrl-C the
    loop flushes whatever partial accumulations remain (:meth:`pending`) so the reps are not lost.
    Pure (no I/O) and unit-tested; the loop only decides *when* to prompt (``# pragma: no cover``).
    """

    def __init__(self, reps: int = DEFAULT_LIVE_REPS) -> None:
        self._reps = max(1, reps)
        self._obs: dict[int, list[LiveObservation]] = {}

    def add(self, observation: LiveObservation) -> None:
        """Record one contacted observation under its move-id (first-seen order preserved)."""
        self._obs.setdefault(observation.fingerprint.move_id, []).append(observation)

    def count(self, move_id: int) -> int:
        """How many observations have been gathered for ``move_id`` so far."""
        return len(self._obs.get(move_id, []))

    def is_ready(self, move_id: int) -> bool:
        """True once ``reps`` observations of ``move_id`` are gathered — time to reduce + prompt."""
        return self.count(move_id) >= self._reps

    def reduce(self, move_id: int) -> MoveFingerprint:
        """The consensus fingerprint over every observation gathered for ``move_id``."""
        return reduce_observations(self._obs[move_id])

    def pending(self, *, min_reps: int = 1) -> list[int]:
        """Move-ids with at least ``min_reps`` samples, first-seen order (the Ctrl-C flush set)."""
        return [mid for mid, obs in self._obs.items() if len(obs) >= min_reps]


class PollMeter:
    """Measure the achieved live poll rate — a correctness property, not a vanity metric (#13).

    ``map-moves --live`` must sample every game frame (~60 fps) or startup/on-block re-acquire the
    ~3-frame jitter this brief set out to remove. The loop targets ~120 Hz, but a target it cannot
    sustain silently re-introduces that jitter, so the run measures the real poll-to-poll cadence
    and prints it (like brief #11's ``PollRate`` heartbeat). Pure arithmetic (fed poll-to-poll
    deltas) so it is unit-tested; the loop that feeds it is ``# pragma: no cover``.
    """

    def __init__(self) -> None:
        self._intervals = 0
        self._elapsed = 0.0

    def record(self, dt: float) -> None:
        """Record one poll-to-poll interval in seconds; non-positive gaps are ignored."""
        if dt > 0:
            self._intervals += 1
            self._elapsed += dt

    @property
    def polls(self) -> int:
        """Number of measured poll-to-poll intervals (≈ polls − 1)."""
        return self._intervals

    @property
    def hz(self) -> float:
        """Observed polls per second (0.0 before the first interval is recorded)."""
        return self._intervals / self._elapsed if self._elapsed > 0 else 0.0

    def summary(self, target_hz: float) -> str:
        """The heartbeat / end-of-run line: achieved rate against the requested target."""
        target = f"target {target_hz:.0f} Hz" if target_hz > 0 else "no cap"
        return f"poll rate: {self.hz:.0f} Hz ({target}) over {self._intervals} polls"


def observation_from_frames(attacker: PlayerFrame, defender: PlayerFrame) -> FrameObservation:
    """Project the two player frames into the fingerprinter's per-frame input (brief #6 §B)."""
    return FrameObservation(
        attacker_char_id=attacker.char_id,
        attacker_move_id=attacker.move_id,
        attacker_move_frame=attacker.move_frame,
        attacker_recovering=not _actionable(attacker.action_state),
        defender_block_stun=defender.block_stun or defender.action_state is ActionState.blockstun,
        defender_hit_stun=defender.hit_stun or defender.action_state is ActionState.hitstun,
        # Shared per-round game-frame clock (mirrored on both structs); measures on-block precisely.
        frame_clock=attacker.frames_since_round_start,
    )


# ---------------------------------------------------------------------------
# The interactive live harness (I/O; the decision core above is unit-tested)
# ---------------------------------------------------------------------------


HEARTBEAT_SECONDS = 3.0  # how often the live loop prints its achieved poll rate while watching


def already_mapped_ids(movemap_dir: str | Path, slug: str, *, overwrite: bool) -> set[int]:
    """Move_ids to skip at the start of a live session: the ones already on disk (brief #16).

    A grind spans many sessions, so a fresh run must not re-prompt moves already committed to
    ``<movemap_dir>/<slug>.json`` — the skip-set is pre-seeded with them. Empty when the character
    has no movemap yet (first-ever session — a missing file is not an error), or when ``overwrite``
    is set (the user explicitly wants every detected move re-mapped, so nothing is pre-skipped).

    The on-disk keys are the game ``move_id`` as JSON **strings** (:class:`CharMoveMap`); they are
    converted to ``int`` to match the live fingerprinter's ids. Pure and unit-tested; the live loop
    that consumes the set is ``# pragma: no cover``.
    """
    if overwrite:
        return set()
    path = Path(movemap_dir) / f"{slug}.json"
    if not path.exists():
        return set()
    return {int(move_id) for move_id in load_char_move_map(path).moves}


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
    interval: float = 1.0 / 120,
    reps: int = DEFAULT_LIVE_REPS,
) -> int:  # pragma: no cover - endless live loop + keypress prompt; the decision core is tested
    """Watch the user's character live, prompt to confirm each new move-id's mapping (brief #6 §B).

    Attaches read-only, decodes both players each poll, feeds the target attacker's frames to a
    :class:`LiveFingerprinter`, and gathers ``reps`` observations of each new move-id before showing
    the ranked Wavu candidates and asking for a one-key confirm. Each confirm merges immediately
    (:func:`merge_mappings`), so a Ctrl-C keeps every mapping made so far.

    Two levers keep the fed numbers frame-accurate (#13). **Part A** paces the loop to a target
    *period* (``interval`` seconds, ~120 Hz default) — subtracting the work already done each iter,
    not sleeping a flat extra ``interval`` — so ``move_frame`` advances ≤1 per poll, and it measures
    and prints the *achieved* Hz (a rate it can't sustain silently re-introduces the jitter). **Part
    B** reduces the ``reps`` observations per move to (min startup, modal on-block) before the join,
    so a single poll's ±1 jitter never decides the reading.
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

    target_hz = (1.0 / interval) if interval > 0 else 0.0
    print(f"map-moves --live: mapping {char} (P{user_player + 1}) on {version} — Ctrl-C to stop")
    print(f"perform each move on block ~{reps}x; confirm the matched notation with Enter, or 's'.")
    if target_hz > 0:
        print(
            f"polling at ~{target_hz:.0f} Hz (--hz to tune) — the achieved rate is printed live.\n"
        )
    else:
        print("polling as fast as reads allow — the achieved rate is printed live.\n")

    fingerprinter: LiveFingerprinter | None = None
    reducer = MoveReducer(reps)
    meter = PollMeter()
    # Pre-seed the skip-set from what's already committed so a multi-session grind never re-prompts
    # a mapped move (brief #16). --overwrite clears the seed: the user wants every detected move
    # back on the table.
    prompted = already_mapped_ids(movemap_dir, slug, overwrite=overwrite)
    if overwrite:
        print(f"{slug}: --overwrite set — re-mapping all detected moves")
    else:
        print(
            f"resuming {slug}: {len(prompted)} already mapped — "
            f"skipping (use --overwrite to re-map)"
        )
    mapped: dict[int, str] = {}

    def _confirm_and_merge(fingerprint: MoveFingerprint, char_id: int) -> None:
        """Prompt for the reduced fingerprint and merge on confirm (merge-on-confirm contract)."""
        chosen = _prompt_confirm(fingerprint, char_fd, reps=reducer.count(fingerprint.move_id))
        if chosen is not None:
            mapped[fingerprint.move_id] = chosen
            merge_mappings(
                slug,
                char_fd,
                game_version,
                [(fingerprint.move_id, chosen)],
                char_id=char_id,
                movemap_dir=movemap_dir,
                overwrite=overwrite,
            )
            print(f"  ✓ wrote {fingerprint.move_id} -> {chosen}\n")

    last_poll: float | None = (
        None  # start of the previous successful poll (None resets the cadence)
    )
    last_beat = time.monotonic()
    next_poll = time.monotonic()  # the target start time of the next poll (period-paced, Part A)

    try:
        while True:
            iter_start = time.monotonic()
            try:
                frame = decode_frame(source, table)
            except MemoryReadError:
                # Can't read yet (menu/load) — pace and retry; this is not a real poll, so it must
                # not enter the rate (and it breaks the poll-to-poll cadence).
                last_poll = None
                if interval > 0:
                    time.sleep(interval)
                next_poll = time.monotonic()
                continue
            if last_poll is not None:
                meter.record(iter_start - last_poll)
            last_poll = iter_start

            attacker = frame.players[user_player]
            defender = frame.players[1 - user_player]
            if fingerprinter is None or fingerprinter._char_id != attacker.char_id:
                fingerprinter = LiveFingerprinter(attacker.char_id)
            obs = fingerprinter.feed(observation_from_frames(attacker, defender))

            prompted_now = False
            if obs is not None and obs.contacted and obs.fingerprint.move_id not in prompted:
                move_id = obs.fingerprint.move_id
                reducer.add(obs)
                if reducer.is_ready(move_id):
                    prompted.add(move_id)
                    _confirm_and_merge(reducer.reduce(move_id), attacker.char_id)
                    prompted_now = True

            now = time.monotonic()
            if now - last_beat >= HEARTBEAT_SECONDS:
                print(f"  … {meter.summary(target_hz)}")
                last_beat = now

            if prompted_now:
                # The prompt blocked for arbitrary human time; don't count that gap as a poll
                # interval, and restart the cadence and heartbeat from here.
                last_poll = None
                next_poll = time.monotonic()
                last_beat = time.monotonic()
                continue
            if interval > 0:
                # Pace to the target *period*: sleep only the remainder after this iter's work, so
                # the poll rate is 1/interval (not 1/(interval + decode-time)), brief #13 Part A.
                next_poll += interval
                remaining = next_poll - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    next_poll = time.monotonic()  # fell behind; don't burst to catch up
    except KeyboardInterrupt:
        # Flush partial accumulations (≥2 reps) so the user's work isn't lost — jitter/whiffs can
        # leave a move a rep short of the target, and it would otherwise never prompt.
        leftover = [mid for mid in reducer.pending(min_reps=2) if mid not in prompted]
        if leftover and fingerprinter is not None:
            print(f"\ngathered {len(leftover)} partial move(s) — confirm them now:")
            for move_id in leftover:
                prompted.add(move_id)
                _confirm_and_merge(reducer.reduce(move_id), fingerprinter._char_id)
        print(f"\nstopped — {len(mapped)} move-id(s) mapped this session.")
        print(f"  {meter.summary(target_hz)}")
    return 0


def _prompt_confirm(
    fp: MoveFingerprint, char_fd: object, *, reps: int
) -> str | None:  # pragma: no cover - I/O
    """Show ranked candidates for a reduced move fingerprint and read a one-key confirm (#6 §B).

    ``fp`` is the multi-rep consensus (min startup, modal on-block; brief #13 §B), and ``reps`` is
    how many observations backed it — surfaced so the user can weigh a thin reading.
    """
    from tekken_coach.framedata.models import CharFrameData

    assert isinstance(char_fd, CharFrameData)
    # Live matches on STARTUP (a crisp observed event), not on live on-block (fuzzy, reads low for
    # fast moves) — brief #14. The observed on-block is shown only as an advisory hint, clearly
    # labelled approximate, and the candidates' frame data is Wavu's (the values coaching uses).
    result = join_move_live(fp, char_fd)
    if fp.on_block is not None:
        block_note = f" on-block≈{fp.on_block:+d} (approximate — reads low for fast moves)"
    else:
        block_note = " (hit; no on-block)"
    detail = f"startup≈{fp.startup}{block_note}"
    print(f"move_id {fp.move_id} (from {reps} rep{'s' if reps != 1 else ''}): {detail}")
    if not result.candidates:
        print(f"  no candidate ({result.reason}); skipping.\n")
        return None
    # A character can carry many moves at one startup, so show a generous window (not just 9) — the
    # true move must be *reachable*, not just ranked. The user picks by number (multi-digit ok).
    shown = result.candidates[:MAX_CANDIDATES_SHOWN]
    for i, cand in enumerate(shown, start=1):
        tag = " <- top" if i == 1 else ""
        startup = f"i{cand.startup}" if cand.startup is not None else "i?"
        on_block = f"{cand.on_block:+d} on block" if cand.on_block is not None else "on-block ?"
        print(f"  [{i}] {cand.framedata_key}  ({startup}, {on_block}){tag}")
    if len(result.candidates) > len(shown):
        print(f"  … {len(result.candidates) - len(shown)} more (share this startup); 's' to skip.")
    top = shown[0].framedata_key
    answer = input(f"  confirm [{top}]? Enter=yes / 1-{len(shown)}=pick / s=skip: ").strip().lower()
    if answer in ("", "y", "1"):
        return top
    if answer == "s":
        return None
    if answer.isdigit() and 1 <= int(answer) <= len(shown):
        return shown[int(answer) - 1].framedata_key
    return None
