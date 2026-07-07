"""CLI / output: mode selection, orchestration, terminal rendering (docs/07). Chunk C6.

C0 ships only the console-script entry point as a stub; the commands land in C6.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Console entry point for ``tekken-coach``. Stub until C6 wires up the commands."""
    _ = argv if argv is not None else sys.argv[1:]
    print("tekken-coach: CLI not implemented yet (see docs/implementation-plan.md, chunk C6).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
