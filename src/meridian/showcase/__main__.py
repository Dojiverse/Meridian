"""Generate the showcase history:  python -m meridian.showcase [path]

Writes showcase/history.json at the repository root by default.
"""

from __future__ import annotations

import sys
from pathlib import Path

from meridian.showcase.history import build_history
from meridian.showcase.snapshots import write

DEFAULT = Path(__file__).resolve().parents[3] / "showcase" / "history.json"


def main(argv: list[str]) -> None:
    target = Path(argv[1]) if len(argv) > 1 else DEFAULT
    history = build_history()
    size = write(history, target)
    approved = sum(1 for r in history.reviews if r.status == "approved")
    rejected = sum(1 for r in history.reviews if r.status == "rejected")
    print(
        f"wrote {target} ({size / 1024:.0f} KB): "
        f"{len(history.reviews)} reviews, {approved} approved, {rejected} rejected, "
        f"{len(history.audit)} audit entries"
    )


if __name__ == "__main__":
    main(sys.argv)
