"""A two-year synthetic history, run through the real engine.

============================================================
WHAT THIS IS FOR
============================================================

The engine reads no clock and the demo store has one fixed day. Both
are deliberate — reproducibility depends on them — but neither shows
the thing an append-only ledger is actually good at: standing on any
past date and seeing exactly what the account held, what the engine
recommended, and what the gate said about it.

This package builds a household with two years of activity — deposits,
dividends, fees, a withdrawal, client-directed purchases, and a monthly
review at which the rebalancer, the lot selector, the wash-sale screen
and the compliance gate all run for real — then takes a snapshot of the
whole state at regular intervals and writes it out as JSON.

Nothing here is hand-written history. Every trade in it was proposed by
`rebalance.generate_proposal`, judged by `compliance.ComplianceGate`,
and either approved and applied or rejected with the gate's reason.
Every number in the output was produced by the engine from the event
stream, and can be reproduced from it.

============================================================
WHAT IS SYNTHETIC, AND SAID SO
============================================================

The PRICES are invented: seeded random walks shaped so the history has
something to show (a rally, a gold spike and collapse, a loss to
harvest). They are not market data and the output labels them as such.
The CLIENT ACTIVITY — how much was deposited and when, the one
withdrawal, the two client-directed purchases — is a script. Everything
downstream of those inputs is the engine.

Run it:

    python -m meridian.showcase            # writes showcase/history.json
    python -m meridian.showcase out.json
"""

from __future__ import annotations

from meridian.showcase.history import History, build_history
from meridian.showcase.snapshots import render

__all__ = ["History", "build_history", "render"]
