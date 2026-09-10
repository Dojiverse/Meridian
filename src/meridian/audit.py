"""The audit log — tamper-evident, append-only, replayable.

============================================================
WHAT THIS HAS TO SURVIVE
============================================================

Advisers Act Rule 204-2 requires records of advice, recommendations,
order execution and performance to be kept for five years from the end
of the fiscal year in which they were made, the first two readily
accessible.

Keeping the records is the easy half. The hard half is being able to
say, years later, that they are the SAME records — that nothing was
edited, backdated, or quietly dropped in between.

The SEC's 2022 amendments to Rule 17a-4(f) are instructive about what
"same" means. They kept WORM storage as an option but added an
AUDIT-TRAIL ALTERNATIVE, which requires a system that captures
modifications and deletions, the date and time of each action, the
identity of the person who took it, and enough information to RECREATE
THE ORIGINAL RECORD if it was changed.

That rule applies to broker-dealers rather than advisers — but 204-2(g)
deems records kept to the 17a-4 standard compliant with 204-2 where they
are substantially the same, so building to the stricter shape satisfies
both. This module is built to that shape.

============================================================
HOW THE CHAIN WORKS
============================================================

Every entry contains the hash of the one before it. Change any byte of
any past entry and its hash changes, which breaks the link the NEXT
entry recorded, and every link after that.

    entry 1  hash: a3f...
    entry 2  previous: a3f...  hash: 91c...
    entry 3  previous: 91c...  hash: 7de...
              ^-- edit entry 2 and this no longer matches

It is a paper ledger written in numbered ink. You cannot alter one line
without visibly renumbering every line after it, and `verify()` will
name the exact entry where the chain first breaks.

This is tamper-EVIDENT, not tamper-proof. Someone with write access to
the whole table could recompute every hash from the tampered point on.
What the chain buys you is that tampering can no longer be QUIET, and
quiet is what makes it dangerous.

============================================================
THE HEAD IS NOT PROTECTED BY THE CHAIN
============================================================

Worth stating plainly, because it is easy to miss and a property test
found it here: the chain protects every entry that has a SUCCESSOR.
Nothing points at the most recent entry, so editing it — or editing the
only entry in a one-entry log — leaves a log that still verifies against
itself.

There is no way to fix that from inside the log. Self-consistency cannot
detect a change to the thing there is nothing to compare against.

The fix is an ANCHOR: publish `head` somewhere the same person cannot
rewrite — a separate append-only store, a signed daily receipt, a
counterparty, a timestamping service — and pass it back to
`verify(expected_head=...)`. Then the last entry is protected too,
because its hash has to match a value recorded elsewhere.

`verify()` with no anchor checks internal consistency and says so.
`verify(expected_head=...)` is the complete check. The difference is
documented rather than papered over, because a security property you
believe you have and do not is worse than one you know you lack.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

__all__ = [
    "GENESIS_HASH",
    "Action",
    "AuditEntry",
    "AuditError",
    "AuditLog",
    "ChainBreak",
]

GENESIS_HASH = "0" * 64
"""What the first entry points at. A fixed, known value, so the chain
has a defined beginning rather than a nullable field that has to be
special-cased at every read."""


class AuditError(ValueError):
    """Raised when the log is asked to do something that would break it."""


class Action(Enum):
    """What happened. A closed set, so a report can group by it and a
    new kind of event cannot be invented at a call site."""

    PROPOSAL_GENERATED = "proposal.generated"
    PROPOSAL_APPROVED = "proposal.approved"
    PROPOSAL_REJECTED = "proposal.rejected"
    ORDER_SUBMITTED = "order.submitted"
    ORDER_FILLED = "order.filled"
    CONSTRAINT_VIOLATED = "compliance.violated"
    CONSTRAINT_OVERRIDDEN = "compliance.overridden"
    MODEL_PUBLISHED = "model.published"
    PRECLEARANCE_REQUESTED = "ethics.preclearance.requested"
    PRECLEARANCE_GRANTED = "ethics.preclearance.granted"
    PRECLEARANCE_DENIED = "ethics.preclearance.denied"
    RECONCILIATION_BREAK = "reconciliation.break"
    CORRECTION = "correction"
    """The only way to fix anything. You do not edit an entry; you
    append one that says what was wrong. The mistake stays visible and
    so does the fix — which is the point. On a whiteboard, a correction
    and a cover-up look identical."""


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """One immutable record. Frozen, and hashed over every field.

    `payload` values are STRINGS, deliberately. Money serialised as a
    JSON number becomes a float the moment anything parses it, and a
    hash computed over a float is a hash over whatever that platform
    rounded to. Strings hash the same everywhere, forever.
    """

    seq: int
    occurred_at: datetime
    actor: str
    """Who did it. A person, or a named system process. Required — an
    audit entry that cannot say who acted is not an audit entry."""

    action: Action
    subject: str
    """What it was about: an account id, a proposal id, an order id."""

    payload: Mapping[str, str] = field(default_factory=dict)
    previous_hash: str = GENESIS_HASH

    @property
    def digest(self) -> str:
        """This entry's hash, computed over its content AND its parent.

        Canonical JSON — keys sorted, no incidental whitespace — so the
        same content always produces the same hash. A serialiser whose
        key order depended on dict insertion would produce a chain that
        failed to verify on a different machine, which is the worst
        possible failure mode: it looks exactly like tampering.
        """
        body = json.dumps(
            {
                "seq": self.seq,
                "occurred_at": self.occurred_at.isoformat(),
                "actor": self.actor,
                "action": self.action.value,
                "subject": self.subject,
                "payload": dict(sorted(self.payload.items())),
                "previous_hash": self.previous_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def __str__(self) -> str:
        return (
            f"[{self.seq}] {self.occurred_at.isoformat()} {self.actor} "
            f"{self.action.value} {self.subject}"
        )


@dataclass(frozen=True, slots=True)
class ChainBreak:
    """Where verification failed, and why."""

    seq: int
    reason: str

    def __str__(self) -> str:
        return f"entry {self.seq}: {self.reason}"


@dataclass(frozen=True, slots=True)
class AuditLog:
    """An append-only, hash-chained sequence of entries.

    Immutable in the Python sense too: `append` returns a NEW log rather
    than mutating this one. A log object that could be modified in place
    is a log whose history depends on who held a reference to it.
    """

    entries: tuple[AuditEntry, ...] = ()

    # ---- writing ----

    def append(
        self,
        *,
        actor: str,
        action: Action,
        subject: str,
        payload: Mapping[str, str] | None = None,
        occurred_at: datetime | None = None,
    ) -> AuditLog:
        """Add an entry. Returns a new log; this one is unchanged."""
        if not actor:
            raise AuditError(
                "every entry needs an actor — a record that cannot say who "
                "acted is not an audit record"
            )

        when = occurred_at or datetime.now(UTC)
        if when.tzinfo is None:
            # A naive timestamp is a timestamp whose meaning depends on
            # which machine wrote it. Market close is 4pm Eastern, which
            # is a different UTC offset in March than in January.
            raise AuditError(
                f"occurred_at must be timezone-aware, got naive {when}. "
                "Store UTC; reason in Eastern."
            )

        entry = AuditEntry(
            seq=len(self.entries) + 1,
            occurred_at=when,
            actor=actor,
            action=action,
            subject=subject,
            payload=dict(payload or {}),
            previous_hash=self.head,
        )
        return AuditLog(entries=(*self.entries, entry))

    # ---- reading ----

    @property
    def head(self) -> str:
        """The hash of the most recent entry — the chain's current tip.

        Publishing this somewhere the firm cannot rewrite (a separate
        store, a signed daily receipt, a counterparty) is what turns
        tamper-EVIDENT into tamper-detectable by an outsider.
        """
        return self.entries[-1].digest if self.entries else GENESIS_HASH

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[AuditEntry]:
        return iter(self.entries)

    def as_of(self, when: datetime) -> AuditLog:
        """The log as it stood at a moment in time.

        The rewind. Combined with the ledger's `replay_to`, this is what
        makes "reproduce the recommendation you made in June" a query
        rather than an archaeology project.
        """
        if when.tzinfo is None:
            raise AuditError("as_of needs a timezone-aware datetime")
        return AuditLog(tuple(e for e in self.entries if e.occurred_at <= when))

    def for_subject(self, subject: str) -> tuple[AuditEntry, ...]:
        return tuple(e for e in self.entries if e.subject == subject)

    def of_action(self, action: Action) -> tuple[AuditEntry, ...]:
        return tuple(e for e in self.entries if e.action is action)

    # ---- verification ----

    def verify(self, expected_head: str | None = None) -> tuple[ChainBreak, ...]:
        """Recompute the whole chain. An empty result means intact.

        Args:
            expected_head: The chain tip as recorded by an EXTERNAL
                anchor. Without it this checks internal consistency
                only, which cannot detect a change to the most recent
                entry — nothing points at the last link. See the module
                docstring.

        Returns every break rather than the first, because a report that
        stops at the first problem invites fixing them one at a time
        without ever seeing the shape of what happened.
        """
        breaks: list[ChainBreak] = []
        expected_previous = GENESIS_HASH

        for position, entry in enumerate(self.entries, start=1):
            if entry.seq != position:
                breaks.append(
                    ChainBreak(
                        entry.seq,
                        f"sequence is {entry.seq} but this is position "
                        f"{position} — an entry was inserted or removed",
                    )
                )

            if entry.previous_hash != expected_previous:
                breaks.append(
                    ChainBreak(
                        entry.seq,
                        "previous_hash does not match the entry before it — "
                        "an earlier entry was modified",
                    )
                )

            expected_previous = entry.digest

        if expected_head is not None and self.head != expected_head:
            breaks.append(
                ChainBreak(
                    len(self.entries),
                    "head does not match the anchored value — the most recent "
                    "entry was modified, or entries were removed from the end",
                )
            )

        return tuple(breaks)

    @property
    def is_intact(self) -> bool:
        """Internally consistent. NOT a complete tamper check — see
        `verify` and the module docstring on why the head needs an
        external anchor."""
        return not self.verify()

    def is_intact_against(self, anchor: str) -> bool:
        """The complete check: internal consistency AND a head matching
        an externally recorded value."""
        return not self.verify(expected_head=anchor)

    @classmethod
    def rebuild(cls, entries: Sequence[AuditEntry]) -> AuditLog:
        """Load entries from storage without re-chaining them.

        Deliberately does NOT recompute hashes. The whole value of the
        chain is that it verifies what was WRITTEN; a loader that
        rebuilt the hashes would launder any tampering into a
        perfectly valid-looking log. Load as-is, then call `verify`.
        """
        return cls(entries=tuple(entries))
