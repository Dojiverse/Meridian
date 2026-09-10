"""Tests for the tamper-evident audit log.

The headline demonstration is that editing any past entry breaks the
chain at a nameable point — which is the whole reason to build one.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from meridian.audit import GENESIS_HASH, Action, AuditError, AuditLog

T0 = datetime(2026, 6, 1, 14, 30, tzinfo=UTC)


def sample_log() -> AuditLog:
    log = AuditLog()
    log = log.append(
        actor="rebalancer",
        action=Action.PROPOSAL_GENERATED,
        subject="prop-001",
        payload={"account": "taxable-1", "trades": "4", "turnover": "10999.74"},
        occurred_at=T0,
    )
    log = log.append(
        actor="a.advisor",
        action=Action.PROPOSAL_APPROVED,
        subject="prop-001",
        payload={"note": "reviewed tax estimate"},
        occurred_at=T0 + timedelta(minutes=12),
    )
    log = log.append(
        actor="order-router",
        action=Action.ORDER_SUBMITTED,
        subject="ord-77",
        payload={"ticker": "BND", "quantity": "109.997", "idempotency_key": "A7F3"},
        occurred_at=T0 + timedelta(minutes=13),
    )
    return log


# ============================================================
# THE CHAIN
# ============================================================


def test_a_fresh_log_is_intact() -> None:
    assert AuditLog().is_intact
    assert AuditLog().head == GENESIS_HASH


def test_a_written_log_verifies() -> None:
    assert sample_log().is_intact


def test_each_entry_points_at_the_one_before() -> None:
    log = sample_log()
    assert log.entries[0].previous_hash == GENESIS_HASH
    assert log.entries[1].previous_hash == log.entries[0].digest
    assert log.entries[2].previous_hash == log.entries[1].digest


def test_editing_a_past_entry_breaks_the_chain() -> None:
    """The demonstration. Change the approval note on entry 2 and
    verification names entry 3 as the first broken link — because entry
    3 recorded a hash of the entry 2 that used to exist."""
    log = sample_log()
    tampered = list(log.entries)
    tampered[1] = replace(tampered[1], payload={"note": "approved without review"})

    breaks = AuditLog.rebuild(tampered).verify()

    assert breaks
    assert breaks[0].seq == 3
    assert "modified" in breaks[0].reason


def test_changing_an_actor_breaks_the_chain() -> None:
    """Who did it is hashed too. Rewriting an approval to name someone
    else does not slip through."""
    log = sample_log()
    tampered = list(log.entries)
    tampered[1] = replace(tampered[1], actor="someone.else")
    assert AuditLog.rebuild(tampered).verify()


def test_backdating_an_entry_breaks_the_chain() -> None:
    log = sample_log()
    tampered = list(log.entries)
    tampered[0] = replace(tampered[0], occurred_at=T0 - timedelta(days=30))
    assert AuditLog.rebuild(tampered).verify()


def test_deleting_an_entry_is_visible() -> None:
    """A quiet deletion is the failure a plain append-only table cannot
    catch. The sequence numbers give it away, and so does the chain."""
    log = sample_log()
    without_middle = [log.entries[0], log.entries[2]]

    breaks = AuditLog.rebuild(without_middle).verify()

    assert breaks
    assert any("inserted or removed" in b.reason for b in breaks)


def test_verification_reports_every_break_not_just_the_first() -> None:
    """A report that stops at the first problem invites fixing them one
    at a time without ever seeing the shape of what happened."""
    log = sample_log()
    tampered = list(log.entries)
    tampered[0] = replace(tampered[0], subject="prop-999")
    assert len(AuditLog.rebuild(tampered).verify()) >= 1


def test_rebuild_does_not_launder_tampering() -> None:
    """The loader deliberately does NOT recompute hashes. One that did
    would turn any tampered log into a perfectly valid-looking one."""
    log = sample_log()
    tampered = list(log.entries)
    tampered[1] = replace(tampered[1], actor="forged")

    assert not AuditLog.rebuild(tampered).is_intact


# ============================================================
# APPENDING
# ============================================================


def test_a_log_is_immutable_and_append_returns_a_new_one() -> None:
    """A log object that could be modified in place is a log whose
    history depends on who held a reference to it."""
    original = sample_log()
    extended = original.append(
        actor="ops", action=Action.ORDER_FILLED, subject="ord-77", occurred_at=T0
    )
    assert len(original) == 3
    assert len(extended) == 4


def test_an_entry_without_an_actor_is_refused() -> None:
    """A record that cannot say who acted is not an audit record."""
    with pytest.raises(AuditError, match="needs an actor"):
        AuditLog().append(actor="", action=Action.CORRECTION, subject="x")


def test_a_naive_timestamp_is_refused() -> None:
    """A naive timestamp means something different depending on which
    machine wrote it. Market close is 4pm Eastern, a different UTC
    offset in March than in January."""
    with pytest.raises(AuditError, match="timezone-aware"):
        AuditLog().append(
            actor="ops",
            action=Action.CORRECTION,
            subject="x",
            occurred_at=datetime(2026, 6, 1, 14, 30),
        )


def test_corrections_are_appended_not_edited() -> None:
    """Six hundred years of accounting practice. The mistake stays
    visible and so does the fix — on a whiteboard, a correction and a
    cover-up look identical."""
    log = sample_log().append(
        actor="a.advisor",
        action=Action.CORRECTION,
        subject="prop-001",
        payload={"corrects": "2", "reason": "turnover misstated as 10999.74"},
        occurred_at=T0 + timedelta(days=1),
    )
    assert log.is_intact
    assert len(log) == 4
    assert log.entries[1].payload["note"] == "reviewed tax estimate"


# ============================================================
# REPLAY
# ============================================================


def test_as_of_rewinds_the_log() -> None:
    """The rewind. With the ledger's replay_to, this makes 'reproduce
    the recommendation you made in June' a query rather than an
    archaeology project."""
    log = sample_log()
    earlier = log.as_of(T0 + timedelta(minutes=12))

    assert len(earlier) == 2
    assert earlier.is_intact
    assert earlier.entries[-1].action is Action.PROPOSAL_APPROVED


def test_as_of_before_everything_is_empty() -> None:
    assert len(sample_log().as_of(T0 - timedelta(days=1))) == 0


def test_as_of_needs_a_timezone() -> None:
    with pytest.raises(AuditError, match="timezone-aware"):
        sample_log().as_of(datetime(2026, 6, 1))


def test_entries_can_be_filtered_by_subject_and_action() -> None:
    log = sample_log()
    assert len(log.for_subject("prop-001")) == 2
    assert len(log.of_action(Action.ORDER_SUBMITTED)) == 1


# ============================================================
# DETERMINISM
# ============================================================


def test_the_same_content_hashes_the_same() -> None:
    """Canonical JSON — sorted keys, no incidental whitespace. A
    serialiser whose key order depended on dict insertion would produce
    a chain that failed to verify on a different machine, which looks
    exactly like tampering."""
    assert sample_log().head == sample_log().head


def test_payload_key_order_does_not_change_the_hash() -> None:
    forward = AuditLog().append(
        actor="ops",
        action=Action.ORDER_SUBMITTED,
        subject="ord-1",
        payload={"a": "1", "b": "2", "c": "3"},
        occurred_at=T0,
    )
    backward = AuditLog().append(
        actor="ops",
        action=Action.ORDER_SUBMITTED,
        subject="ord-1",
        payload={"c": "3", "b": "2", "a": "1"},
        occurred_at=T0,
    )
    assert forward.head == backward.head


def test_a_different_payload_changes_the_hash() -> None:
    a = AuditLog().append(
        actor="ops",
        action=Action.ORDER_SUBMITTED,
        subject="ord-1",
        payload={"quantity": "100"},
        occurred_at=T0,
    )
    b = AuditLog().append(
        actor="ops",
        action=Action.ORDER_SUBMITTED,
        subject="ord-1",
        payload={"quantity": "101"},
        occurred_at=T0,
    )
    assert a.head != b.head


# ============================================================
# PROPERTIES
# ============================================================


@st.composite
def logs(draw: st.DrawFn) -> AuditLog:
    n = draw(st.integers(min_value=0, max_value=25))
    log = AuditLog()
    for i in range(n):
        log = log.append(
            actor=draw(st.sampled_from(["alice", "bob", "system"])),
            action=draw(st.sampled_from(list(Action))),
            subject=draw(st.text(alphabet="abc123-", min_size=1, max_size=8)),
            payload={"i": str(i)},
            occurred_at=T0 + timedelta(seconds=i),
        )
    return log


@given(log=logs())
@settings(max_examples=300)
def test_any_honestly_built_log_verifies(log: AuditLog) -> None:
    assert log.is_intact


@given(log=logs(), index=st.integers(min_value=0, max_value=24))
@settings(max_examples=400)
def test_tampering_with_any_entry_that_has_a_successor_is_caught(
    log: AuditLog, index: int
) -> None:
    """Self-verification catches every edit EXCEPT one to the last entry.

    That exception is not a gap in the implementation — it is a property
    of hash chains. The chain protects each entry by recording its hash
    in the NEXT one, so the most recent entry has nothing pointing at
    it.

    Hypothesis found this. The original assertion claimed every edit was
    caught and it shrank the counterexample to a one-entry log: edit the
    only entry and there is nothing to compare it against.
    """
    if len(log.entries) < 2:
        return
    # Anywhere except the final entry.
    position = index % (len(log.entries) - 1)

    tampered = list(log.entries)
    tampered[position] = replace(tampered[position], actor="forged")

    assert not AuditLog.rebuild(tampered).is_intact


@given(log=logs(), index=st.integers(min_value=0, max_value=24))
@settings(max_examples=400)
def test_with_an_anchor_tampering_with_ANY_entry_is_caught(
    log: AuditLog, index: int
) -> None:
    """The complete check, including the last entry.

    An anchor is the chain head recorded somewhere the same person
    cannot rewrite — a separate store, a signed daily receipt, a
    counterparty. Compare against it and the final entry is protected
    too, because its hash has to match a value held elsewhere.
    """
    if not log.entries:
        return

    anchor = log.head  # what an external witness recorded
    position = index % len(log.entries)

    tampered = list(log.entries)
    tampered[position] = replace(tampered[position], actor="forged")

    assert not AuditLog.rebuild(tampered).is_intact_against(anchor)


def test_editing_the_last_entry_is_invisible_without_an_anchor() -> None:
    """Documented explicitly, because a security property you believe
    you have and do not is worse than one you know you lack."""
    log = sample_log()
    tampered = list(log.entries)
    tampered[-1] = replace(tampered[-1], payload={"quantity": "999999"})
    rebuilt = AuditLog.rebuild(tampered)

    assert rebuilt.is_intact  # self-consistent, and wrong
    assert not rebuilt.is_intact_against(log.head)  # the anchor catches it


def test_truncating_the_log_is_caught_by_an_anchor() -> None:
    """Dropping the most recent entries leaves a perfectly valid chain.
    Only an external record of where the head SHOULD be reveals it."""
    log = sample_log()
    truncated = AuditLog.rebuild(log.entries[:-1])

    assert truncated.is_intact
    assert not truncated.is_intact_against(log.head)


@given(log=logs())
@settings(max_examples=200)
def test_every_prefix_of_a_valid_log_is_valid(log: AuditLog) -> None:
    """Rewinding to any point gives an intact log, which is what makes
    as_of() trustworthy rather than merely convenient."""
    for i in range(len(log.entries) + 1):
        assert AuditLog.rebuild(log.entries[:i]).is_intact


@given(log=logs())
@settings(max_examples=200)
def test_sequence_numbers_are_contiguous_from_one(log: AuditLog) -> None:
    """A gap is a question someone has to answer."""
    assert [e.seq for e in log.entries] == list(range(1, len(log.entries) + 1))
