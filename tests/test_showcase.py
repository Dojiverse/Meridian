"""Tests for the showcase history.

The history is generated, not written, so the tests pin two kinds of
thing: that the generator keeps the engine's guarantees (every number a
string, every snapshot reproducible from the events before it, no wash
sale anywhere in two years), and that the story it exists to tell
actually happens — because a seeded random walk could drift away from
it, and a showcase whose headline example quietly vanished would be
worse than none.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from meridian.audit import Action
from meridian.ledger import fold
from meridian.money import Money
from meridian.showcase.history import History, build_history
from meridian.showcase.snapshots import render, snapshot_dates
from meridian.taxlot import build_lots, replay_disposals
from meridian.washsale import find_wash_sales


@pytest.fixture(scope="module")
def history() -> History:
    return build_history()


@pytest.fixture(scope="module")
def document(history: History) -> dict[str, Any]:
    # Through JSON and back, so the tests see what a browser would.
    return dict(json.loads(json.dumps(render(history))))


# ============================================================
# THE ENGINE'S GUARANTEES SURVIVE THE GENERATOR
# ============================================================


def test_the_history_is_deterministic() -> None:
    """Two builds, byte for byte. A showcase that changed between runs
    could not be compared against yesterday's screenshot."""
    first = json.dumps(render(build_history()), sort_keys=True)
    second = json.dumps(render(build_history()), sort_keys=True)
    assert first == second


def test_every_ledger_folds_cleanly(history: History) -> None:
    """Every trade in the history was applied to the ledger as it was
    generated. Folding the finished stream must still raise nothing —
    no overdraft, no short, no out-of-order event."""
    for state in history.accounts.values():
        portfolio = fold(state.events)
        assert not portfolio.total_value(history.prices.on(history.today)).is_zero


def test_sequence_numbers_are_contiguous(history: History) -> None:
    for state in history.accounts.values():
        assert [e.seq for e in state.events] == list(range(1, len(state.events) + 1))


def _walk(node: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, f"{path}.{key}")
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item, f"{path}[]")
    else:
        yield path, node


INTEGER_KEYS = {
    "seq",
    "events_through_seq",
    "model_version",
    "version",
    "days",
    "breaches",
    "disposals",
    "snapshots",
    "reviews",
    "audit_entries",
    "applied_seqs",
    "year",
    "approved_reviews",
    "rejected_reviews",
    "no_action_reviews",
    "harvest_reviews",
}
"""The only JSON numbers allowed: counts and sequence numbers, which
are exact as integers, plus the charting integers ending in _cents or
_bps. Everything that is an amount is a string."""


def test_no_amount_anywhere_is_a_json_number(document: dict[str, Any]) -> None:
    """The API's rule, applied to the file a browser will load. A float
    anywhere is a bug; an int is allowed only under a counting key."""
    offenders = []
    for path, value in _walk(document):
        if isinstance(value, bool):
            continue
        if isinstance(value, float):
            offenders.append(f"float at {path} = {value!r}")
        elif isinstance(value, int):
            leaf = path.rsplit(".", 1)[-1].removesuffix("[]")
            if leaf not in INTEGER_KEYS and not leaf.endswith(("_cents", "_bps")):
                offenders.append(f"int at {path} = {value!r}")
    assert not offenders, "\n".join(offenders)


def test_amounts_parse_back_to_exact_decimals(document: dict[str, Any]) -> None:
    last = document["snapshots"][-1]["accounts"]["taxable-1"]
    assert Decimal(last["total"]) == sum(
        (Decimal(p["value"]) for p in last["positions"]), Decimal(last["cash"])
    )


def test_every_snapshot_is_the_fold_of_the_events_before_it(
    history: History, document: dict[str, Any]
) -> None:
    """The as-of property, checked from the outside. Each snapshot's
    cash and positions must equal what the ledger says when replayed
    only through that date — and its lots must still add up to the
    position, because they come from the same prefix."""
    for snapshot in document["snapshots"]:
        on = date.fromisoformat(snapshot["on"])
        for account_id, shown in snapshot["accounts"].items():
            state = history.accounts[account_id]
            events = state.events_through(on)
            portfolio = fold(events)
            assert Decimal(shown["cash"]) == portfolio.cash.amount
            assert shown["events_through_seq"] == (events[-1].seq if events else 0)

            lots = build_lots(events)
            for row in shown["positions"]:
                held = portfolio.positions[row["ticker"]].quantity
                assert Decimal(row["quantity"]) == held
                in_lots = sum(
                    (lot.quantity.quantity for lot in lots[row["ticker"]]), Decimal(0)
                )
                assert in_lots == held


def test_snapshots_cover_every_event_day_and_every_friday(history: History) -> None:
    days = set(snapshot_dates(history))
    for state in history.accounts.values():
        assert all(e.on in days for e in state.events)
    fridays = [d for d in history.prices.days if d.weekday() == 4]
    assert all(f in days for f in fridays)
    assert history.today in days


def test_the_audit_chain_is_intact_and_covers_every_decision(
    history: History,
) -> None:
    assert history.audit.is_intact
    acted = [r for r in history.reviews if r.status != "no_action"]
    assert len(history.audit.of_action(Action.PROPOSAL_GENERATED)) == len(acted)


def test_the_gate_prevented_every_wash_sale(history: History) -> None:
    """Two years, dozens of trades, a gold position sold at a loss, and
    a Roth that bought the same thing — and not one wash sale on the
    books. That is the gate doing its job, verified by the detector
    running over everything that actually happened."""
    taxable = history.accounts["taxable-1"]
    report = find_wash_sales(
        replay_disposals(taxable.events),
        history.household_acquisitions_through(history.today),
        history.substitutes,
    )
    assert not report.findings


def test_taxable_sells_are_all_lot_identified(history: History) -> None:
    from meridian.ledger import Sell

    sells = [e for e in history.accounts["taxable-1"].events if isinstance(e, Sell)]
    assert sells
    assert all(e.lot_ids for e in sells)


# ============================================================
# THE STORY ACTUALLY HAPPENS
# ============================================================


def test_the_gold_sale_is_blocked_under_rev_rul_2008_5(
    history: History, document: dict[str, Any]
) -> None:
    """December 2025: the rebalancer wants to trim gold, picks the loss
    lot, and the gate blocks it because the Roth bought gold on
    1 December. The forfeiture citation must be on the record."""
    review_id = document["story"]["forfeiture_block_review"]
    assert review_id is not None
    review = next(r for r in history.reviews if r.review_id == review_id)

    assert review.on == date(2025, 12, 15)
    assert review.account_id == "taxable-1"
    blocks = {v.subject: v for v in review.compliance.blocks}
    assert blocks["GLD"].authority == "Rev. Rul. 2008-5"
    assert "roth-1" in blocks["GLD"].message
    assert "2025-12-01" in blocks["GLD"].message


def test_blocking_the_funding_sale_took_the_whole_review_down(
    history: History, document: dict[str, Any]
) -> None:
    """The gold sale was paying for every buy. Remove it and nothing is
    fundable, so the gate dropped the buys too, each with its own
    reason, and the advisor had nothing left to approve."""
    review_id = document["story"]["forfeiture_block_review"]
    review = next(r for r in history.reviews if r.review_id == review_id)
    assert review.status == "rejected"
    assert not review.compliance.passed
    funding = [v for v in review.compliance.violations if v.constraint_id == "funding"]
    assert funding


def test_the_same_sale_clears_once_the_window_has_closed(
    history: History, document: dict[str, Any]
) -> None:
    """January 2026: thirty days have passed since the Roth purchase.
    The same lot is sold, the loss is realised, and it is not washed."""
    review_id = document["story"]["clean_loss_sale_review"]
    assert review_id is not None
    review = next(r for r in history.reviews if r.review_id == review_id)

    assert review.on == date(2026, 1, 15)
    assert review.status == "approved"
    gold = [t for t in review.compliance.passed if t.ticker == "GLD"]
    assert gold
    assert gold[0].realized_gain is not None
    assert gold[0].realized_gain < Money("-2000.00")
    # The lot it sold is the client's November purchase.
    assert all(sel.acquired == date(2025, 11, 20) for sel in gold[0].lots)


def test_the_harvest_screen_showed_the_block_before_the_review(
    document: dict[str, Any],
) -> None:
    """On the December snapshot the harvest screen already lists the
    gold lot as a blocked opportunity, naming the Roth purchase and
    the forfeiture — the advisor could have seen it coming."""
    december = next(s for s in document["snapshots"] if s["on"] == "2025-12-15")
    harvest = december["accounts"]["taxable-1"]["harvest"]
    gold = [h for h in harvest if h["ticker"] == "GLD"]
    assert gold
    assert gold[0]["is_blocked"]
    assert "PERMANENTLY FORFEITED" in gold[0]["block_reason"]
    assert any(b["forfeits"] for b in gold[0]["blocked_by"])


def test_sells_choose_lots_across_the_whole_sleeve(history: History) -> None:
    """At least one taxable review trims the equity sleeve by selling
    from more than one security, each order naming its lots — the
    cross-ticker ranking in action."""
    multi = [
        r
        for r in history.reviews
        if r.account_id == "taxable-1"
        and len(
            {
                t.ticker
                for t in r.compliance.passed
                if t.sleeve == "equity" and t.side.value == "sell" and not t.harvest
            }
        )
        > 1
    ]
    assert multi
    for trade in multi[0].compliance.passed:
        if trade.sleeve == "equity" and trade.side.value == "sell":
            assert trade.lots


def test_idle_cash_is_put_to_work(history: History) -> None:
    """Quarterly contributions do not sit for a year. The cash trigger
    fires, the review deploys, and the record says why it acted."""
    cash_reviews = [
        r
        for r in history.reviews
        if r.status == "approved" and r.proposal.trigger == "cash"
    ]
    assert len(cash_reviews) >= 4
    for state in history.accounts.values():
        final = state.portfolio_on(history.today)
        total = final.total_value(history.prices.on(history.today))
        assert final.cash.ratio_to(total).value < Decimal("0.03")


def test_gips_refusal_is_visible_in_the_first_year(document: dict[str, Any]) -> None:
    """Early snapshots cannot be annualised and say why. Later ones can."""
    early = next(s for s in document["snapshots"] if s["on"] == "2025-03-14")
    late = document["snapshots"][-1]
    early_perf = early["accounts"]["taxable-1"]["performance"]
    late_perf = late["accounts"]["taxable-1"]["performance"]
    assert early_perf["annualized"] is None
    assert "GIPS" in early_perf["annualized_note"]
    assert late_perf["annualized"] is not None
    assert late_perf["twr_gross"] != late_perf["twr_net"]


def test_the_document_says_it_is_synthetic(document: dict[str, Any]) -> None:
    assert document["meta"]["synthetic"] is True
    assert "not market data" in document["meta"]["note"]


# ============================================================
# HARVESTING AND THE BLACKOUT
# ============================================================


def test_losses_are_harvested_and_replaced(history: History) -> None:
    """At least one review is a pure harvest: a lot-identified sell at a
    loss paired with a buy of the firm's alternative, same sleeve."""
    harvests = [
        r
        for r in history.reviews
        if r.status == "approved" and r.proposal.trigger == "harvest"
    ]
    assert harvests
    first = harvests[0]
    sells = [t for t in first.compliance.passed if t.side.value == "sell" and t.harvest]
    buys = [t for t in first.compliance.passed if t.side.value == "buy" and t.harvest]
    assert sells and buys
    assert sells[0].pair == buys[0].pair
    assert sells[0].sleeve == buys[0].sleeve
    assert sells[0].realized_gain is not None and sells[0].realized_gain.is_negative


def test_the_blackout_reaches_across_the_household(history: History) -> None:
    """After the taxable account harvests a VTI loss, the Roth's next
    review wants to buy VTI and may not. The gap is reported with the
    rule rather than quietly filled."""
    gaps = [
        (r, u)
        for r in history.reviews
        if r.account_id == "roth-1"
        for u in r.proposal.unplaced
        if "IRC 1091" in u.reason
    ]
    assert gaps
    _review, gap = gaps[0]
    assert gap.sleeve == "equity"
    assert "VTI" in gap.reason


def test_no_harvest_leg_executed_alone(history: History) -> None:
    """Every executed harvest sell has its replacement buy in the same
    review, or no replacement was designated — never a replacement
    that was blocked while the sell went through."""
    for review in history.reviews:
        if review.status != "approved":
            continue
        executed = {t.pair for t in review.compliance.passed if t.pair}
        blocked = {t.pair for t in review.compliance.blocked if t.pair}
        assert not (executed & blocked)
