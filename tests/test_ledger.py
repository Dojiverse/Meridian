"""Tests for the append-only ledger.

The headline property is the ROUND TRIP: cashing up the till. Two
independent routes to the same number — the recorded history, and the
state the system believes it is in — must agree exactly. A bug would
have to corrupt both routes identically to slip past.
"""

from __future__ import annotations

from datetime import date

import pytest
from hypothesis import given, settings

from meridian.ledger import (
    Buy,
    Deposit,
    Dividend,
    Fee,
    LedgerError,
    LedgerEvent,
    Portfolio,
    Sell,
    Withdrawal,
    fold,
    market_values,
    replay_to,
)
from meridian.money import Money, Price, Shares
from tests.strategies import event_streams

D = date(2026, 3, 3)


# ============================================================
# THE ROUND TRIP
# ============================================================


@given(events=event_streams())
@settings(max_examples=300)
def test_cash_reconciles(events: list[LedgerEvent]) -> None:
    """Route A: add up every cash movement in the history.
    Route B: ask the folded portfolio what the cash is.

    Cashing up the till. If these disagree, money entered or left
    without a record, which is the failure the whole design exists to
    make impossible.
    """
    expected = Money.zero()
    for e in events:
        match e:
            case Deposit(amount=a) | Dividend(amount=a):
                expected = expected + a
            case Withdrawal(amount=a) | Fee(amount=a):
                expected = expected - a
            case Buy():
                expected = expected - e.consideration
            case Sell():
                expected = expected + e.proceeds

    assert fold(events).cash == expected


@given(events=event_streams())
@settings(max_examples=300)
def test_positions_reconcile(events: list[LedgerEvent]) -> None:
    """The same check for shares. Buys in, sells out, nothing else."""
    expected: dict[str, Shares] = {}
    for e in events:
        match e:
            case Buy(ticker=t, quantity=q):
                expected[t] = expected.get(t, Shares.zero()) + q
            case Sell(ticker=t, quantity=q):
                expected[t] = expected.get(t, Shares.zero()) - q
            case _:
                pass

    folded = fold(events)
    for ticker, quantity in expected.items():
        assert folded.shares_of(ticker) == quantity


@given(events=event_streams())
@settings(max_examples=200)
def test_folding_is_deterministic(events: list[LedgerEvent]) -> None:
    """Replay the same history twice, get the same portfolio."""
    assert fold(events) == fold(events)


@given(events=event_streams())
@settings(max_examples=200)
def test_no_position_ever_goes_short(events: list[LedgerEvent]) -> None:
    """A negative share count would mean the ledger permitted selling
    what was never held."""
    for quantity in fold(events).positions.values():
        assert quantity.quantity >= 0


@given(events=event_streams())
@settings(max_examples=200)
def test_replaying_a_prefix_matches_folding_that_prefix(
    events: list[LedgerEvent],
) -> None:
    """The rewind. Stopping at any point in the stream gives exactly the
    portfolio that existed at that point — which is what makes a past
    report reproducible rather than merely archived."""
    for e in events:
        assert replay_to(events, e.seq) == fold([x for x in events if x.seq <= e.seq])


# ============================================================
# THE FOLD IS A REDUCE
# ============================================================


def test_an_empty_stream_is_an_empty_portfolio() -> None:
    p = fold([])
    assert p.cash == Money.zero()
    assert p.positions == {}


def test_a_worked_history() -> None:
    """Deposit, buy, dividend, fee, partial sell — checked by hand."""
    events: list[LedgerEvent] = [
        Deposit(1, D, Money("10000.00")),
        Buy(2, D, "VTI", Shares("40"), Price("187.43")),  # -7497.20
        Dividend(3, D, "VTI", Money("12.50")),
        Fee(4, D, Money("8.00")),
        Sell(5, D, "VTI", Shares("10"), Price("191.10")),  # +1911.00
    ]
    p = fold(events)

    # 10000 - 7497.20 + 12.50 - 8.00 + 1911.00
    assert p.cash == Money("4418.30")
    assert p.shares_of("VTI") == Shares("30")


def test_valuation_includes_cash() -> None:
    events: list[LedgerEvent] = [
        Deposit(1, D, Money("10000.00")),
        Buy(2, D, "VTI", Shares("40"), Price("187.43")),
    ]
    p = fold(events)
    total = p.total_value({"VTI": Price("190.00")})
    # 2,502.80 cash + 7,600.00 of VTI
    assert total == Money("10102.80")


def test_market_values_excludes_cash_and_closed_positions() -> None:
    events: list[LedgerEvent] = [
        Deposit(1, D, Money("10000.00")),
        Buy(2, D, "VTI", Shares("10"), Price("100.00")),
        Buy(3, D, "BND", Shares("10"), Price("50.00")),
        Sell(4, D, "BND", Shares("10"), Price("50.00")),
    ]
    prices = {"VTI": Price("110.00"), "BND": Price("50.00")}
    assert market_values(fold(events), prices) == {"VTI": Money("1100.00")}


# ============================================================
# REFUSALS
# ============================================================


def test_out_of_order_events_are_refused() -> None:
    """A stream reassembled wrongly upstream would otherwise produce a
    plausible portfolio built from the wrong history."""
    with pytest.raises(LedgerError, match="strictly increase"):
        fold([Deposit(2, D, Money("100.00")), Deposit(1, D, Money("100.00"))])


def test_duplicate_sequence_numbers_are_refused() -> None:
    with pytest.raises(LedgerError, match="strictly increase"):
        fold([Deposit(1, D, Money("100.00")), Deposit(1, D, Money("100.00"))])


def test_cannot_sell_shares_not_held() -> None:
    with pytest.raises(LedgerError, match=r"only .* held"):
        fold(
            [
                Deposit(1, D, Money("10000.00")),
                Sell(2, D, "VTI", Shares("10"), Price("100.00")),
            ]
        )


def test_cannot_buy_beyond_available_cash() -> None:
    """No margin. This reconciles fine in software and bounces at the
    custodian, which is the worst combination."""
    with pytest.raises(LedgerError, match=r"only .* is available"):
        fold(
            [
                Deposit(1, D, Money("100.00")),
                Buy(2, D, "VTI", Shares("10"), Price("100.00")),
            ]
        )


def test_cannot_withdraw_beyond_cash() -> None:
    with pytest.raises(LedgerError, match="exceeds cash"):
        fold(
            [
                Deposit(1, D, Money("100.00")),
                Withdrawal(2, D, Money("101.00")),
            ]
        )


def test_fees_may_overdraw() -> None:
    """A custodian charges the account whether or not the cash is there.
    A system that cannot record that cannot represent a real state."""
    p = fold([Deposit(1, D, Money("5.00")), Fee(2, D, Money("8.00"))])
    assert p.cash == Money("-3.00")


def test_a_negative_deposit_is_refused() -> None:
    """Direction is expressed by the event type, never by the sign.
    Otherwise a negative Deposit becomes a Withdrawal that skips the
    sufficient-funds check."""
    with pytest.raises(LedgerError, match="must be positive"):
        fold([Deposit(1, D, Money("-100.00"))])


def test_a_zero_quantity_trade_is_refused() -> None:
    with pytest.raises(LedgerError, match="must be positive"):
        fold(
            [
                Deposit(1, D, Money("10000.00")),
                Buy(2, D, "VTI", Shares("0"), Price("100.00")),
            ]
        )


def test_valuing_without_a_price_is_refused() -> None:
    """Guessing zero would understate the account silently — the worst
    way to be wrong."""
    p = fold(
        [
            Deposit(1, D, Money("10000.00")),
            Buy(2, D, "VTI", Shares("10"), Price("100.00")),
        ]
    )
    with pytest.raises(LedgerError, match="no price for"):
        p.total_value({})


# ============================================================
# IMMUTABILITY
# ============================================================


def test_events_cannot_be_modified() -> None:
    """The append-only rule, enforced by the type rather than by a
    comment asking people to be careful."""
    e = Deposit(1, D, Money("100.00"))
    with pytest.raises(AttributeError):
        e.amount = Money("999.00")  # type: ignore[misc]


def test_portfolios_cannot_be_modified() -> None:
    p = Portfolio(cash=Money("100.00"), positions={})
    with pytest.raises(AttributeError):
        p.cash = Money("999.00")  # type: ignore[misc]
