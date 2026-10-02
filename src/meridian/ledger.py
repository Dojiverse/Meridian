"""The append-only ledger.

============================================================
WHY NOTHING IS EVER EDITED
============================================================

Most software uses a whiteboard. Cash is 5,000; the client withdraws
200; erase it and write 4,800. Fast, simple, and the old number is gone
forever. Nobody can say what it read last Tuesday or who changed it.

This uses a bank statement instead. Lines are only ever added.

    bought 40 VTI @ 187.43
    dividend received 12.50
    paid fee 8.00
    sold 10 VTI @ 191.10

Nowhere does that say what you own. You work it out by reading the whole
list — which is exactly a reduce, the same shape as the drill:

    holdings.reduce((sum, h) => sum + h.value, 0)

Three things follow from that choice:

  * REWIND. What did this account hold on 3 March? Replay the list and
    stop at 3 March.
  * TAMPER EVIDENCE. There is no UPDATE. In production the database
    grant is revoked, so history cannot be rewritten by a bug, a
    migration, or a person.
  * RETENTION FOR FREE. Advisers Act Rule 204-2 wants five years of
    records. Nothing here can be deleted, so the requirement is a
    property of the design rather than a policy someone has to follow.

Corrections work the way accountants have done them for six hundred
years: you do not erase, you post a correcting entry. The mistake stays
visible and so does the fix. On a whiteboard, a correction and a
cover-up look identical.

============================================================
WHAT THIS MODULE DOES NOT DO
============================================================

The events themselves are not hash-chained; that protection lives in
`audit.py`, which records every decision made ABOUT the ledger. The
structural guarantee here — events immutable, positions derived,
nothing edited in place — is what everything above it depends on.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

from meridian.money import Money, Price, Shares

__all__ = [
    "Buy",
    "Deposit",
    "Dividend",
    "Fee",
    "LedgerError",
    "LedgerEvent",
    "Portfolio",
    "Sell",
    "Withdrawal",
    "fold",
    "market_values",
]


class LedgerError(ValueError):
    """Raised when an event cannot be applied to the portfolio before it."""


# ============================================================
# Events
# ============================================================
# Each event is a frozen dataclass: once created it cannot be modified,
# which is the append-only rule expressed in the type system rather than
# in a comment asking people to be careful.
#
# `seq` is the position in the stream. It must strictly increase.
# Sequence numbers are how a missing event becomes visible — a gap is a
# question someone has to answer.


@dataclass(frozen=True, slots=True)
class Deposit:
    """Cash arriving from the client."""

    seq: int
    on: date
    amount: Money


@dataclass(frozen=True, slots=True)
class Withdrawal:
    """Cash leaving to the client."""

    seq: int
    on: date
    amount: Money


@dataclass(frozen=True, slots=True)
class Buy:
    """Shares acquired. Cash out, position up."""

    seq: int
    on: date
    ticker: str
    quantity: Shares
    price: Price

    @property
    def consideration(self) -> Money:
        """What it cost. Shares x Price -> Money."""
        return self.quantity.value_at(self.price)


@dataclass(frozen=True, slots=True)
class Sell:
    """Shares disposed. Cash in, position down."""

    seq: int
    on: date
    ticker: str
    quantity: Shares
    price: Price

    lot_ids: tuple[str, ...] = ()
    """Which lots this sale disposes of, in the order they are consumed.

    Empty means no identification was made and the default method
    applies when lots are rebuilt (FIFO for stock). Populated when the
    order was sized against specific lots — which is what a tax-aware
    rebalance does — so that replaying the ledger consumes exactly the
    lots the trade was built on. Treas. Reg. 1.1012-1(c) makes the
    identification part of the sale; here it is part of the event.
    """

    @property
    def proceeds(self) -> Money:
        return self.quantity.value_at(self.price)


@dataclass(frozen=True, slots=True)
class Dividend:
    """Cash paid by a holding. Recorded against the ticker that paid it,
    because performance attribution later needs to know where income
    came from, not merely that cash appeared."""

    seq: int
    on: date
    ticker: str
    amount: Money


@dataclass(frozen=True, slots=True)
class Fee:
    """Advisory fee, custodial charge, commission."""

    seq: int
    on: date
    amount: Money
    description: str = ""


# A closed set. Adding a seventh event type means updating `fold`, and
# mypy will point at the exact line that needs it — which is the reason
# to write this as a union rather than a base class with a hook.
LedgerEvent = Deposit | Withdrawal | Buy | Sell | Dividend | Fee


# ============================================================
# The derived state
# ============================================================


@dataclass(frozen=True, slots=True)
class Portfolio:
    """What the events add up to. Never stored — always recomputed.

    Frozen, like the events. A portfolio is a photograph of a moment in
    the stream, and photographs do not change after the fact.
    """

    cash: Money = field(default_factory=Money.zero)
    positions: Mapping[str, Shares] = field(default_factory=dict)

    def shares_of(self, ticker: str) -> Shares:
        return self.positions.get(ticker, Shares.zero())

    def total_value(self, prices: Mapping[str, Price]) -> Money:
        """Cash plus the market value of every position.

        Raises if a held position has no price. A portfolio valued with
        a missing price is a portfolio valued wrongly, and guessing zero
        would understate the account silently — the worst way to be
        wrong.
        """
        total = self.cash
        for ticker, quantity in self.positions.items():
            if quantity.is_zero:
                continue
            if ticker not in prices:
                raise LedgerError(
                    f"cannot value the portfolio: no price for {ticker!r}"
                )
            total = total + quantity.value_at(prices[ticker])
        return total


def market_values(
    portfolio: Portfolio, prices: Mapping[str, Price]
) -> dict[str, Money]:
    """The market value of each position, by ticker. Cash is excluded.

    Zero-quantity positions are dropped. A position closed to zero is
    not a holding worth 0 — it is not a holding, and leaving it in would
    put empty rows on every report downstream.
    """
    values: dict[str, Money] = {}
    for ticker, quantity in portfolio.positions.items():
        if quantity.is_zero:
            continue
        if ticker not in prices:
            raise LedgerError(f"no price for {ticker!r}")
        values[ticker] = quantity.value_at(prices[ticker])
    return values


# ============================================================
# The fold
# ============================================================


def fold(events: Iterable[LedgerEvent]) -> Portfolio:
    """Replay the event stream into the portfolio it describes.

    This is the reduce. Start empty, walk every event, carry the running
    state. The same shape as summing a list of holdings — only the
    accumulator is a portfolio instead of a number.

    Raises:
        LedgerError: on an out-of-order sequence, a sale of shares not
            held, or a purchase the cash cannot fund.

    Those are refusals, not warnings. An event stream that cannot be
    applied describes a portfolio that never existed, and continuing
    past it would produce numbers no one could defend.
    """
    cash = Money.zero()
    positions: dict[str, Shares] = {}
    last_seq: int | None = None

    for event in events:
        # ---- ordering ------------------------------------------
        # A stream that arrives out of order has been reassembled
        # wrongly somewhere upstream, and applying it would produce a
        # plausible-looking portfolio built from the wrong history.
        if last_seq is not None and event.seq <= last_seq:
            raise LedgerError(
                f"event {event.seq} arrived after {last_seq}; "
                "the stream must strictly increase"
            )
        last_seq = event.seq

        # ---- apply ---------------------------------------------
        # `match` on the event union. mypy checks this is exhaustive,
        # so adding a seventh event type without handling it here is a
        # type error rather than a silently ignored event.
        match event:
            case Deposit(amount=amount):
                _require_positive(amount, event, "deposit")
                cash = cash + amount

            case Withdrawal(amount=amount):
                _require_positive(amount, event, "withdrawal")
                if amount > cash:
                    raise LedgerError(
                        f"event {event.seq}: withdrawal of {amount} exceeds "
                        f"cash of {cash}"
                    )
                cash = cash - amount

            case Buy(ticker=ticker, quantity=quantity, price=price):
                _require_positive_shares(quantity, event)
                cost = quantity.value_at(price)
                if cost > cash:
                    # No margin in this system. Buying with money that
                    # is not there is the kind of thing that reconciles
                    # fine in the software and bounces at the custodian.
                    raise LedgerError(
                        f"event {event.seq}: buying {ticker} costs {cost} "
                        f"but only {cash} is available"
                    )
                cash = cash - cost
                positions[ticker] = positions.get(ticker, Shares.zero()) + quantity

            case Sell(ticker=ticker, quantity=quantity, price=price):
                _require_positive_shares(quantity, event)
                held = positions.get(ticker, Shares.zero())
                if quantity.quantity > held.quantity:
                    # No short selling. A sale of shares not held is a
                    # data error, not a trading strategy.
                    raise LedgerError(
                        f"event {event.seq}: selling {quantity} of {ticker} "
                        f"but only {held} held"
                    )
                positions[ticker] = held - quantity
                cash = cash + quantity.value_at(price)

            case Dividend(amount=amount):
                _require_positive(amount, event, "dividend")
                cash = cash + amount

            case Fee(amount=amount):
                _require_positive(amount, event, "fee")
                # Fees are allowed to overdraw. A custodian charges the
                # account whether or not the cash is there, and a system
                # that refuses to record that cannot represent a real
                # state the account can be in.
                cash = cash - amount

    return Portfolio(cash=cash, positions=positions)


def replay_to(events: Sequence[LedgerEvent], seq: int) -> Portfolio:
    """The portfolio as it stood immediately after event `seq`.

    The rewind. Every past report is reproducible because the state it
    was built from can be reconstructed exactly, rather than being
    whatever the stored balance happens to say today.
    """
    return fold(e for e in events if e.seq <= seq)


# ============================================================
# Guards
# ============================================================
# Amounts are signed types, so nothing stops a caller constructing a
# Deposit of -500 and using it as a withdrawal that skips the
# sufficient-funds check. Each event means one direction; the sign is
# not where the direction is expressed.


def _require_positive(amount: Money, event: LedgerEvent, what: str) -> None:
    if amount.is_negative or amount.is_zero:
        raise LedgerError(
            f"event {event.seq}: {what} of {amount} must be positive — "
            f"use the opposite event type rather than a negative amount"
        )


def _require_positive_shares(quantity: Shares, event: LedgerEvent) -> None:
    if quantity.quantity <= 0:
        raise LedgerError(
            f"event {event.seq}: quantity of {quantity} must be positive — "
            "use Buy or Sell to express direction, not the sign"
        )
