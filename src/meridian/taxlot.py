"""Tax lots — which shares you sold, and what it costs.

============================================================
WHY A POSITION IS NOT A NUMBER
============================================================

Say you bought Apple three times: 2019 at $50, 2022 at $170, and last
month at $180. You hold 300 shares. Now you sell 100.

WHICH 100? The answer changes the client's tax bill enormously, and
nothing about "you own 300 shares" can tell you. So this system never
stores a position as a quantity — it stores every purchase separately,
as a LOT, and a position is the sum of the open lots.

Cartons of milk in a fridge, each with its own date and its own price
sticker. Selling "some milk" is not a thing; you take specific cartons.

============================================================
THE RULES THIS IMPLEMENTS
============================================================

Grounded in the actual regulations rather than in folklore. The three
that are most often implemented wrongly:

1. HOLDING PERIOD — IRC section 1223 and IRS Publication 544.
   Counting starts the day AFTER acquisition, and long-term treatment
   requires MORE than one year.

   Bought 1 March 2025, sold 1 March 2026 -> exactly one year -> SHORT
   term. Sold 2 March 2026 -> long term.

   That one-day boundary is a genuine production bug source, and it is
   the first thing worth writing a test for.

2. SPECIFIC IDENTIFICATION — Treas. Reg. section 1.1012-1(c)(3)(i)
   and (c)(8).
   Choosing which lots to sell is not merely a preference. The
   identification must be made by the SETTLEMENT DATE, and the broker
   must confirm it in writing. Miss the deadline and the default
   applies, which for stock is FIFO.

   So a specific-ID disposal carries a deadline, and this module
   records it. An engine that lets an advisor "choose lots" without
   surfacing that deadline is describing a choice the taxpayer may not
   actually have made in time.

3. COVERED VS NON-COVERED — the 1099-B reporting boundary.
   Brokers report adjusted basis only for covered securities: stock
   acquired on or after 1 Jan 2011, mutual funds and DRIP shares from
   1 Jan 2012, debt and options from 1 Jan 2014. For anything older the
   taxpayer reconstructs the basis themselves, and our number is an
   estimate the custodian will not confirm.

   A system that treats a reconstructed basis as authoritative is
   asserting something it cannot support, so the distinction is a field
   on the lot rather than an assumption.

============================================================
WHAT IS DELIBERATELY NOT HERE
============================================================

Wash sales are Phase 05 — the field exists on the lot
(`disallowed_loss_added`) so that basis adjustments have somewhere to
land, but nothing computes it yet.

The AVERAGE COST method is not implemented, and that is a rule rather
than an omission: it is available only for regulated investment company
(mutual fund) shares and DRIP shares, never for ordinary stock. Offering
it as a general option would invite using it where it is not allowed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta
from decimal import Decimal
from enum import Enum
from typing import Final

from meridian.allocate import allocate, normalize_weights
from meridian.money import Money, Price, Shares, Weight

__all__ = [
    "Disposal",
    "DisposalResult",
    "HoldingPeriod",
    "LotError",
    "LotMethod",
    "TaxLot",
    "TaxRates",
    "build_lots",
    "holding_period",
    "is_covered",
    "select_lots",
    "settlement_date",
]


NO_NIIT: Final = Weight("0")
"""Default for TaxRates.niit. Named rather than inlined: a dataclass
default that is a function CALL is evaluated once at class-definition
time and shared across instances."""


class LotError(ValueError):
    """Raised when a disposal cannot be made from the lots on hand."""


# ============================================================
# Holding period
# ============================================================


class HoldingPeriod(Enum):
    SHORT = "short"
    """One year or less. Taxed as ordinary income — the expensive one."""

    LONG = "long"
    """More than one year. 0/15/20% federal, plus NIIT where it applies."""


def holding_period(acquired: date, disposed: date) -> HoldingPeriod:
    """Classify a holding period per IRC section 1223 / Pub. 544.

    The rule as the IRS states it: begin counting on the day FOLLOWING
    acquisition, and the gain is long term only if the asset was held
    for MORE than one year.

    The practical consequence, and the reason this is a function rather
    than a subtraction inline somewhere:

        acquired 2025-03-01, disposed 2026-03-01  ->  SHORT
        acquired 2025-03-01, disposed 2026-03-02  ->  LONG

    Selling one day early converts a 15% or 20% rate into an ordinary
    income rate. It is a large, silent, entirely avoidable cost.
    """
    if disposed < acquired:
        raise LotError(f"disposal date {disposed} precedes acquisition {acquired}")

    try:
        anniversary = acquired.replace(year=acquired.year + 1)
    except ValueError:
        # 29 February has no anniversary in a non-leap year. The
        # convention is to treat 1 March as the anniversary, so a
        # leap-day purchase is not silently granted an extra day of
        # long-term treatment.
        anniversary = date(acquired.year + 1, 3, 1)

    return HoldingPeriod.LONG if disposed > anniversary else HoldingPeriod.SHORT


# ============================================================
# Covered status
# ============================================================


class SecurityKind(Enum):
    """Determines the date from which a broker must report basis."""

    STOCK = "stock"
    FUND = "fund"
    """Regulated investment company shares, and DRIP shares."""
    DEBT_OR_OPTION = "debt_or_option"


_COVERED_FROM: Mapping[SecurityKind, date] = {
    SecurityKind.STOCK: date(2011, 1, 1),
    SecurityKind.FUND: date(2012, 1, 1),
    SecurityKind.DEBT_OR_OPTION: date(2014, 1, 1),
}


def is_covered(acquired: date, kind: SecurityKind = SecurityKind.STOCK) -> bool:
    """Whether the broker is required to report adjusted basis on 1099-B.

    For a NON-covered lot the basis is whatever the taxpayer can
    reconstruct, and the custodian will not confirm it. Treating such a
    number as authoritative asserts something the system cannot support,
    so the flag travels with the lot.
    """
    return acquired >= _COVERED_FROM[kind]


# ============================================================
# Settlement
# ============================================================


def settlement_date(trade_date: date, *, days: int = 1) -> date:
    """The date by which a specific-ID election must be made.

    US equities settle T+1 as of May 2024. Treas. Reg.
    section 1.1012-1(c)(8) requires the identification to be made "at
    the time of the sale," which the regulation clarifies as by the
    settlement date.

    NOTE: this counts business days only as Monday-Friday. Market
    holidays are not yet modelled, so a settlement date falling on a
    holiday will be a day early. A real NYSE calendar arrives with the
    compliance work in Phase 07; hardcoding "business days are
    Mon-Fri" is wrong roughly ten times a year, every year, and that is
    worth being explicit about rather than leaving as a surprise.
    """
    result = trade_date
    remaining = days
    while remaining > 0:
        result += timedelta(days=1)
        if result.weekday() < 5:
            remaining -= 1
    return result


# ============================================================
# The lot
# ============================================================


@dataclass(frozen=True, slots=True)
class TaxLot:
    """One purchase. The atomic unit of a position.

    Basis is stored as the TOTAL for the lot, not per share. Per-share
    basis is a division and therefore inexact; keeping the total exact
    means that splitting a lot can conserve basis to the cent, which
    per-share storage could not guarantee.
    """

    lot_id: str
    ticker: str
    acquired: date
    quantity: Shares
    cost_basis: Money
    """What was paid for the whole lot, including any wash-sale
    adjustment already applied."""

    covered: bool = True
    disallowed_loss_added: Money | None = None
    """Wash-sale basis adjustment carried into this lot. Populated in
    Phase 05; present now so the field does not have to be retrofitted
    onto records that already exist."""

    def __post_init__(self) -> None:
        if self.quantity.quantity <= 0:
            raise LotError(f"lot {self.lot_id}: quantity must be positive")

    def basis_per_share(self) -> Decimal:
        """Inexact by nature — use only for RANKING lots, never for
        computing a disposal's basis. Splitting uses allocate() so the
        cents conserve exactly."""
        return self.cost_basis.amount / self.quantity.quantity

    def market_value(self, price: Price) -> Money:
        return self.quantity.value_at(price)

    def unrealized(self, price: Price) -> Money:
        return self.market_value(price) - self.cost_basis

    def period_at(self, on: date) -> HoldingPeriod:
        return holding_period(self.acquired, on)

    def split(self, quantity: Shares) -> tuple[TaxLot, TaxLot]:
        """Divide into (sold, retained), conserving basis exactly.

        The basis split runs through the SAME allocate() that divides
        money everywhere else in this system — the third caller of the
        largest remainder method, after cash allocation and weight
        normalisation.

        That is the point of having one audited implementation. Dividing
        basis by hand here would be a second place for a cent to go
        missing, and a cent of basis lost is a cent of phantom gain that
        someone eventually pays tax on.
        """
        if quantity.quantity <= 0:
            raise LotError("split quantity must be positive")
        if quantity.quantity > self.quantity.quantity:
            raise LotError(
                f"lot {self.lot_id} holds {self.quantity}, cannot split {quantity}"
            )
        if quantity.quantity == self.quantity.quantity:
            raise LotError("splitting a whole lot is a disposal, not a split")

        retained = Shares(self.quantity.quantity - quantity.quantity)

        parts = allocate(
            self.cost_basis,
            normalize_weights(
                {"sold": quantity.quantity, "retained": retained.quantity}
            ),
        )

        return (
            replace(
                self,
                lot_id=f"{self.lot_id}#s",
                quantity=quantity,
                cost_basis=parts["sold"],
            ),
            replace(
                self,
                lot_id=f"{self.lot_id}#r",
                quantity=retained,
                cost_basis=parts["retained"],
            ),
        )


# ============================================================
# Tax rates
# ============================================================


@dataclass(frozen=True, slots=True)
class TaxRates:
    """The client's marginal rates. Supplied, never assumed.

    A rate depends on the taxpayer's bracket, state, and filing status,
    none of which this engine knows. Guessing them would produce a
    tax estimate that looks authoritative and is not.
    """

    short_term: Weight
    """Ordinary income rate — 2026 federal brackets run to 37%."""

    long_term: Weight
    """0%, 15%, or 20% federal depending on taxable income."""

    niit: Weight = NO_NIIT
    """Net investment income tax: 3.8% above $200k single / $250k MFJ
    of modified AGI. Applies on top of the capital gains rate, so a
    15% gain really costs 18.8% and a 20% gain costs 23.8%."""

    def rate_for(self, period: HoldingPeriod) -> Weight:
        base = self.short_term if period is HoldingPeriod.SHORT else self.long_term
        return Weight(base.value + self.niit.value)


# ============================================================
# Selection
# ============================================================


class LotMethod(Enum):
    """How to choose which lots to sell."""

    FIFO = "fifo"
    """Oldest first. The IRS DEFAULT for stock when no adequate
    identification is made. Tends to realise long-term gains."""

    LIFO = "lifo"
    """Newest first. Often minimises gain in a rising market, but
    realises short-term gains, which are taxed at ordinary rates."""

    HIFO = "hifo"
    """Highest cost basis first. Minimises realised gain this year, and
    the usual default for a tax-sensitive account."""

    MIN_TAX = "min_tax"
    """Lowest tax cost first, holding period aware. Ranks by the actual
    tax each share would cost, which produces the conventional ordering
    — short-term losses, then long-term losses, then long-term gains,
    then short-term gains — as a consequence rather than as a hardcoded
    list."""

    SPECIFIC_ID = "specific_id"
    """The advisor names the lots. Requires identification by the
    settlement date with written broker confirmation."""


def select_lots(
    lots: Sequence[TaxLot],
    quantity: Shares,
    method: LotMethod,
    *,
    on: date,
    price: Price,
    rates: TaxRates | None = None,
    chosen: Sequence[str] | None = None,
) -> list[tuple[TaxLot, Shares]]:
    """Choose which lots satisfy a sale of `quantity`.

    Returns (lot, quantity_from_that_lot) pairs, in the order they
    should be consumed. The final pair may be a partial lot.

    Raises:
        LotError: if the lots on hand do not cover the quantity.
    """
    if quantity.quantity <= 0:
        raise LotError("disposal quantity must be positive")

    available = sum((lot.quantity.quantity for lot in lots), Decimal(0))
    if quantity.quantity > available:
        raise LotError(
            f"cannot dispose of {quantity}; only {Shares(available)} held across "
            f"{len(lots)} lot(s)"
        )

    ordered = _order_lots(lots, method, on=on, price=price, rates=rates, chosen=chosen)

    picked: list[tuple[TaxLot, Shares]] = []
    remaining = quantity.quantity

    for lot in ordered:
        if remaining <= 0:
            break
        take = min(remaining, lot.quantity.quantity)
        picked.append((lot, Shares(take)))
        remaining -= take

    if remaining > 0:
        # Only reachable under SPECIFIC_ID, where the named lots may not
        # add up. Every other method sees all lots and the total was
        # checked above.
        raise LotError(
            f"the identified lots hold {quantity.quantity - remaining} but "
            f"{quantity} was requested"
        )

    return picked


def _order_lots(
    lots: Sequence[TaxLot],
    method: LotMethod,
    *,
    on: date,
    price: Price,
    rates: TaxRates | None,
    chosen: Sequence[str] | None,
) -> list[TaxLot]:
    """Rank lots by the chosen method.

    Every sort key ends in `lot_id` as a total order. Two lots bought
    the same day at the same price are genuinely interchangeable for
    tax purposes, but they must still be SELECTED in a stable order —
    otherwise the same instruction produces different lot records on
    different runs, and a past disposal cannot be reproduced.
    """
    if method is LotMethod.SPECIFIC_ID:
        if not chosen:
            raise LotError("SPECIFIC_ID requires the lot ids to be named")
        by_id = {lot.lot_id: lot for lot in lots}
        missing = [lot_id for lot_id in chosen if lot_id not in by_id]
        if missing:
            raise LotError(f"identified lots not held: {', '.join(missing)}")
        return [by_id[lot_id] for lot_id in chosen]

    if method is LotMethod.FIFO:
        return sorted(lots, key=lambda lot: (lot.acquired, lot.lot_id))

    if method is LotMethod.LIFO:
        return sorted(lots, key=lambda lot: (lot.acquired, lot.lot_id), reverse=True)

    if method is LotMethod.HIFO:
        return sorted(lots, key=lambda lot: (-lot.basis_per_share(), lot.lot_id))

    if rates is None:
        raise LotError("MIN_TAX requires the client's tax rates")

    def tax_per_share(lot: TaxLot) -> Decimal:
        gain = price.amount - lot.basis_per_share()
        rate = rates.rate_for(lot.period_at(on)).value
        return gain * rate

    return sorted(lots, key=lambda lot: (tax_per_share(lot), lot.lot_id))


# ============================================================
# Disposal
# ============================================================


@dataclass(frozen=True, slots=True)
class Disposal:
    """One lot sold, in whole or in part."""

    lot_id: str
    ticker: str
    acquired: date
    disposed: date
    quantity: Shares
    proceeds: Money
    cost_basis: Money
    period: HoldingPeriod
    covered: bool

    @property
    def gain(self) -> Money:
        """Positive is a gain, negative is a loss."""
        return self.proceeds - self.cost_basis

    @property
    def is_loss(self) -> bool:
        return self.gain.is_negative

    def tax(self, rates: TaxRates) -> Money:
        """What this disposal costs in tax. Negative for a loss, which
        offsets other gains rather than producing a refund on its own."""
        return self.gain * rates.rate_for(self.period)

    def __str__(self) -> str:
        kind = "loss" if self.is_loss else "gain"
        return (
            f"{self.quantity} {self.ticker} acquired {self.acquired} "
            f"({self.period.value}) — {kind} of {abs(self.gain)}"
        )


@dataclass(frozen=True, slots=True)
class DisposalResult:
    """The outcome of a sale: what was sold, and what is left."""

    disposals: tuple[Disposal, ...]
    remaining_lots: tuple[TaxLot, ...]
    method: LotMethod
    identification_deadline: date | None
    """For SPECIFIC_ID: the settlement date by which the election must
    reach the broker, per Treas. Reg. 1.1012-1(c)(8). None for the
    default methods, which need no election."""

    @property
    def proceeds(self) -> Money:
        return sum((d.proceeds for d in self.disposals), Money.zero())

    @property
    def cost_basis(self) -> Money:
        return sum((d.cost_basis for d in self.disposals), Money.zero())

    @property
    def realized_gain(self) -> Money:
        return self.proceeds - self.cost_basis

    def gain_by_period(self) -> dict[HoldingPeriod, Money]:
        """Split the result, because the two halves are taxed
        differently and are reported on different parts of Form 8949."""
        totals = {HoldingPeriod.SHORT: Money.zero(), HoldingPeriod.LONG: Money.zero()}
        for d in self.disposals:
            totals[d.period] = totals[d.period] + d.gain
        return totals

    def tax(self, rates: TaxRates) -> Money:
        return sum((d.tax(rates) for d in self.disposals), Money.zero())


def dispose(
    lots: Sequence[TaxLot],
    quantity: Shares,
    price: Price,
    *,
    on: date,
    method: LotMethod = LotMethod.FIFO,
    rates: TaxRates | None = None,
    chosen: Sequence[str] | None = None,
) -> DisposalResult:
    """Sell `quantity` at `price`, choosing lots by `method`.

    Conserves both quantity and basis exactly: what leaves in disposals
    plus what stays in remaining_lots equals what was there before.
    """
    picked = select_lots(
        lots, quantity, method, on=on, price=price, rates=rates, chosen=chosen
    )

    consumed = {lot.lot_id: qty for lot, qty in picked}

    disposals: list[Disposal] = []
    remaining: list[TaxLot] = []

    for lot in lots:
        taken = consumed.get(lot.lot_id)

        if taken is None:
            remaining.append(lot)
            continue

        if taken.quantity == lot.quantity.quantity:
            sold_part, kept_part = lot, None
        else:
            sold_part, kept_part = lot.split(taken)

        disposals.append(
            Disposal(
                lot_id=lot.lot_id,
                ticker=lot.ticker,
                acquired=lot.acquired,
                disposed=on,
                quantity=sold_part.quantity,
                proceeds=sold_part.quantity.value_at(price),
                cost_basis=sold_part.cost_basis,
                period=holding_period(lot.acquired, on),
                covered=lot.covered,
            )
        )
        if kept_part is not None:
            remaining.append(kept_part)

    deadline = settlement_date(on) if method is LotMethod.SPECIFIC_ID else None

    return DisposalResult(
        disposals=tuple(disposals),
        remaining_lots=tuple(remaining),
        method=method,
        identification_deadline=deadline,
    )


# ============================================================
# Building lots from the ledger
# ============================================================


def build_lots(
    events: Iterable[object],
    *,
    method: LotMethod = LotMethod.FIFO,
    rates: TaxRates | None = None,
) -> dict[str, list[TaxLot]]:
    """Replay a ledger event stream into open tax lots per ticker.

    Buys open lots; sells consume them by `method`. The same fold shape
    as the ledger's, carrying lots instead of a share count — because a
    share count is exactly the information that turns out not to be
    enough.
    """
    from meridian.ledger import Buy, Sell  # local: avoids a cycle

    lots: dict[str, list[TaxLot]] = {}

    for event in events:
        if isinstance(event, Buy):
            lots.setdefault(event.ticker, []).append(
                TaxLot(
                    lot_id=f"{event.ticker}-{event.seq}",
                    ticker=event.ticker,
                    acquired=event.on,
                    quantity=event.quantity,
                    cost_basis=event.consideration,
                    covered=is_covered(event.on),
                )
            )
        elif isinstance(event, Sell):
            held = lots.get(event.ticker, [])
            result = dispose(
                held,
                event.quantity,
                event.price,
                on=event.on,
                method=method,
                rates=rates,
            )
            lots[event.ticker] = list(result.remaining_lots)

    return {ticker: open_lots for ticker, open_lots in lots.items() if open_lots}
