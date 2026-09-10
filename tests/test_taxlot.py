"""Tests for tax lots.

Two themes:

  1. CONSERVATION, again. Splitting a lot must not lose a cent of basis
     — a cent of basis lost is a cent of phantom gain someone
     eventually pays tax on.

  2. THE RULES AS WRITTEN. Holding periods, the specific-ID deadline,
     and covered-security boundaries are checked against the actual
     regulations, not against what seems reasonable.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from meridian.ledger import Buy, Deposit, LedgerEvent, Sell
from meridian.money import Money, Price, Shares, Weight
from meridian.taxlot import (
    HoldingPeriod,
    LotError,
    LotMethod,
    SecurityKind,
    TaxLot,
    TaxRates,
    build_lots,
    dispose,
    holding_period,
    is_covered,
    select_lots,
    settlement_date,
)

# A high-bracket taxable client: 37% ordinary, 20% long-term, plus NIIT.
# Effective: 40.8% short, 23.8% long.
HIGH = TaxRates(
    short_term=Weight("0.37"), long_term=Weight("0.20"), niit=Weight("0.038")
)


def lot(
    lot_id: str,
    acquired: str,
    quantity: str,
    basis: str,
    ticker: str = "AAPL",
) -> TaxLot:
    return TaxLot(
        lot_id=lot_id,
        ticker=ticker,
        acquired=date.fromisoformat(acquired),
        quantity=Shares(quantity),
        cost_basis=Money(basis),
    )


# ============================================================
# HOLDING PERIOD — the one-day boundary
# ============================================================


def test_exactly_one_year_is_short_term() -> None:
    """IRC 1223 / Pub. 544: counting starts the day AFTER acquisition,
    and long-term requires MORE than one year.

    Bought 1 March 2025, sold 1 March 2026 is exactly one year — which
    is NOT more than one year. Short term.
    """
    assert holding_period(date(2025, 3, 1), date(2026, 3, 1)) is HoldingPeriod.SHORT


def test_one_year_and_a_day_is_long_term() -> None:
    assert holding_period(date(2025, 3, 1), date(2026, 3, 2)) is HoldingPeriod.LONG


def test_the_boundary_is_worth_real_money() -> None:
    """Selling one day early converts 23.8% into 40.8%."""
    early = lot("A", "2025-03-01", "100", "10000.00")
    sold_early = dispose([early], Shares("100"), Price("200.00"), on=date(2026, 3, 1))
    sold_late = dispose([early], Shares("100"), Price("200.00"), on=date(2026, 3, 2))

    assert sold_early.disposals[0].period is HoldingPeriod.SHORT
    assert sold_late.disposals[0].period is HoldingPeriod.LONG

    # Same $10,000 gain, very different bills.
    assert sold_early.tax(HIGH) == Money("4080.00")
    assert sold_late.tax(HIGH) == Money("2380.00")


def test_a_leap_day_purchase_has_no_free_extra_day() -> None:
    """29 February has no anniversary in a non-leap year. Treating
    1 March as the anniversary stops a leap-day purchase silently
    qualifying for long-term treatment a day early."""
    acquired = date(2024, 2, 29)
    assert holding_period(acquired, date(2025, 3, 1)) is HoldingPeriod.SHORT
    assert holding_period(acquired, date(2025, 3, 2)) is HoldingPeriod.LONG


def test_same_day_sale_is_short_term() -> None:
    assert holding_period(date(2026, 1, 5), date(2026, 1, 5)) is HoldingPeriod.SHORT


def test_selling_before_buying_is_refused() -> None:
    with pytest.raises(LotError, match="precedes acquisition"):
        holding_period(date(2026, 3, 1), date(2025, 3, 1))


# ============================================================
# CONSERVATION — splitting a lot
# ============================================================


def test_splitting_conserves_quantity_and_basis() -> None:
    original = lot("A", "2020-01-01", "100", "5000.00")
    sold, kept = original.split(Shares("30"))

    assert sold.quantity + kept.quantity == original.quantity
    assert sold.cost_basis + kept.cost_basis == original.cost_basis


def test_splitting_a_basis_that_does_not_divide_evenly() -> None:
    """$100.00 across 3 shares is $33.333... each. The cent has to land
    somewhere documented rather than evaporate."""
    original = lot("A", "2020-01-01", "3", "100.00")
    sold, kept = original.split(Shares("1"))
    assert sold.cost_basis + kept.cost_basis == Money("100.00")


@given(
    quantity=st.integers(min_value=2, max_value=10_000),
    basis_cents=st.integers(min_value=1, max_value=100_000_000),
    take=st.integers(min_value=1, max_value=9_999),
)
@settings(max_examples=400)
def test_split_conserves_basis_for_any_lot(
    quantity: int, basis_cents: int, take: int
) -> None:
    """The conservation property, applied to cost basis.

    A cent of basis lost here is a cent of phantom gain that someone
    eventually pays tax on — the same failure as losing a cent of cash,
    just deferred and harder to spot.
    """
    if take >= quantity:
        take = quantity - 1

    original = TaxLot(
        lot_id="A",
        ticker="AAPL",
        acquired=date(2020, 1, 1),
        quantity=Shares(quantity),
        cost_basis=Money.from_cents(basis_cents),
    )
    sold, kept = original.split(Shares(take))

    assert sold.quantity + kept.quantity == original.quantity
    assert sold.cost_basis + kept.cost_basis == original.cost_basis


def test_splitting_a_whole_lot_is_refused() -> None:
    """That is a disposal, not a split, and calling it a split would
    leave a zero-quantity lot lying around."""
    original = lot("A", "2020-01-01", "100", "5000.00")
    with pytest.raises(LotError, match="disposal, not a split"):
        original.split(Shares("100"))


def test_splitting_more_than_held_is_refused() -> None:
    original = lot("A", "2020-01-01", "100", "5000.00")
    with pytest.raises(LotError, match="cannot split"):
        original.split(Shares("101"))


# ============================================================
# SELECTION METHODS
# ============================================================

LOTS = [
    lot("old-cheap", "2019-06-01", "100", "5000.00"),  # $50/sh, long
    lot("mid-dear", "2022-06-01", "100", "17000.00"),  # $170/sh, long
    lot("new-dear", "2026-06-01", "100", "18000.00"),  # $180/sh, short
]
SALE_ON = date(2026, 9, 8)
SALE_PRICE = Price("175.00")


def pick(
    lots: list[TaxLot],
    quantity: str,
    method: LotMethod,
    *,
    rates: TaxRates | None = None,
    chosen: list[str] | None = None,
) -> list[tuple[TaxLot, Shares]]:
    """select_lots with the sale fixed, so the tests read as the thing
    they are actually about: which lots come back, in which order."""
    return select_lots(
        lots,
        Shares(quantity),
        method,
        on=SALE_ON,
        price=SALE_PRICE,
        rates=rates,
        chosen=chosen,
    )


def test_fifo_takes_the_oldest() -> None:
    """The IRS default for stock when no adequate identification is
    made. Tends to realise long-term gains."""
    picked = pick(LOTS, "150", LotMethod.FIFO)
    assert [lot_.lot_id for lot_, _ in picked] == ["old-cheap", "mid-dear"]
    assert picked[0][1] == Shares("100")
    assert picked[1][1] == Shares("50")


def test_lifo_takes_the_newest() -> None:
    picked = pick(LOTS, "150", LotMethod.LIFO)
    assert [lot_.lot_id for lot_, _ in picked] == ["new-dear", "mid-dear"]


def test_hifo_takes_the_highest_basis() -> None:
    """Minimises realised gain this year — the usual default for a
    tax-sensitive account."""
    picked = pick(LOTS, "150", LotMethod.HIFO)
    assert [lot_.lot_id for lot_, _ in picked] == ["new-dear", "mid-dear"]


def test_min_tax_prefers_losses_and_then_the_cheaper_rate() -> None:
    """At $175 the new lot ($180 basis) is a SHORT-TERM LOSS, the mid
    lot ($170) is a small long-term gain, the old lot ($50) is a large
    long-term gain.

    Ranking by actual tax per share produces the conventional ordering
    without it being hardcoded: take the loss first, then the smaller
    gain, then the larger one.
    """
    picked = pick(LOTS, "300", LotMethod.MIN_TAX, rates=HIGH)
    assert [lot_.lot_id for lot_, _ in picked] == [
        "new-dear",  # short-term loss, worth the most as an offset
        "mid-dear",  # small long-term gain
        "old-cheap",  # large long-term gain
    ]


def test_min_tax_beats_fifo_on_the_tax_bill() -> None:
    """The whole reason the method exists."""
    fifo = dispose(
        LOTS, Shares("100"), Price("175.00"), method=LotMethod.FIFO, on=SALE_ON
    )
    smart = dispose(
        LOTS,
        Shares("100"),
        Price("175.00"),
        method=LotMethod.MIN_TAX,
        rates=HIGH,
        on=SALE_ON,
    )
    assert smart.tax(HIGH) < fifo.tax(HIGH)


def test_min_tax_requires_rates() -> None:
    """Guessing a client's bracket would produce an authoritative-looking
    number that is not."""
    with pytest.raises(LotError, match="requires the client's tax rates"):
        pick(LOTS, "50", LotMethod.MIN_TAX)


def test_selection_is_deterministic_under_ties() -> None:
    """Two lots bought the same day at the same price are
    interchangeable for tax purposes but must still be SELECTED in a
    stable order, or a past disposal cannot be reproduced."""
    tied = [
        lot("B", "2020-01-01", "10", "1000.00"),
        lot("A", "2020-01-01", "10", "1000.00"),
    ]
    first = pick(tied, "10", LotMethod.FIFO)
    second = pick(list(reversed(tied)), "10", LotMethod.FIFO)
    assert [lot_.lot_id for lot_, _ in first] == [lot_.lot_id for lot_, _ in second]


# ============================================================
# SPECIFIC IDENTIFICATION — the deadline is part of the rule
# ============================================================


def test_specific_id_carries_a_settlement_deadline() -> None:
    """Treas. Reg. 1.1012-1(c)(8): the identification must be made by
    the SETTLEMENT date, with written broker confirmation. An engine
    that lets an advisor pick lots without surfacing that deadline is
    describing a choice the taxpayer may not have made in time."""
    result = dispose(
        LOTS,
        Shares("100"),
        Price("175.00"),
        on=date(2026, 9, 8),  # a Tuesday
        method=LotMethod.SPECIFIC_ID,
        chosen=["mid-dear"],
    )
    assert result.identification_deadline == date(2026, 9, 9)


def test_default_methods_need_no_election() -> None:
    result = dispose(LOTS, Shares("100"), Price("175.00"), on=date(2026, 9, 8))
    assert result.identification_deadline is None


def test_settlement_skips_the_weekend() -> None:
    """T+1 since May 2024. Friday trades settle Monday."""
    assert settlement_date(date(2026, 9, 11)) == date(2026, 9, 14)  # Fri -> Mon


def test_specific_id_needs_the_lots_named() -> None:
    with pytest.raises(LotError, match="requires the lot ids"):
        pick(LOTS, "50", LotMethod.SPECIFIC_ID)


def test_identifying_a_lot_not_held_is_refused() -> None:
    with pytest.raises(LotError, match="not held"):
        pick(LOTS, "50", LotMethod.SPECIFIC_ID, chosen=["ghost"])


def test_identified_lots_must_cover_the_sale() -> None:
    with pytest.raises(LotError, match="was requested"):
        pick(LOTS, "150", LotMethod.SPECIFIC_ID, chosen=["mid-dear"])


# ============================================================
# COVERED SECURITIES — the 1099-B boundary
# ============================================================


def test_the_covered_boundaries() -> None:
    """Stock from 2011, funds and DRIP from 2012, debt and options from
    2014. Before those dates the taxpayer reconstructs the basis and the
    custodian will not confirm it."""
    assert not is_covered(date(2010, 12, 31), SecurityKind.STOCK)
    assert is_covered(date(2011, 1, 1), SecurityKind.STOCK)

    assert not is_covered(date(2011, 6, 1), SecurityKind.FUND)
    assert is_covered(date(2012, 1, 1), SecurityKind.FUND)

    assert not is_covered(date(2013, 6, 1), SecurityKind.DEBT_OR_OPTION)
    assert is_covered(date(2014, 1, 1), SecurityKind.DEBT_OR_OPTION)


def test_covered_status_travels_with_the_disposal() -> None:
    """It has to reach Form 8949, where non-covered lots are reported
    differently."""
    legacy = TaxLot(
        lot_id="legacy",
        ticker="AAPL",
        acquired=date(2008, 5, 1),
        quantity=Shares("100"),
        cost_basis=Money("1000.00"),
        covered=False,
    )
    result = dispose([legacy], Shares("100"), Price("175.00"), on=date(2026, 9, 8))
    assert not result.disposals[0].covered


# ============================================================
# DISPOSAL ARITHMETIC
# ============================================================


def test_gain_and_tax_are_computed_per_lot() -> None:
    result = dispose(
        LOTS, Shares("100"), Price("175.00"), method=LotMethod.FIFO, on=date(2026, 9, 8)
    )
    d = result.disposals[0]
    assert d.proceeds == Money("17500.00")
    assert d.cost_basis == Money("5000.00")
    assert d.gain == Money("12500.00")
    assert d.period is HoldingPeriod.LONG
    assert d.tax(HIGH) == Money("2975.00")  # 12,500 x 23.8%


def test_a_loss_produces_a_negative_tax() -> None:
    """It offsets other gains rather than producing a refund on its
    own, but the sign is what makes that arithmetic work."""
    result = dispose(
        [lot("A", "2026-06-01", "100", "18000.00")],
        Shares("100"),
        Price("175.00"),
        on=date(2026, 9, 8),
    )
    assert result.disposals[0].is_loss
    assert result.tax(HIGH).is_negative


def test_gains_are_split_by_holding_period() -> None:
    """The two halves are taxed differently and reported on different
    parts of Form 8949."""
    result = dispose(
        LOTS, Shares("300"), Price("175.00"), method=LotMethod.FIFO, on=date(2026, 9, 8)
    )
    split = result.gain_by_period()
    assert split[HoldingPeriod.LONG] == Money("12500.00") + Money("500.00")
    assert split[HoldingPeriod.SHORT] == Money("-500.00")


def test_disposing_more_than_held_is_refused() -> None:
    with pytest.raises(LotError, match="only"):
        dispose(LOTS, Shares("500"), Price("175.00"), on=date(2026, 9, 8))


# ============================================================
# CONSERVATION ACROSS A DISPOSAL
# ============================================================


@given(
    take=st.integers(min_value=1, max_value=300),
    method=st.sampled_from([LotMethod.FIFO, LotMethod.LIFO, LotMethod.HIFO]),
)
@settings(max_examples=300)
def test_a_disposal_conserves_quantity_and_basis(take: int, method: LotMethod) -> None:
    """Whatever leaves plus whatever stays equals what was there.

    The same round-trip idea as the ledger: two independent routes to
    the same number, which a bug would have to corrupt identically to
    slip past.
    """
    result = dispose(
        LOTS, Shares(take), Price("175.00"), method=method, on=date(2026, 9, 8)
    )

    quantity_before = sum((lot_.quantity.quantity for lot_ in LOTS), Decimal(0))
    quantity_out = sum((d.quantity.quantity for d in result.disposals), Decimal(0))
    quantity_left = sum(
        (lot_.quantity.quantity for lot_ in result.remaining_lots), Decimal(0)
    )
    assert quantity_out + quantity_left == quantity_before

    basis_before = sum((lot_.cost_basis for lot_ in LOTS), Money.zero())
    basis_out = sum((d.cost_basis for d in result.disposals), Money.zero())
    basis_left = sum((lot_.cost_basis for lot_ in result.remaining_lots), Money.zero())
    assert basis_out + basis_left == basis_before


@given(take=st.integers(min_value=1, max_value=300))
@settings(max_examples=200)
def test_every_method_disposes_the_requested_quantity(take: int) -> None:
    for method in (LotMethod.FIFO, LotMethod.LIFO, LotMethod.HIFO):
        result = dispose(
            LOTS, Shares(take), Price("175.00"), method=method, on=date(2026, 9, 8)
        )
        total = sum((d.quantity.quantity for d in result.disposals), Decimal(0))
        assert total == Decimal(take)


# ============================================================
# BUILDING LOTS FROM THE LEDGER
# ============================================================


def test_lots_are_built_from_an_event_stream() -> None:
    """The same fold shape as the ledger's, carrying lots instead of a
    share count — because a share count is exactly the information that
    turns out not to be enough."""
    events: list[LedgerEvent] = [
        Deposit(1, date(2024, 1, 2), Money("100000.00")),
        Buy(2, date(2024, 1, 2), "VTI", Shares("100"), Price("140.00")),
        Buy(3, date(2025, 6, 2), "VTI", Shares("100"), Price("180.00")),
        Sell(4, date(2026, 9, 8), "VTI", Shares("50"), Price("200.00")),
    ]
    lots = build_lots(events, method=LotMethod.FIFO)

    assert len(lots["VTI"]) == 2
    total = sum((lot_.quantity.quantity for lot_ in lots["VTI"]), Decimal(0))
    assert total == Decimal(150)

    # FIFO consumed half the 2024 lot, so its basis halved.
    oldest = min(lots["VTI"], key=lambda lot_: lot_.acquired)
    assert oldest.cost_basis == Money("7000.00")


def test_lots_from_the_ledger_carry_covered_status() -> None:
    events: list[LedgerEvent] = [
        Deposit(1, date(2009, 5, 1), Money("100000.00")),
        Buy(2, date(2009, 5, 1), "VTI", Shares("100"), Price("50.00")),
    ]
    lots = build_lots(events)
    assert not lots["VTI"][0].covered


def test_a_fully_sold_position_leaves_no_empty_lots() -> None:
    events: list[LedgerEvent] = [
        Deposit(1, date(2024, 1, 2), Money("100000.00")),
        Buy(2, date(2024, 1, 2), "VTI", Shares("100"), Price("140.00")),
        Sell(3, date(2026, 9, 8), "VTI", Shares("100"), Price("200.00")),
    ]
    assert "VTI" not in build_lots(events)


# ============================================================
# TAX RATES
# ============================================================


def test_niit_stacks_on_top_of_the_capital_gains_rate() -> None:
    """A 15% gain really costs 18.8%; a 20% gain costs 23.8%."""
    mid = TaxRates(
        short_term=Weight("0.32"), long_term=Weight("0.15"), niit=Weight("0.038")
    )
    assert mid.rate_for(HoldingPeriod.LONG) == Weight("0.188")
    assert mid.rate_for(HoldingPeriod.SHORT) == Weight("0.358")
