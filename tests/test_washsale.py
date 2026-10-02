"""Tests for wash sales — IRC section 1091.

The rules here are checked against the regulations as written, not
against what seems reasonable. Several of them are counterintuitive, and
the counterintuitive ones are where client money is lost.
"""

from __future__ import annotations

from datetime import date

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from meridian.household import Account, AccountType, Household
from meridian.money import Money, Price, Shares, Weight
from meridian.taxlot import Disposal, HoldingPeriod, TaxLot, TaxRates, dispose
from meridian.washsale import (
    Acquisition,
    HarvestOpportunity,
    MatchOutcome,
    SubstituteMap,
    apply_basis_adjustment,
    blackout_tickers,
    find_harvest_opportunities,
    find_wash_sales,
    wash_sale_window,
    would_trigger_wash_sale,
)

HIGH = TaxRates(
    short_term=Weight("0.37"), long_term=Weight("0.20"), niit=Weight("0.038")
)

# The firm's policy. VOO/IVV/SPLG all track the S&P 500 and this firm
# has decided to treat them as substantially identical — a defensible
# reading of an undefined term, recorded as data rather than assumed.
POLICY = SubstituteMap.symmetric(
    groups=[["VOO", "IVV", "SPLG"]],
    alternatives={"VOO": ("VTI",), "IVV": ("VTI",)},
)

SALE_DATE = date(2026, 6, 15)


def loss_disposal(
    ticker: str = "VOO",
    quantity: str = "100",
    proceeds: str = "9000.00",
    basis: str = "10000.00",
    disposed: date = SALE_DATE,
    acquired: date = date(2025, 1, 10),
    lot_id: str = "lot-1",
) -> Disposal:
    return Disposal(
        lot_id=lot_id,
        ticker=ticker,
        acquired=acquired,
        disposed=disposed,
        quantity=Shares(quantity),
        proceeds=Money(proceeds),
        cost_basis=Money(basis),
        period=HoldingPeriod.LONG,
        covered=True,
    )


def buy(
    acq_id: str,
    on: date,
    quantity: str = "100",
    ticker: str = "VOO",
    account: str = "taxable-1",
    account_type: AccountType = AccountType.TAXABLE,
    reinvestment: bool = False,
) -> Acquisition:
    return Acquisition(
        acquisition_id=acq_id,
        account_id=account,
        account_type=account_type,
        ticker=ticker,
        on=on,
        quantity=Shares(quantity),
        is_reinvestment=reinvestment,
    )


# ============================================================
# THE WINDOW
# ============================================================


def test_the_window_is_61_days() -> None:
    """30 before, the day of sale, 30 after. Calendar days — weekends
    and market holidays are inside it."""
    start, end = wash_sale_window(SALE_DATE)
    assert start == date(2026, 5, 16)
    assert end == date(2026, 7, 15)
    assert (end - start).days + 1 == 61


def test_a_purchase_31_days_before_is_outside() -> None:
    report = find_wash_sales([loss_disposal()], [buy("a", date(2026, 5, 15))], POLICY)
    assert not report


def test_a_purchase_30_days_before_is_inside() -> None:
    """The rule looks BACKWARD as well as forward. Buying, then selling
    an older lot at a loss, is a wash sale — which surprises people."""
    report = find_wash_sales([loss_disposal()], [buy("a", date(2026, 5, 16))], POLICY)
    assert report
    assert report.total_disallowed == Money("-1000.00")


def test_a_purchase_30_days_after_is_inside() -> None:
    report = find_wash_sales([loss_disposal()], [buy("a", date(2026, 7, 15))], POLICY)
    assert report


def test_a_purchase_31_days_after_is_outside() -> None:
    report = find_wash_sales([loss_disposal()], [buy("a", date(2026, 7, 16))], POLICY)
    assert not report


def test_a_gain_is_never_washed() -> None:
    """Section 1091 disallows LOSSES. A gain is taxable now whatever you
    buy afterwards."""
    gain = loss_disposal(proceeds="12000.00", basis="10000.00")
    assert find_wash_sales([gain], [buy("a", SALE_DATE)], POLICY).findings == ()


# ============================================================
# THE IRA TRAP — Rev. Rul. 2008-5
# ============================================================


def test_replacement_in_an_ira_forfeits_the_loss_permanently() -> None:
    """The most valuable thing this engine catches.

    Normally a disallowed loss is added to the replacement's basis and
    comes back later — deferred, not lost. But section 1091(d) does not
    increase an IRA's basis, and basis inside an IRA is irrelevant to
    the taxpayer anyway.

    The loss is simply gone. An engine that reports every wash sale as a
    deferral is telling the client they will get a deduction they never
    will.
    """
    report = find_wash_sales(
        [loss_disposal()],
        [buy("ira", SALE_DATE, account="ira-1", account_type=AccountType.ROTH_IRA)],
        POLICY,
    )
    finding = report.findings[0]

    assert finding.replacements[0].outcome is MatchOutcome.FORFEITED
    assert report.total_forfeited == Money("-1000.00")
    assert report.total_deferred == Money.zero()


def test_replacement_in_a_taxable_account_only_defers() -> None:
    report = find_wash_sales([loss_disposal()], [buy("tax", SALE_DATE)], POLICY)
    finding = report.findings[0]

    assert finding.replacements[0].outcome is MatchOutcome.DEFERRED
    assert report.total_deferred == Money("-1000.00")
    assert report.total_forfeited == Money.zero()


def test_a_split_replacement_splits_the_outcome() -> None:
    """Half the shares replaced in a taxable account, half in an IRA:
    half the loss deferred, half destroyed."""
    report = find_wash_sales(
        [loss_disposal()],
        [
            buy("tax", date(2026, 6, 16), quantity="50"),
            buy(
                "ira",
                date(2026, 6, 17),
                quantity="50",
                account="ira-1",
                account_type=AccountType.TRADITIONAL_IRA,
            ),
        ],
        POLICY,
    )
    assert report.total_deferred == Money("-500.00")
    assert report.total_forfeited == Money("-500.00")
    assert report.total_disallowed == Money("-1000.00")


# ============================================================
# HOUSEHOLD SCOPE
# ============================================================


def test_a_spouses_purchase_washes_the_sale() -> None:
    """The rule follows the taxpayer, not the account. Anything scoped
    to a single account reports clean harvests that are not clean."""
    report = find_wash_sales(
        [loss_disposal()],
        [buy("spouse", SALE_DATE, account="taxable-2")],
        POLICY,
    )
    assert report


def test_dividend_reinvestment_triggers_the_rule() -> None:
    """The most common accidental wash sale in real portfolios, and
    invisible unless you model lots."""
    report = find_wash_sales(
        [loss_disposal()],
        [buy("drip", date(2026, 6, 30), quantity="2", reinvestment=True)],
        POLICY,
    )
    assert report
    assert report.findings[0].replacements[0].is_reinvestment
    # Two shares out of a hundred: 2% of the loss.
    assert report.total_disallowed == Money("-20.00")


# ============================================================
# PROPORTIONAL DISALLOWANCE — section 1091(b)
# ============================================================


def test_partial_replacement_disallows_proportionally() -> None:
    """Replace 40 of 100 shares and 40% of the loss is disallowed. The
    other 60% is still deductible this year."""
    report = find_wash_sales(
        [loss_disposal()], [buy("a", SALE_DATE, quantity="40")], POLICY
    )
    finding = report.findings[0]

    assert finding.matched == Shares("40")
    assert finding.disallowed == Money("-400.00")
    assert finding.allowed == Money("-600.00")
    assert not finding.is_fully_disallowed


def test_buying_more_than_was_sold_disallows_only_the_matched_shares() -> None:
    """The excess shares simply have their own basis. You cannot
    disallow more loss than existed."""
    report = find_wash_sales(
        [loss_disposal()], [buy("a", SALE_DATE, quantity="500")], POLICY
    )
    finding = report.findings[0]

    assert finding.matched == Shares("100")
    assert finding.disallowed == Money("-1000.00")
    assert finding.allowed == Money.zero()


def test_a_replacement_share_can_only_absorb_one_loss() -> None:
    """Without consumption bookkeeping a single small repurchase would
    appear to disallow several different losses in full."""
    report = find_wash_sales(
        [
            loss_disposal(lot_id="a", disposed=date(2026, 6, 15)),
            loss_disposal(lot_id="b", disposed=date(2026, 6, 20)),
        ],
        [buy("one", date(2026, 6, 16), quantity="100")],
        POLICY,
    )
    # 100 replacement shares against 200 sold: only the first disposal
    # is washed, and fully.
    assert len(report.findings) == 1
    assert report.findings[0].disposal.lot_id == "a"
    assert report.total_disallowed == Money("-1000.00")


def test_the_disallowed_amount_splits_exactly_across_replacements() -> None:
    """Conservation again: each part becomes a basis adjustment on a
    different lot, and a cent lost here is a cent of phantom gain later.

    $1,000 across three replacements does not divide evenly.
    """
    report = find_wash_sales(
        [loss_disposal()],
        [
            buy("a", date(2026, 6, 16), quantity="34"),
            buy("b", date(2026, 6, 17), quantity="33"),
            buy("c", date(2026, 6, 18), quantity="33"),
        ],
        POLICY,
    )
    finding = report.findings[0]
    parts = sum((r.disallowed for r in finding.replacements), Money.zero())
    assert parts == finding.disallowed


# ============================================================
# SUBSTANTIALLY IDENTICAL — curated, not computed
# ============================================================


def test_the_same_ticker_is_always_identical() -> None:
    empty = SubstituteMap(identical={})
    assert empty.are_identical("VOO", "VOO")


def test_the_firms_policy_decides_the_contested_cases() -> None:
    """Two S&P 500 ETFs from different issuers is genuinely contested —
    the IRS has never ruled and practitioners disagree. The engine
    applies whatever the firm decided rather than resolving a legal
    question it has no authority over."""
    assert POLICY.are_identical("VOO", "IVV")

    permissive = SubstituteMap.symmetric(groups=[])
    assert not permissive.are_identical("VOO", "IVV")


def test_an_unrelated_security_does_not_wash() -> None:
    report = find_wash_sales(
        [loss_disposal(ticker="VOO")], [buy("a", SALE_DATE, ticker="BND")], POLICY
    )
    assert not report


def test_an_asymmetric_substitute_map_is_refused() -> None:
    """'Substantially identical' is a symmetric relation. A one-way map
    would catch selling A and buying B but not the reverse — a bug that
    only ever surfaces as an inconsistency in a tax return."""
    with pytest.raises(ValueError, match="asymmetric"):
        SubstituteMap(identical={"VOO": frozenset({"IVV"})})


# ============================================================
# THE CONSEQUENCE — basis and holding period
# ============================================================


def test_basis_goes_up_by_the_disallowed_loss() -> None:
    """Section 1091(d). The deduction is deferred into this lot."""
    replacement = TaxLot(
        lot_id="new",
        ticker="VOO",
        acquired=date(2026, 6, 20),
        quantity=Shares("100"),
        cost_basis=Money("9000.00"),
    )
    adjusted = apply_basis_adjustment(
        replacement, Money("-1000.00"), original_acquired=date(2025, 1, 10)
    )
    assert adjusted.cost_basis == Money("10000.00")


def test_the_holding_period_tacks_back() -> None:
    """Section 1223(3): the replacement's holding period begins on the
    same day as the shares sold. Selling and rebuying does not reset the
    clock — which cuts both ways, since it can hand you long-term
    treatment you had not earned on the calendar."""
    replacement = TaxLot(
        lot_id="new",
        ticker="VOO",
        acquired=date(2026, 6, 20),
        quantity=Shares("100"),
        cost_basis=Money("9000.00"),
    )
    adjusted = apply_basis_adjustment(
        replacement, Money("-1000.00"), original_acquired=date(2025, 1, 10)
    )
    assert adjusted.acquired == date(2025, 1, 10)

    # And the tacked date is what makes the next sale long-term.
    result = dispose([adjusted], Shares("100"), Price("110.00"), on=date(2026, 7, 1))
    assert result.disposals[0].period is HoldingPeriod.LONG


def test_the_deferred_loss_comes_back_on_the_next_sale() -> None:
    """The round trip. A wash sale defers; it does not destroy — outside
    a retirement account."""
    replacement = TaxLot(
        lot_id="new",
        ticker="VOO",
        acquired=date(2026, 6, 20),
        quantity=Shares("100"),
        cost_basis=Money("9000.00"),
    )
    adjusted = apply_basis_adjustment(
        replacement, Money("-1000.00"), original_acquired=date(2025, 1, 10)
    )
    # Sold later at the same $90/share it was bought at: the original
    # $1,000 loss reappears, exactly.
    result = dispose([adjusted], Shares("100"), Price("90.00"), on=date(2026, 12, 1))
    assert result.realized_gain == Money("-1000.00")


def test_a_positive_disallowed_amount_is_refused() -> None:
    """Getting the sign backwards would halve the basis instead of
    raising it — an error that looks entirely plausible on a screen."""
    replacement = TaxLot(
        lot_id="new",
        ticker="VOO",
        acquired=date(2026, 6, 20),
        quantity=Shares("100"),
        cost_basis=Money("9000.00"),
    )
    with pytest.raises(ValueError, match="should be negative"):
        apply_basis_adjustment(
            replacement, Money("1000.00"), original_acquired=date(2025, 1, 10)
        )


# ============================================================
# PRE-TRADE SCREEN
# ============================================================


def test_the_screen_catches_a_purchase_before_the_trade_is_placed() -> None:
    """Runs BEFORE the order, so it can be blocked or re-timed rather
    than discovered in April."""
    blockers = would_trigger_wash_sale(
        "VOO", SALE_DATE, [buy("a", date(2026, 6, 1))], POLICY
    )
    assert len(blockers) == 1


def test_the_screen_covers_scheduled_future_purchases() -> None:
    """A DRIP date or recurring contribution already inside the forward
    window washes a sale that has not happened yet."""
    blockers = would_trigger_wash_sale(
        "VOO",
        SALE_DATE,
        [buy("drip", date(2026, 7, 1), reinvestment=True)],
        POLICY,
    )
    assert len(blockers) == 1


def test_a_clear_sale_returns_nothing() -> None:
    assert (
        would_trigger_wash_sale("VOO", SALE_DATE, [buy("a", date(2026, 1, 1))], POLICY)
        == ()
    )


# ============================================================
# HARVESTING
# ============================================================


def harvest_lot(lot_id: str, ticker: str, basis: str, quantity: str = "100") -> TaxLot:
    return TaxLot(
        lot_id=lot_id,
        ticker=ticker,
        acquired=date(2024, 1, 10),
        quantity=Shares(quantity),
        cost_basis=Money(basis),
    )


def test_harvesting_finds_losses_and_values_them() -> None:
    opportunities = find_harvest_opportunities(
        [harvest_lot("a", "VOO", "12000.00")],
        {"VOO": Price("100.00")},
        [],
        POLICY,
        HIGH,
        on=SALE_DATE,
        account_id="taxable-1",
        account_type=AccountType.TAXABLE,
    )
    assert len(opportunities) == 1
    assert opportunities[0].unrealized_loss == Money("-2000.00")
    # Long-term, so 23.8% including NIIT.
    assert opportunities[0].tax_benefit == Money("476.00")


def test_gains_are_not_harvest_opportunities() -> None:
    assert (
        find_harvest_opportunities(
            [harvest_lot("a", "VOO", "8000.00")],
            {"VOO": Price("100.00")},
            [],
            POLICY,
            HIGH,
            on=SALE_DATE,
            account_id="taxable-1",
            account_type=AccountType.TAXABLE,
        )
        == ()
    )


def test_harvesting_inside_an_ira_returns_nothing() -> None:
    """There is no loss to deduct because there was never a gain to tax.
    Returning a list of illusory opportunities would be worse than
    returning none."""
    assert (
        find_harvest_opportunities(
            [harvest_lot("a", "VOO", "12000.00")],
            {"VOO": Price("100.00")},
            [],
            POLICY,
            HIGH,
            on=SALE_DATE,
            account_id="ira-1",
            account_type=AccountType.ROTH_IRA,
        )
        == ()
    )


def test_a_blocked_opportunity_says_why_and_names_the_alternative() -> None:
    opportunities = find_harvest_opportunities(
        [harvest_lot("a", "VOO", "12000.00")],
        {"VOO": Price("100.00")},
        [buy("recent", date(2026, 6, 1))],
        POLICY,
        HIGH,
        on=SALE_DATE,
        account_id="taxable-1",
        account_type=AccountType.TAXABLE,
    )
    opportunity = opportunities[0]
    assert opportunity.is_blocked
    assert "deferred" in opportunity.block_reason
    assert opportunity.alternatives == ("VTI",)


def test_a_block_by_an_ira_purchase_warns_about_forfeiture() -> None:
    """The warning that matters most, because the consequence is
    permanent rather than merely inconvenient."""
    opportunities = find_harvest_opportunities(
        [harvest_lot("a", "VOO", "12000.00")],
        {"VOO": Price("100.00")},
        [
            buy(
                "ira",
                date(2026, 6, 1),
                account="ira-1",
                account_type=AccountType.ROTH_IRA,
            )
        ],
        POLICY,
        HIGH,
        on=SALE_DATE,
        account_id="taxable-1",
        account_type=AccountType.TAXABLE,
    )
    assert "PERMANENTLY FORFEITED" in opportunities[0].block_reason


def test_opportunities_are_ordered_by_benefit() -> None:
    opportunities = find_harvest_opportunities(
        [
            harvest_lot("small", "VOO", "10500.00"),
            harvest_lot("big", "IVV", "15000.00"),
        ],
        {"VOO": Price("100.00"), "IVV": Price("100.00")},
        [],
        POLICY,
        HIGH,
        on=SALE_DATE,
        account_id="taxable-1",
        account_type=AccountType.TAXABLE,
    )
    assert [o.lot.lot_id for o in opportunities] == ["big", "small"]


def test_a_minimum_loss_filters_out_noise() -> None:
    assert (
        find_harvest_opportunities(
            [harvest_lot("tiny", "VOO", "10010.00")],
            {"VOO": Price("100.00")},
            [],
            POLICY,
            HIGH,
            on=SALE_DATE,
            account_id="taxable-1",
            account_type=AccountType.TAXABLE,
            minimum_loss=Money("500.00"),
        )
        == ()
    )


# ============================================================
# HOUSEHOLD
# ============================================================


def test_an_unknown_account_is_refused_rather_than_assumed_taxable() -> None:
    """Guessing 'taxable' is the dangerous default: it is the one answer
    under which a wash sale looks merely deferred rather than
    forfeited."""
    household = Household.of("h1", [Account("taxable-1", AccountType.TAXABLE)])
    with pytest.raises(KeyError, match="cannot be determined"):
        household.type_of("mystery")


def test_retirement_accounts_are_identified() -> None:
    for kind in (
        AccountType.TRADITIONAL_IRA,
        AccountType.ROTH_IRA,
        AccountType.RETIREMENT_PLAN,
    ):
        assert kind.forfeits_wash_sale_basis
        assert kind.is_tax_advantaged
    assert not AccountType.TAXABLE.forfeits_wash_sale_basis


# ============================================================
# PROPERTIES
# ============================================================


@given(
    sold=st.integers(min_value=1, max_value=1000),
    replaced=st.integers(min_value=0, max_value=2000),
)
@settings(max_examples=400)
def test_disallowed_never_exceeds_the_loss(sold: int, replaced: int) -> None:
    """You cannot disallow more loss than existed. Buying back ten times
    what you sold still only washes what you sold."""
    disposal = loss_disposal(
        quantity=str(sold),
        proceeds=str(sold * 90),
        basis=str(sold * 100),
    )
    acquisitions = [buy("a", SALE_DATE, quantity=str(replaced))] if replaced else []
    report = find_wash_sales([disposal], acquisitions, POLICY)

    total_loss = abs(disposal.gain)
    assert abs(report.total_disallowed) <= total_loss


@given(
    sold=st.integers(min_value=1, max_value=1000),
    replaced=st.integers(min_value=1, max_value=1000),
)
@settings(max_examples=400)
def test_allowed_plus_disallowed_equals_the_loss(sold: int, replaced: int) -> None:
    """Conservation, applied to a loss. Nothing evaporates between the
    part you may deduct and the part you may not."""
    disposal = loss_disposal(
        quantity=str(sold),
        proceeds=str(sold * 90),
        basis=str(sold * 100),
    )
    report = find_wash_sales(
        [disposal], [buy("a", SALE_DATE, quantity=str(replaced))], POLICY
    )
    if not report:
        return

    finding = report.findings[0]
    assert finding.allowed + finding.disallowed == disposal.gain


@given(replacements=st.lists(st.integers(1, 100), min_size=1, max_size=6))
@settings(max_examples=300)
def test_the_split_across_replacements_always_sums(replacements: list[int]) -> None:
    """Each part becomes a basis adjustment on a different lot, so they
    must sum to the disallowed total exactly — the fourth caller of the
    same largest remainder method."""
    disposal = loss_disposal(quantity="1000", proceeds="90000.00", basis="100000.00")
    acquisitions = [
        buy(f"a{i}", date(2026, 6, 16), quantity=str(q))
        for i, q in enumerate(replacements)
    ]
    report = find_wash_sales([disposal], acquisitions, POLICY)
    if not report:
        return

    finding = report.findings[0]
    parts = sum((r.disallowed for r in finding.replacements), Money.zero())
    assert parts == finding.disallowed


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=200)
def test_detection_is_deterministic(seed: int) -> None:
    disposal = loss_disposal(quantity=str(1 + seed % 500))
    acquisitions = [
        buy("a", date(2026, 6, 16), quantity="40"),
        buy("b", date(2026, 6, 17), quantity="40"),
    ]
    first = find_wash_sales([disposal], acquisitions, POLICY)
    second = find_wash_sales([disposal], acquisitions, POLICY)
    assert first.total_disallowed == second.total_disallowed
    assert [r.acquisition_id for f in first.findings for r in f.replacements] == [
        r.acquisition_id for f in second.findings for r in f.replacements
    ]


@given(quantity=st.integers(min_value=1, max_value=500))
@settings(max_examples=200)
def test_basis_adjustment_preserves_total_economics(quantity: int) -> None:
    """Round trip: adjust a lot's basis by a disallowed loss, sell it at
    the price it was bought at, and exactly that loss comes back."""
    lot = TaxLot(
        lot_id="new",
        ticker="VOO",
        acquired=date(2026, 6, 20),
        quantity=Shares(quantity),
        cost_basis=Money.from_cents(quantity * 9000),
    )
    disallowed = Money.from_cents(-quantity * 1000)
    adjusted = apply_basis_adjustment(
        lot, disallowed, original_acquired=date(2025, 1, 10)
    )
    result = dispose([adjusted], Shares(quantity), Price("90.00"), on=date(2026, 12, 1))
    assert result.realized_gain == disallowed


def test_the_block_reason_names_the_worst_blocker_not_the_first() -> None:
    """A taxable purchase and an IRA purchase both sit in the window.
    The earlier one only defers; the IRA one forfeits. The reason must
    say forfeited, because that is what will happen."""
    lot_ = TaxLot("gld-1", "GLD", date(2024, 1, 1), Shares("100"), Money("20000.00"))
    opportunity = HarvestOpportunity(
        lot=lot_,
        account_id="taxable-1",
        account_type=AccountType.TAXABLE,
        market_value=Money("15000.00"),
        unrealized_loss=Money("-5000.00"),
        period=HoldingPeriod.LONG,
        tax_benefit=Money("1190.00"),
        blocked_by=(
            Acquisition(
                "a",
                "taxable-1",
                AccountType.TAXABLE,
                "GLD",
                date(2026, 6, 1),
                Shares("10"),
            ),
            Acquisition(
                "b",
                "roth-1",
                AccountType.ROTH_IRA,
                "GLD",
                date(2026, 6, 10),
                Shares("10"),
            ),
        ),
    )
    assert "PERMANENTLY FORFEITED" in opportunity.block_reason
    assert "roth-1" in opportunity.block_reason


# ============================================================
# A LOT IS NOT ITS OWN REPLACEMENT
# ============================================================


def test_the_screen_does_not_count_the_lot_being_sold() -> None:
    """Bought 20 November, sold 15 December. The purchase is inside the
    window, but it created the lot being sold — those shares are not
    replacements for themselves."""
    own = Acquisition(
        "t:GLD-5",
        "taxable-1",
        AccountType.TAXABLE,
        "GLD",
        date(2025, 11, 20),
        Shares("100"),
    )
    clear = would_trigger_wash_sale(
        "GLD",
        date(2025, 12, 15),
        [own],
        POLICY,
        account_id="taxable-1",
        selling=[(date(2025, 11, 20), Shares("100"))],
    )
    assert clear == ()


def test_excess_shares_bought_the_same_day_still_wash() -> None:
    """Bought 100 on 20 November, sell 60 of them on 15 December. The
    other 40 were acquired in the window and are still held: they ARE
    replacements, for 40 of the 60 sold."""
    own = Acquisition(
        "t:GLD-5",
        "taxable-1",
        AccountType.TAXABLE,
        "GLD",
        date(2025, 11, 20),
        Shares("100"),
    )
    blockers = would_trigger_wash_sale(
        "GLD",
        date(2025, 12, 15),
        [own],
        POLICY,
        account_id="taxable-1",
        selling=[(date(2025, 11, 20), Shares("60"))],
    )
    assert len(blockers) == 1
    assert blockers[0].quantity == Shares("40")


def test_a_same_day_purchase_in_another_account_is_not_netted() -> None:
    """The netting is account-aware. The Roth buying the same ticker on
    the same day as the taxable lot is a different purchase, and it is
    a replacement."""
    roth = Acquisition(
        "r:GLD-1",
        "roth-1",
        AccountType.ROTH_IRA,
        "GLD",
        date(2025, 11, 20),
        Shares("100"),
    )
    blockers = would_trigger_wash_sale(
        "GLD",
        date(2025, 12, 15),
        [roth],
        POLICY,
        account_id="taxable-1",
        selling=[(date(2025, 11, 20), Shares("100"))],
    )
    assert blockers == (roth,)


def test_the_detector_applies_rev_rul_56_602() -> None:
    """Sell an old lot and a recently bought lot on the same day, both at
    a loss. The recent purchase is inside the window, but its shares
    went out in the same sale — nothing was replaced, so no wash."""
    old = Disposal(
        "GLD-1",
        "GLD",
        date(2024, 1, 2),
        date(2026, 4, 15),
        Shares("25"),
        Money("3750.00"),
        Money("5000.00"),
        HoldingPeriod.LONG,
        True,
    )
    recent = Disposal(
        "GLD-9",
        "GLD",
        date(2026, 3, 17),
        date(2026, 4, 15),
        Shares("1"),
        Money("150.00"),
        Money("165.00"),
        HoldingPeriod.SHORT,
        True,
    )
    recent_purchase = Acquisition(
        "t:GLD-9",
        "taxable-1",
        AccountType.TAXABLE,
        "GLD",
        date(2026, 3, 17),
        Shares("1"),
    )
    report = find_wash_sales([old, recent], [recent_purchase], POLICY)
    assert not report.findings


def test_the_detector_still_catches_a_recent_lot_that_is_kept() -> None:
    """Same facts, but the recent lot is NOT sold. Now those shares are
    held after the loss sale, and they are replacements."""
    old = Disposal(
        "GLD-1",
        "GLD",
        date(2024, 1, 2),
        date(2026, 4, 15),
        Shares("25"),
        Money("3750.00"),
        Money("5000.00"),
        HoldingPeriod.LONG,
        True,
    )
    recent_purchase = Acquisition(
        "t:GLD-9",
        "taxable-1",
        AccountType.TAXABLE,
        "GLD",
        date(2026, 3, 17),
        Shares("1"),
    )
    report = find_wash_sales([old], [recent_purchase], POLICY)
    assert len(report.findings) == 1
    assert report.findings[0].matched == Shares("1")


# ============================================================
# THE BLACKOUT — the buy side of section 1091
# ============================================================


GOLD = SubstituteMap.symmetric(groups=[["GLD", "IAU"]])


def test_blackout_covers_the_loss_and_its_identical_substitutes() -> None:
    loss = Disposal(
        "GLD-1",
        "GLD",
        date(2024, 1, 2),
        date(2026, 2, 16),
        Shares("10"),
        Money("1500.00"),
        Money("2000.00"),
        HoldingPeriod.LONG,
        True,
    )
    assert blackout_tickers([loss], GOLD, on=date(2026, 3, 1)) == {"GLD", "IAU"}


def test_blackout_ends_after_thirty_days() -> None:
    loss = Disposal(
        "GLD-1",
        "GLD",
        date(2024, 1, 2),
        date(2026, 2, 16),
        Shares("10"),
        Money("1500.00"),
        Money("2000.00"),
        HoldingPeriod.LONG,
        True,
    )
    assert blackout_tickers([loss], GOLD, on=date(2026, 3, 18)) == {"GLD", "IAU"}
    assert blackout_tickers([loss], GOLD, on=date(2026, 3, 19)) == frozenset()


def test_a_gain_imposes_no_blackout() -> None:
    gain = Disposal(
        "GLD-1",
        "GLD",
        date(2024, 1, 2),
        date(2026, 2, 16),
        Shares("10"),
        Money("2500.00"),
        Money("2000.00"),
        HoldingPeriod.LONG,
        True,
    )
    assert blackout_tickers([gain], GOLD, on=date(2026, 3, 1)) == frozenset()
