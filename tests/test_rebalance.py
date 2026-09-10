"""Tests for trade generation.

Two headline properties:

  1. A proposal CONSERVES VALUE. Moving money between buckets never
     creates or destroys any.
  2. A proposal is APPLICABLE. Every proposal the engine generates can
     be fed straight into the ledger without overdrawing cash or
     shorting a position — which means the ledger's guards and the
     rebalancer's arithmetic agree.

The second one is the connective tissue between Phase 02 and Phase 03.
If the rebalancer could produce an order the ledger refuses, then it
could produce an order the CUSTODIAN refuses, and the first anyone would
know is a rejected trade and a client statement that does not tie.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from meridian.drift import compute_drift, group_values
from meridian.ledger import Buy, Deposit, LedgerEvent, Portfolio, fold, market_values
from meridian.model import BandKind, Model, Sleeve
from meridian.money import Money, Price, Shares, Weight
from meridian.rebalance import (
    Proposal,
    RebalancePolicy,
    Side,
    TargetMode,
    generate_proposal,
)

D = date(2026, 9, 8)

PRICES = {
    "VTI": Price("140.00"),
    "AAPL": Price("180.00"),
    "BND": Price("50.00"),
    "GLD": Price("150.00"),
}

CLASSIFICATION = {
    "VTI": "equity",
    "AAPL": "equity",
    "BND": "bond",
    "GLD": "alt",
}

MODEL = Model(
    model_id="classic-60-40",
    version=1,
    sleeves={
        "equity": Sleeve(Weight("0.55"), Weight("0.05"), security="VTI"),
        "bond": Sleeve(Weight("0.35"), Weight("0.05"), security="BND"),
        "alt": Sleeve(Weight("0.10"), Weight("0.03"), security="GLD"),
    },
)

# The drill's portfolio: 60% equity, 25% bond, 15% alt on $100,000.
DRIFTED: list[LedgerEvent] = [
    Deposit(1, D, Money("100000.00")),
    Buy(2, D, "VTI", Shares("300"), Price("140.00")),  # 42,000
    Buy(3, D, "AAPL", Shares("100"), Price("180.00")),  # 18,000
    Buy(4, D, "BND", Shares("500"), Price("50.00")),  # 25,000
    Buy(5, D, "GLD", Shares("100"), Price("150.00")),  # 15,000
]


def propose(
    events: list[LedgerEvent], policy: RebalancePolicy | None = None
) -> Proposal:
    return generate_proposal(
        fold(events), PRICES, MODEL, CLASSIFICATION, on=D, policy=policy
    )


# ============================================================
# THE HEADLINE PROPERTIES
# ============================================================


def test_a_proposal_is_applicable_to_the_ledger() -> None:
    """The connective tissue. The rebalancer's arithmetic and the
    ledger's guards must agree, or the custodian finds out first."""
    proposal = propose(DRIFTED)
    combined = [*DRIFTED, *proposal.to_events(starting_seq=100)]
    after = fold(combined)  # raises if it overdraws or shorts

    assert after.cash >= Money.zero()


def test_a_proposal_conserves_value() -> None:
    """Selling $5,000 of a holding produces $5,000 of cash. A rebalance
    moves value between buckets; it never creates or destroys any."""
    before = fold(DRIFTED)
    proposal = propose(DRIFTED)
    after = fold([*DRIFTED, *proposal.to_events(100)])

    assert after.total_value(PRICES) == before.total_value(PRICES)


def test_rebalancing_reduces_drift() -> None:
    """The whole point. Every breached sleeve must end up closer to
    target than it started."""
    before = fold(DRIFTED)
    proposal = propose(DRIFTED)
    after = fold([*DRIFTED, *proposal.to_events(100)])

    def drift_of(p: Portfolio) -> dict[str, Decimal]:
        values = group_values(market_values(p, PRICES), CLASSIFICATION)
        report = compute_drift(values, MODEL)
        return {r.sleeve: abs(r.drift.value) for r in report.sleeves}

    d_before, d_after = drift_of(before), drift_of(after)
    for sleeve in d_before:
        assert d_after[sleeve] <= d_before[sleeve]


# ============================================================
# THE WORKED CASE
# ============================================================


def test_the_drill_produces_sensible_trades() -> None:
    """60/25/15 against a 55/35/10 model with 5/5/3 point bands.

    bond  -10pts, band 5 -> BREACH, must be bought
    alt    +5pts, band 3 -> BREACH, must be sold
    equity +5pts, band 5 -> inside its band, but STILL TRIMMED

    That last line is the part worth understanding. An earlier version
    of the engine left equity alone, reasoning that a sleeve inside its
    band should not be touched — and then could not fund bond, reporting
    $3,000 as unplaced.

    The arithmetic forbids it. Move bond to 30% and alt to 13% while
    equity holds at 60% and the weights sum to 103%. No such portfolio
    exists. A band decides WHETHER to rebalance; once one trips, the
    whole portfolio participates, because the money has to come from
    somewhere.
    """
    proposal = propose(DRIFTED)

    traded_sleeves = {t.sleeve for t in proposal.trades}
    assert traded_sleeves == {"equity", "bond", "alt"}

    assert all(t.side is Side.BUY for t in proposal.trades if t.sleeve == "bond")
    assert all(t.side is Side.SELL for t in proposal.trades if t.sleeve == "alt")
    assert all(t.side is Side.SELL for t in proposal.trades if t.sleeve == "equity")

    # And the breach actually closes, rather than being reported as
    # unfundable.
    assert not proposal.unplaced


def test_a_breach_is_funded_rather_than_declared_impossible() -> None:
    """Regression guard for the bug above.

    Bond starts ten points light. After the rebalance it must be back
    INSIDE its band — not merely closer, and not left short with an
    apologetic unplaced row.
    """
    proposal = propose(DRIFTED)
    after = fold([*DRIFTED, *proposal.to_events(100)])
    report = compute_drift(
        group_values(market_values(after, PRICES), CLASSIFICATION), MODEL
    )

    bond = report.by_sleeve("bond")
    assert bond is not None
    assert not bond.breached


def test_band_edge_trades_less_than_to_target() -> None:
    """The turnover argument, demonstrated rather than asserted.

    Trading only to the near edge of the band moves less money, which is
    less spread, less commission, and fewer realised gains.
    """
    edge = propose(DRIFTED, RebalancePolicy(target_mode=TargetMode.TO_BAND_EDGE))
    target = propose(DRIFTED, RebalancePolicy(target_mode=TargetMode.TO_TARGET))

    assert edge.turnover < target.turnover


def test_an_on_target_portfolio_generates_nothing() -> None:
    """'Nothing to do' is a valid and common answer. A rebalancer that
    trades on every review costs its clients money."""
    on_target: list[LedgerEvent] = [
        Deposit(1, D, Money("100000.00")),
        Buy(2, D, "VTI", Shares("392.857"), Price("140.00")),  # ~55,000
        Buy(3, D, "BND", Shares("700"), Price("50.00")),  # 35,000
        Buy(4, D, "GLD", Shares("66.666"), Price("150.00")),  # ~10,000
    ]
    proposal = propose(on_target)
    assert proposal.is_empty
    assert not proposal.drift_before.needs_rebalancing


# ============================================================
# CASH FIRST
# ============================================================


def test_idle_cash_is_deployed_before_anything_is_sold() -> None:
    """The cheapest rebalance places no sell orders at all.

    This is not a special case in the code — targets are measured
    against total investable value, so idle cash shows up as a shortfall
    across the sleeves and gets spent first.
    """
    with_cash: list[LedgerEvent] = [
        Deposit(1, D, Money("100000.00")),
        Buy(2, D, "VTI", Shares("300"), Price("140.00")),  # 42,000
        Buy(3, D, "BND", Shares("500"), Price("50.00")),  # 25,000
        # 33,000 left as cash
    ]
    proposal = propose(with_cash)

    assert proposal.buys
    assert not proposal.sells
    assert proposal.cash_after < proposal.cash_before


def test_a_cash_buffer_is_never_spent() -> None:
    """Cash held back for fees and settlement stays held back."""
    with_cash: list[LedgerEvent] = [
        Deposit(1, D, Money("100000.00")),
        Buy(2, D, "VTI", Shares("300"), Price("140.00")),
        Buy(3, D, "BND", Shares("500"), Price("50.00")),
    ]
    buffer = Money("5000.00")
    proposal = propose(with_cash, RebalancePolicy(cash_buffer=buffer))
    assert proposal.cash_after >= buffer


# ============================================================
# ROUNDING AND MINIMUMS
# ============================================================


def test_orders_round_down_so_they_can_always_be_funded() -> None:
    """A buy rounded up orders more than the cash can fund. The leftover
    stays in cash instead."""
    proposal = propose(DRIFTED, RebalancePolicy(share_increment=Decimal("1")))
    for trade in proposal.trades:
        assert trade.quantity.quantity == trade.quantity.quantity.to_integral_value()


def test_trades_below_the_minimum_are_skipped() -> None:
    """A $3 order costs more in spread and attention than the tracking
    error it removes."""
    barely: list[LedgerEvent] = [
        Deposit(1, D, Money("100000.00")),
        Buy(2, D, "VTI", Shares("400"), Price("140.00")),  # 56,000
        Buy(3, D, "BND", Shares("690"), Price("50.00")),  # 34,500
        Buy(4, D, "GLD", Shares("63"), Price("150.00")),  # 9,450
    ]
    generous = propose(barely, RebalancePolicy(min_trade=Money("1000000.00")))
    assert generous.is_empty or all(
        t.consideration >= Money("1000000.00") for t in generous.trades
    )


def test_aiming_at_the_exact_band_edge_leaves_the_breach_open() -> None:
    """Why band_entry defaults to 10% rather than 0.

    With band_entry=0 the rebalance targets precisely 30.0%. Share
    quantities round DOWN, so it lands a few cents short — still in
    breach, and the shortfall is too small to trade next time. The
    sleeve would be flagged forever and never fixable.

    This test documents the trap by reproducing it deliberately.
    """
    on_the_edge = propose(DRIFTED, RebalancePolicy(band_entry=Decimal("0")))
    after = fold([*DRIFTED, *on_the_edge.to_events(100)])
    report = compute_drift(
        group_values(market_values(after, PRICES), CLASSIFICATION), MODEL
    )
    bond = report.by_sleeve("bond")
    assert bond is not None
    assert bond.breached  # landed a hair outside

    # The default lands inside instead.
    inside = propose(DRIFTED)
    after_inside = fold([*DRIFTED, *inside.to_events(100)])
    report_inside = compute_drift(
        group_values(market_values(after_inside, PRICES), CLASSIFICATION), MODEL
    )
    bond_inside = report_inside.by_sleeve("bond")
    assert bond_inside is not None
    assert not bond_inside.breached


def test_rounding_residue_is_not_reported_as_a_gap() -> None:
    """A report that cries wolf on eleven cents is a report nobody reads.

    Sells round down, buys round down, and the residue stays in cash.
    That is correct behaviour, not a shortfall — so it must not appear
    as an unplaced row alongside the ones that matter.
    """
    proposal = propose(DRIFTED)
    assert not proposal.unplaced
    for row in proposal.unplaced:
        assert row.shortfall >= proposal.policy.min_trade


def test_a_real_gap_is_still_reported() -> None:
    """The filter must not swallow something an advisor could act on."""
    anonymous = Model(
        "anon",
        1,
        {
            "equity": Sleeve(Weight("0.50"), Weight("0.02"), security="VTI"),
            "bond": Sleeve(Weight("0.50"), Weight("0.02")),  # no security named
        },
    )
    events: list[LedgerEvent] = [
        Deposit(1, D, Money("100000.00")),
        Buy(2, D, "VTI", Shares("500"), Price("140.00")),
    ]
    proposal = generate_proposal(fold(events), PRICES, anonymous, CLASSIFICATION, on=D)

    assert proposal.unplaced
    assert all(u.shortfall >= Money("100.00") for u in proposal.unplaced)


def test_never_sells_more_than_is_held() -> None:
    """Even by a rounding step. A short position is a data error, not a
    strategy."""
    proposal = propose(DRIFTED)
    held = fold(DRIFTED).positions
    for trade in proposal.sells:
        assert trade.quantity.quantity <= held[trade.ticker].quantity


# ============================================================
# HONEST FAILURE
# ============================================================


def test_an_empty_sleeve_with_no_named_security_is_reported_not_guessed() -> None:
    """Guessing would be the engine making an investment decision it has
    no authority to make."""
    anonymous = Model(
        "anon",
        1,
        {
            "equity": Sleeve(Weight("0.50"), Weight("0.02"), security="VTI"),
            "bond": Sleeve(Weight("0.50"), Weight("0.02")),  # no security named
        },
    )
    events: list[LedgerEvent] = [
        Deposit(1, D, Money("100000.00")),
        Buy(2, D, "VTI", Shares("500"), Price("140.00")),  # 70,000, all equity
    ]
    proposal = generate_proposal(fold(events), PRICES, anonymous, CLASSIFICATION, on=D)

    assert any("names no security" in u.reason for u in proposal.unplaced)
    assert not any(t.sleeve == "bond" for t in proposal.trades)


def test_an_untargeted_holding_is_sold_off() -> None:
    """Something the model never mentioned has a target of zero, so the
    whole position is the delta."""
    narrow = Model(
        "narrow",
        1,
        {
            "equity": Sleeve(Weight("0.60"), Weight("0.05"), security="VTI"),
            "bond": Sleeve(Weight("0.40"), Weight("0.05"), security="BND"),
        },
    )
    events: list[LedgerEvent] = [
        Deposit(1, D, Money("100000.00")),
        Buy(2, D, "VTI", Shares("400"), Price("140.00")),  # 56,000
        Buy(3, D, "BND", Shares("600"), Price("50.00")),  # 30,000
        Buy(4, D, "GLD", Shares("60"), Price("150.00")),  # 9,000 — untargeted
    ]
    proposal = generate_proposal(fold(events), PRICES, narrow, CLASSIFICATION, on=D)

    gld_sells = [t for t in proposal.sells if t.ticker == "GLD"]
    assert gld_sells
    assert sum((t.consideration for t in gld_sells), Money.zero()) > Money("8000.00")


def test_proposals_are_immutable() -> None:
    """The fiduciary artifact. It has to say the same thing in three
    years as it does today."""
    proposal = propose(DRIFTED)
    with pytest.raises(AttributeError):
        proposal.trades = ()  # type: ignore[misc]


def test_a_proposal_records_the_policy_it_was_generated_under() -> None:
    """Reconstructing a past recommendation needs the knobs, not just
    the inputs."""
    policy = RebalancePolicy(target_mode=TargetMode.TO_TARGET, min_trade=Money("50.00"))
    proposal = propose(DRIFTED, policy)
    assert proposal.policy == policy
    assert proposal.model_version == 1


# ============================================================
# PROPERTIES OVER GENERATED PORTFOLIOS
# ============================================================


@st.composite
def drifted_portfolios(draw: st.DrawFn) -> list[LedgerEvent]:
    """A funded portfolio holding an arbitrary mix of the four tickers.

    The `invested` draw is doing real work. An earlier version capped
    each holding at a quarter of remaining cash, which left every
    generated portfolio cash-heavy — so almost every rebalance was
    buy-only and the SELL path (the one that can overshoot into a short
    position) ran in about 7% of cases.

    Drawing a deployment fraction from 50% to 100% instead produces
    genuinely invested portfolios, where closing an overweight requires
    selling. Measured after the change: roughly two thirds of generated
    cases now exercise sells.

    The lesson generalises: a property test that never reaches the
    dangerous branch is a property test that proves nothing about it.
    """
    deposit = Money.from_cents(draw(st.integers(50_000_00, 5_000_000_00)))
    events: list[LedgerEvent] = [Deposit(1, D, deposit)]

    tickers = ["VTI", "AAPL", "BND", "GLD"]

    # How much of the deposit gets invested, and how it splits.
    invested = draw(st.integers(50, 100))
    shares_of_split = [draw(st.integers(0, 100)) for _ in tickers]
    if sum(shares_of_split) == 0:
        shares_of_split[0] = 1

    budget = deposit.amount * Decimal(invested) / 100
    split_total = Decimal(sum(shares_of_split))

    seq = 2
    cash = deposit
    for ticker, portion in zip(tickers, shares_of_split, strict=True):
        if portion == 0:
            continue
        price = PRICES[ticker]
        allowance = budget * Decimal(portion) / split_total
        quantity = int(allowance / price.amount)
        if quantity < 1:
            continue
        held = Shares(quantity)
        cost = held.value_at(price)
        if cost > cash:
            continue
        events.append(Buy(seq, D, ticker, held, price))
        cash = cash - cost
        seq += 1

    return events


@given(events=drifted_portfolios())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_every_proposal_is_applicable(events: list[LedgerEvent]) -> None:
    """For ANY portfolio, the proposal can be applied without the ledger
    refusing it. No overdraft, no short, ever."""
    proposal = propose(events)
    fold([*events, *proposal.to_events(1000)])  # raises on any violation


@given(events=drifted_portfolios())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_every_proposal_conserves_value(events: list[LedgerEvent]) -> None:
    before = fold(events)
    proposal = propose(events)
    after = fold([*events, *proposal.to_events(1000)])
    assert after.total_value(PRICES) == before.total_value(PRICES)


@given(events=drifted_portfolios())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_generation_is_deterministic(events: list[LedgerEvent]) -> None:
    """The same portfolio always produces the same trade list."""
    assert propose(events).trades == propose(events).trades


@given(events=drifted_portfolios())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_cash_never_goes_negative(events: list[LedgerEvent]) -> None:
    proposal = propose(events)
    assert not proposal.cash_after.is_negative


@given(events=drifted_portfolios())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_relative_bands_also_produce_applicable_proposals(
    events: list[LedgerEvent],
) -> None:
    """The other band kind, through the same machinery."""
    relative = Model(
        "rel",
        1,
        {
            "equity": Sleeve(
                Weight("0.55"), Weight("0.10"), BandKind.RELATIVE, security="VTI"
            ),
            "bond": Sleeve(
                Weight("0.35"), Weight("0.10"), BandKind.RELATIVE, security="BND"
            ),
            "alt": Sleeve(
                Weight("0.10"), Weight("0.10"), BandKind.RELATIVE, security="GLD"
            ),
        },
    )
    proposal = generate_proposal(fold(events), PRICES, relative, CLASSIFICATION, on=D)
    fold([*events, *proposal.to_events(1000)])
