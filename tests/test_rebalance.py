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
from meridian.taxlot import LotError, LotMethod, TaxRates, build_lots, dispose

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


# ============================================================
# LOT-AWARE SELLS
# ============================================================
# The same drill portfolio — 60/25/15 against 55/35/10 — but the
# equity sleeve's two holdings were bought at different times and
# prices, so WHICH equity to sell now has a tax answer:
#
#   VTI   300 sh bought 2024-01-02 at $100, now $140  -> +$12,000 long
#   AAPL  100 sh bought 2026-08-01 at $200, now $180  ->  -$2,000 short
#
# Market values are unchanged from DRIFTED, so the sleeve deltas are
# identical. Only the choice of lot differs.

HIGH = TaxRates(
    short_term=Weight("0.37"), long_term=Weight("0.20"), niit=Weight("0.038")
)

LOTTED: list[LedgerEvent] = [
    Deposit(1, date(2024, 1, 2), Money("90000.00")),
    Buy(2, date(2024, 1, 2), "VTI", Shares("300"), Price("100.00")),  # 30,000
    Buy(3, date(2024, 1, 2), "BND", Shares("500"), Price("50.00")),  # 25,000
    Buy(4, date(2024, 1, 2), "GLD", Shares("100"), Price("150.00")),  # 15,000
    Buy(5, date(2026, 8, 1), "AAPL", Shares("100"), Price("200.00")),  # 20,000
]


def propose_with_lots(
    events: list[LedgerEvent],
    policy: RebalancePolicy | None = None,
    rates: TaxRates | None = HIGH,
) -> Proposal:
    return generate_proposal(
        fold(events),
        PRICES,
        MODEL,
        CLASSIFICATION,
        on=D,
        policy=policy,
        lots=build_lots(events),
        rates=rates,
    )


def equity_sells(proposal: Proposal) -> list[str]:
    return sorted(t.ticker for t in proposal.sells if t.sleeve == "equity")


def test_min_tax_sells_the_loss_lot_not_the_winner() -> None:
    """The headline. Trimming equity by a few thousand dollars can
    realise a $12,000 long-term gain or harvest a short-term loss,
    depending on which holding is sold. Ranking lots by tax per dollar
    picks the loss without any list saying losses go first."""
    proposal = propose_with_lots(LOTTED)
    assert equity_sells(proposal) == ["AAPL"]

    aapl = next(t for t in proposal.sells if t.ticker == "AAPL")
    assert aapl.realized_gain is not None
    assert aapl.realized_gain.is_negative
    assert all(sel.period.value == "short" for sel in aapl.lots)


def test_fifo_sells_the_oldest_lot_instead() -> None:
    """Same portfolio, different policy, different trade. The policy is
    recorded on the proposal so the choice can be explained later."""
    proposal = propose_with_lots(
        LOTTED, RebalancePolicy(lot_method=LotMethod.FIFO), rates=None
    )
    assert equity_sells(proposal) == ["VTI"]
    vti = next(t for t in proposal.sells if t.ticker == "VTI")
    assert vti.realized_gain is not None
    assert not vti.realized_gain.is_negative


def test_hifo_needs_no_rates_and_still_prefers_the_high_basis_lot() -> None:
    proposal = propose_with_lots(
        LOTTED, RebalancePolicy(lot_method=LotMethod.HIFO), rates=None
    )
    assert equity_sells(proposal) == ["AAPL"]


def test_lot_quantities_sum_to_the_order() -> None:
    """A trade's lots ARE the trade. If they did not add up, the ledger
    would consume a different disposal than the one the gate judged."""
    proposal = propose_with_lots(LOTTED)
    for trade in proposal.sells:
        assert trade.lots
        total = sum((sel.quantity for sel in trade.lots), Shares.zero())
        assert total == trade.quantity
        proceeds = sum((sel.proceeds for sel in trade.lots), Money.zero())
        assert proceeds == trade.consideration


def test_the_identification_travels_into_the_ledger() -> None:
    """The order was sized against specific lots, so the Sell event
    names them, and rebuilding lots from the ledger consumes exactly
    those — not FIFO's choice."""
    proposal = propose_with_lots(LOTTED)
    events = proposal.to_events(100)

    sells = [e for e in events if e.__class__.__name__ == "Sell"]
    assert sells
    assert all(getattr(e, "lot_ids", ()) for e in sells)

    after = build_lots([*LOTTED, *events])
    # VTI was the winner; min-tax left it alone, so the 2024 lot is
    # intact. FIFO replay would have eaten into it instead.
    assert [lot.quantity for lot in after["VTI"]] == [Shares("300")]
    assert sum((lot.quantity.quantity for lot in after["AAPL"]), Decimal(0)) < 100


def test_lots_that_disagree_with_positions_are_refused() -> None:
    """Lots and positions are two views of one ledger. If they differ,
    one is wrong, and sizing a sale against the wrong one hands the
    custodian an order that bounces."""
    lots = build_lots(LOTTED)
    lots["AAPL"] = lots["AAPL"][:0]  # the lots have gone missing
    with pytest.raises(LotError, match="must agree"):
        generate_proposal(
            fold(LOTTED), PRICES, MODEL, CLASSIFICATION, on=D, lots=lots, rates=HIGH
        )


def test_min_tax_without_rates_is_refused() -> None:
    """Guessing a bracket would produce an authoritative-looking tax
    figure that is not."""
    with pytest.raises(LotError, match="requires the client's tax rates"):
        propose_with_lots(LOTTED, rates=None)


def test_specific_id_is_not_a_rebalancing_policy() -> None:
    with pytest.raises(ValueError, match="not a rebalancing policy"):
        RebalancePolicy(lot_method=LotMethod.SPECIFIC_ID)


def test_without_lots_sells_carry_no_identification() -> None:
    """The proportional path, for accounts where lot choice has no tax
    consequence. An unknown gain is None, not zero."""
    proposal = propose(DRIFTED)
    assert proposal.sells
    for trade in proposal.sells:
        assert trade.lots == ()
        assert trade.realized_gain is None


# ============================================================
# PROPERTIES OVER PORTFOLIOS WITH REAL LOT HISTORIES
# ============================================================


@st.composite
def lotted_portfolios(draw: st.DrawFn) -> list[LedgerEvent]:
    """Several purchases per ticker, at varied dates and prices, so lots
    genuinely differ in basis and holding period — the case where lot
    choice matters and ties are rare."""
    deposit = Money.from_cents(draw(st.integers(100_000_00, 5_000_000_00)))
    events: list[LedgerEvent] = [Deposit(1, date(2023, 1, 3), deposit)]

    tickers = ["VTI", "AAPL", "BND", "GLD"]
    n_lots = draw(st.integers(1, 6))
    cash = deposit
    seq = 2
    day = date(2023, 1, 3)

    for _ in range(n_lots):
        ticker = draw(st.sampled_from(tickers))
        # Basis anywhere from half to one-and-a-half times today's price.
        scale = Decimal(draw(st.integers(50, 150))) / 100
        price = Price((PRICES[ticker].amount * scale).quantize(Decimal("0.01")))
        day = day.fromordinal(day.toordinal() + draw(st.integers(1, 300)))
        if day >= D:
            break
        budget = cash.amount * Decimal(draw(st.integers(5, 40))) / 100
        quantity = int(budget / price.amount)
        if quantity < 1:
            continue
        held = Shares(quantity)
        events.append(Buy(seq, day, ticker, held, price))
        cash = cash - held.value_at(price)
        seq += 1

    return events


@given(events=lotted_portfolios())
@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
def test_lot_aware_proposals_are_applicable(events: list[LedgerEvent]) -> None:
    proposal = propose_with_lots(events)
    fold([*events, *proposal.to_events(1000)])


@given(events=lotted_portfolios())
@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
def test_lot_aware_proposals_conserve_value(events: list[LedgerEvent]) -> None:
    before = fold(events)
    proposal = propose_with_lots(events)
    after = fold([*events, *proposal.to_events(1000)])
    assert after.total_value(PRICES) == before.total_value(PRICES)


@given(events=lotted_portfolios())
@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
def test_lot_aware_generation_is_deterministic(events: list[LedgerEvent]) -> None:
    assert propose_with_lots(events).trades == propose_with_lots(events).trades


@given(events=lotted_portfolios())
@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
def test_the_ledger_replays_the_lots_the_order_named(
    events: list[LedgerEvent],
) -> None:
    """Two routes to the same disposal. The rebalancer picked lots and
    recorded them; replaying the Sell under specific identification
    must consume the same lots in the same quantities, and the lots
    left afterwards must still add up to the position."""
    lots_before = build_lots(events)
    proposal = propose_with_lots(events)

    for trade in proposal.sells:
        replayed = dispose(
            lots_before[trade.ticker],
            trade.quantity,
            trade.price,
            on=D,
            method=LotMethod.SPECIFIC_ID,
            chosen=[sel.lot_id for sel in trade.lots],
        )
        # Compared by lot, not by position: `dispose` reports in the
        # ledger's lot order, the trade in consumption order. Same set,
        # same quantities, same basis — that is the identification.
        assert {d.lot_id: d.quantity for d in replayed.disposals} == {
            sel.lot_id: sel.quantity for sel in trade.lots
        }
        assert replayed.cost_basis == sum(
            (sel.cost_basis for sel in trade.lots), Money.zero()
        )

    combined = [*events, *proposal.to_events(1000)]
    positions = fold(combined).positions
    lots_after = build_lots(combined)
    for ticker, held in positions.items():
        in_lots = sum(
            (lot.quantity.quantity for lot in lots_after.get(ticker, [])), Decimal(0)
        )
        assert in_lots == held.quantity


# ============================================================
# THE CASH TRIGGER
# ============================================================
# Drift is measured over invested value, so a deposit sitting as cash
# moves no sleeve and trips no band. A policy can say that idle cash is
# itself a reason to act.

ON_TARGET_WITH_CASH: list[LedgerEvent] = [
    Deposit(1, D, Money("110000.00")),
    Buy(2, D, "VTI", Shares("392.857"), Price("140.00")),  # ~55,000
    Buy(3, D, "BND", Shares("700"), Price("50.00")),  # 35,000
    Buy(4, D, "GLD", Shares("66.666"), Price("150.00")),  # ~10,000
    # 10,000 left as cash: 9% of the account, every sleeve on target.
]


def test_without_a_trigger_idle_cash_waits_for_a_breach() -> None:
    """The default. Nothing is breached, so nothing happens — and the
    cash sits there. True of the bands, false of the portfolio."""
    proposal = propose(ON_TARGET_WITH_CASH)
    assert proposal.is_empty
    assert proposal.trigger == "none"


def test_idle_cash_above_the_trigger_is_deployed() -> None:
    """With a 2% trigger, 9% cash is a reason to act. Buys only: there
    is no overweight to sell, just money to put to work."""
    proposal = propose(
        ON_TARGET_WITH_CASH, RebalancePolicy(cash_trigger=Weight("0.02"))
    )
    assert proposal.trigger == "cash"
    assert proposal.buys
    assert not proposal.sells
    assert proposal.cash_after < Money("500.00")


def test_cash_below_the_trigger_is_left_alone() -> None:
    proposal = propose(
        ON_TARGET_WITH_CASH, RebalancePolicy(cash_trigger=Weight("0.20"))
    )
    assert proposal.is_empty


def test_a_breach_is_recorded_as_the_trigger_even_with_cash() -> None:
    proposal = propose(DRIFTED, RebalancePolicy(cash_trigger=Weight("0.02")))
    assert proposal.trigger == "band"


def test_the_cash_buffer_does_not_count_as_idle() -> None:
    """Cash the policy holds back on purpose is not cash drag."""
    proposal = propose(
        ON_TARGET_WITH_CASH,
        RebalancePolicy(cash_trigger=Weight("0.02"), cash_buffer=Money("9000.00")),
    )
    assert proposal.is_empty


def test_cash_trigger_must_be_a_fraction() -> None:
    with pytest.raises(ValueError, match="cash_trigger"):
        RebalancePolicy(cash_trigger=Weight("1.5"))


@given(events=drifted_portfolios())
@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
def test_cash_triggered_proposals_are_applicable(events: list[LedgerEvent]) -> None:
    """The second trigger goes through the same machinery and must keep
    the same guarantees: applicable, and value-conserving."""
    policy = RebalancePolicy(cash_trigger=Weight("0.01"))
    before = fold(events)
    proposal = propose(events, policy)
    after = fold([*events, *proposal.to_events(1000)])
    assert after.total_value(PRICES) == before.total_value(PRICES)


# ============================================================
# THE BLACKOUT
# ============================================================


def test_a_blacked_out_holding_is_not_topped_up() -> None:
    """Equity holds VTI and AAPL. VTI was sold at a loss within 30 days,
    so the top-up goes entirely to AAPL."""
    with_cash: list[LedgerEvent] = [
        Deposit(1, D, Money("110000.00")),
        Buy(2, D, "VTI", Shares("300"), Price("140.00")),  # 42,000
        Buy(3, D, "AAPL", Shares("100"), Price("180.00")),  # 18,000
        Buy(4, D, "BND", Shares("500"), Price("50.00")),  # 25,000
        Buy(5, D, "GLD", Shares("100"), Price("150.00")),  # 15,000
    ]
    proposal = generate_proposal(
        fold(with_cash),
        PRICES,
        MODEL,
        CLASSIFICATION,
        on=D,
        policy=RebalancePolicy(cash_trigger=Weight("0.02")),
        blackout={"VTI"},
    )
    bought = {t.ticker for t in proposal.buys}
    assert "VTI" not in bought
    assert "AAPL" in bought


def test_a_sleeve_with_every_candidate_blacked_out_reports_the_gap() -> None:
    """Equity holds only VTI, the model names VTI, and VTI is blacked
    out. Buying it anyway would wash the loss; the gap is reported with
    the reason instead."""
    with_cash: list[LedgerEvent] = [
        Deposit(1, D, Money("110000.00")),
        Buy(2, D, "VTI", Shares("300"), Price("140.00")),
        Buy(3, D, "BND", Shares("500"), Price("50.00")),
        Buy(4, D, "GLD", Shares("100"), Price("150.00")),
    ]
    proposal = generate_proposal(
        fold(with_cash),
        PRICES,
        MODEL,
        CLASSIFICATION,
        on=D,
        policy=RebalancePolicy(cash_trigger=Weight("0.02")),
        blackout={"VTI"},
    )
    assert not any(t.ticker == "VTI" for t in proposal.buys)
    gap = [u for u in proposal.unplaced if u.sleeve == "equity"]
    assert gap and "IRC 1091" in gap[0].reason
