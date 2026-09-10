"""Tests for performance measurement.

The headline property is the ATTRIBUTION IDENTITY: allocation plus
selection plus interaction must equal the excess return exactly. It is a
free correctness check that no amount of plausible-looking arithmetic
can fake.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from meridian.money import Money, Weight
from meridian.performance import (
    ExternalFlow,
    FlowKind,
    PerformanceError,
    SleeveReturn,
    Valuation,
    annualize,
    attribute,
    compute_performance,
    max_drawdown,
    modified_dietz,
    money_weighted_return,
    risk_statistics,
    sleeve_returns_from,
    time_weighted_return,
)

EPSILON = Decimal("1e-20")
"""Returns are ratios of money, and division is not closed over
decimals. Same boundary as drift.py — money assertions use exact
equality, return assertions use a stated epsilon."""


def v(day: str, value: str) -> Valuation:
    return Valuation(date.fromisoformat(day), Money(value))


def flow(day: str, amount: str, kind: FlowKind = FlowKind.EXTERNAL) -> ExternalFlow:
    return ExternalFlow(date.fromisoformat(day), Money(amount), kind)


# ============================================================
# TIME-WEIGHTED RETURN
# ============================================================


def test_a_simple_gain() -> None:
    assert time_weighted_return(
        [v("2026-01-01", "100.00"), v("2026-12-31", "110.00")], []
    ) == Decimal("0.1")


def test_a_deposit_is_not_return() -> None:
    """The whole point of time-weighting. The client added $50; the
    manager did not earn it."""
    result = time_weighted_return(
        [v("2026-01-01", "100.00"), v("2026-06-30", "150.00")],
        [flow("2026-06-30", "50.00")],
    )
    assert result == Decimal("0")


def test_returns_link_geometrically_not_additively() -> None:
    """A 50% gain then a 50% loss is a 25% LOSS, not break-even.

    Adding period returns is the most common performance bug, and it
    always flatters the manager in volatile periods.
    """
    result = time_weighted_return(
        [
            v("2026-01-01", "100.00"),
            v("2026-06-30", "150.00"),
            v("2026-12-31", "75.00"),
        ],
        [],
    )
    assert result == Decimal("-0.25")


def test_the_manager_is_not_blamed_for_bad_client_timing() -> None:
    """$1,000 in January, $50,000 in November right before a crash.

    The dollar-weighted result is dreadful. The manager's result is not,
    and TWR is what says so.
    """
    valuations = [
        v("2026-01-01", "1000.00"),
        v("2026-11-01", "51100.00"),  # 1,100 grown + 50,000 added
        v("2026-12-31", "45990.00"),  # -10%
    ]
    flows = [flow("2026-11-01", "50000.00")]

    twr = time_weighted_return(valuations, flows)
    # +10% then -10% = -1%
    assert abs(twr - Decimal("-0.01")) < EPSILON


def test_a_single_valuation_is_refused() -> None:
    with pytest.raises(PerformanceError, match="at least a start and an end"):
        time_weighted_return([v("2026-01-01", "100.00")], [])


def test_funding_an_empty_account_is_not_an_error() -> None:
    """A period that opens at zero and is funded has no return, rather
    than an infinite one. New accounts are the common case."""
    result = time_weighted_return(
        [v("2026-01-01", "0.00"), v("2026-01-02", "1000.00")],
        [flow("2026-01-02", "1000.00")],
    )
    assert result == Decimal("0")


def test_growth_from_literally_nothing_is_refused() -> None:
    with pytest.raises(PerformanceError, match="undefined"):
        time_weighted_return([v("2026-01-01", "0.00"), v("2026-01-02", "1000.00")], [])


# ============================================================
# MODIFIED DIETZ
# ============================================================


def test_dietz_matches_the_simple_case() -> None:
    result = modified_dietz(
        Money("100.00"), Money("110.00"), [], date(2026, 1, 1), date(2026, 1, 31)
    )
    assert result == Decimal("0.1")


def test_dietz_weights_a_flow_by_time_invested() -> None:
    """$10,000 arriving on day 2 should carry nearly full weight; the
    same $10,000 arriving on day 29 almost none. That weighting is the
    entire idea."""
    early = modified_dietz(
        Money("100000.00"),
        Money("115000.00"),
        [flow("2026-01-02", "10000.00")],
        date(2026, 1, 1),
        date(2026, 1, 31),
    )
    late = modified_dietz(
        Money("100000.00"),
        Money("115000.00"),
        [flow("2026-01-29", "10000.00")],
        date(2026, 1, 1),
        date(2026, 1, 31),
    )
    # Same profit, but the late money was barely invested — so the
    # return on capital actually at work is higher.
    assert late > early


def test_dietz_rejects_a_flow_outside_the_period() -> None:
    with pytest.raises(PerformanceError, match="outside the period"):
        modified_dietz(
            Money("100.00"),
            Money("110.00"),
            [flow("2026-02-15", "10.00")],
            date(2026, 1, 1),
            date(2026, 1, 31),
        )


def test_dietz_rejects_a_backwards_period() -> None:
    with pytest.raises(PerformanceError, match="must be positive"):
        modified_dietz(
            Money("100.00"), Money("110.00"), [], date(2026, 2, 1), date(2026, 1, 1)
        )


# ============================================================
# MONEY-WEIGHTED RETURN
# ============================================================


def test_a_simple_irr() -> None:
    """$100 in, $110 out a year later: 10%."""
    rate = money_weighted_return(
        [(date(2026, 1, 1), Money("-100.00")), (date(2027, 1, 1), Money("110.00"))]
    )
    assert abs(rate - 0.10) < 1e-6


def test_the_client_is_blamed_for_their_own_timing() -> None:
    """The same portfolio as the TWR test above. TWR said -1%; the
    money-weighted answer is far worse, because most of the money was
    only present for the fall.

    Both numbers are correct. They answer different questions.
    """
    rate = money_weighted_return(
        [
            (date(2026, 1, 1), Money("-1000.00")),
            (date(2026, 11, 1), Money("-50000.00")),
            (date(2026, 12, 31), Money("45990.00")),
        ]
    )
    assert rate < -0.30


def test_an_irr_needs_flows_in_both_directions() -> None:
    with pytest.raises(PerformanceError, match="both positive and negative"):
        money_weighted_return(
            [(date(2026, 1, 1), Money("-100.00")), (date(2027, 1, 1), Money("-50.00"))]
        )


def test_the_bisection_fallback_handles_awkward_flows() -> None:
    """Newton-Raphson can wander off on sign-alternating flows. The
    bracketed fallback cannot diverge."""
    rate = money_weighted_return(
        [
            (date(2026, 1, 1), Money("-1000.00")),
            (date(2026, 3, 1), Money("2000.00")),
            (date(2026, 6, 1), Money("-1500.00")),
            (date(2026, 12, 1), Money("700.00")),
        ]
    )
    assert isinstance(rate, float)


# ============================================================
# ANNUALISATION — the GIPS refusal
# ============================================================


def test_annualising_a_quarter_is_refused() -> None:
    """A 4% quarter is a 4% quarter. Extrapolating it to 17% invents
    performance for nine months that have not happened, and GIPS
    prohibits it outright."""
    with pytest.raises(PerformanceError, match="less than one year"):
        annualize(Decimal("0.04"), days=90)


def test_a_full_year_annualises_to_itself() -> None:
    assert abs(annualize(Decimal("0.10"), days=365) - Decimal("0.10")) < Decimal("1e-9")


def test_two_years_annualises_by_compounding() -> None:
    """21% over two years is 10% a year, not 10.5%."""
    result = annualize(Decimal("0.21"), days=730)
    assert abs(result - Decimal("0.10")) < Decimal("1e-6")


def test_the_refusal_is_an_error_not_a_warning() -> None:
    """A warning in a log is not a control — it is a thing nobody reads
    until after the advertisement went out."""
    with pytest.raises(PerformanceError):
        annualize(Decimal("0.04"), days=364)


# ============================================================
# GROSS AND NET TRAVEL TOGETHER
# ============================================================


def test_gross_and_net_come_from_the_same_call() -> None:
    """Marketing Rule 206(4)-1 requires net wherever gross is shown.
    One object rather than two calls turns that from a rule someone must
    remember into a shape the code enforces."""
    result = compute_performance(
        [v("2026-01-01", "100000.00"), v("2026-12-31", "108000.00")],
        [flow("2026-06-30", "-1000.00", FlowKind.FEE)],
    )
    assert result.twr_gross > result.twr_net
    assert result.fees_paid == Money("1000.00")
    assert result.fee_drag > 0


def test_a_portfolio_with_no_fees_has_no_gap() -> None:
    result = compute_performance(
        [v("2026-01-01", "100000.00"), v("2026-12-31", "110000.00")], []
    )
    assert result.twr_gross == result.twr_net
    assert result.fee_drag == Decimal(0)


def test_a_fee_must_be_negative() -> None:
    with pytest.raises(PerformanceError, match="must be negative"):
        ExternalFlow(date(2026, 6, 30), Money("1000.00"), FlowKind.FEE)


def test_a_sub_year_result_will_not_annualize() -> None:
    result = compute_performance(
        [v("2026-01-01", "100000.00"), v("2026-03-31", "104000.00")], []
    )
    assert not result.can_annualize
    with pytest.raises(PerformanceError, match="less than one year"):
        result.annualized()


def test_a_full_year_result_annualizes_both_figures() -> None:
    result = compute_performance(
        [v("2026-01-01", "100000.00"), v("2027-01-01", "110000.00")],
        [flow("2026-06-30", "-1000.00", FlowKind.FEE)],
    )
    gross, net = result.annualized()
    assert gross > net


# ============================================================
# ATTRIBUTION — the identity
# ============================================================


def sleeve(name: str, pw: str, pr: str, bw: str, br: str) -> SleeveReturn:
    return SleeveReturn(name, Weight(pw), Decimal(pr), Weight(bw), Decimal(br))


def test_the_effects_sum_to_the_excess_return() -> None:
    """The free correctness check. No amount of plausible-looking
    arithmetic can fake this identity."""
    result = attribute(
        [
            sleeve("equity", "0.60", "0.12", "0.55", "0.10"),
            sleeve("bond", "0.30", "0.03", "0.35", "0.04"),
            sleeve("alt", "0.10", "0.08", "0.10", "0.06"),
        ]
    )
    total = result.total_allocation + result.total_selection + result.total_interaction
    assert abs(total - result.excess_return) < EPSILON


def test_overweighting_a_winner_scores_positive_allocation() -> None:
    """The Fachler refinement. Equity beat the overall benchmark and we
    were overweight it, so the allocation decision worked."""
    result = attribute(
        [
            sleeve("equity", "0.70", "0.10", "0.50", "0.10"),
            sleeve("bond", "0.30", "0.02", "0.50", "0.02"),
        ]
    )
    equity = next(r for r in result.rows if r.sleeve == "equity")
    assert equity.allocation > 0


def test_overweighting_a_laggard_scores_negative_allocation() -> None:
    """This is exactly what Brinson-Hood-Beebower gets wrong.

    BHB uses the raw sector return, so overweighting ANY sector with a
    positive return scores positively — even one that trailed the
    benchmark badly. Subtracting the total benchmark return fixes that.
    """
    result = attribute(
        [
            sleeve("bond", "0.70", "0.02", "0.50", "0.02"),  # +2%, but a laggard
            sleeve("equity", "0.30", "0.10", "0.50", "0.10"),
        ]
    )
    bond = next(r for r in result.rows if r.sleeve == "bond")
    assert bond.allocation < 0


def test_good_stock_picking_shows_as_selection() -> None:
    result = attribute(
        [
            sleeve("equity", "0.50", "0.15", "0.50", "0.10"),  # beat the sector
            sleeve("bond", "0.50", "0.04", "0.50", "0.04"),
        ]
    )
    equity = next(r for r in result.rows if r.sleeve == "equity")
    assert equity.selection > 0
    assert equity.allocation == 0  # no active weight


def test_attribution_needs_a_sleeve() -> None:
    with pytest.raises(PerformanceError, match="at least one sleeve"):
        attribute([])


def test_sleeves_are_assembled_over_the_union_of_names() -> None:
    """A sleeve the benchmark holds and the portfolio does not is an
    allocation decision worth attributing — and would vanish from a
    report that iterated over the portfolio alone."""
    sleeves = sleeve_returns_from(
        portfolio_weights={"equity": Weight("1.0")},
        portfolio_returns={"equity": Decimal("0.10")},
        benchmark_weights={"equity": Weight("0.6"), "bond": Weight("0.4")},
        benchmark_returns={"equity": Decimal("0.08"), "bond": Decimal("0.03")},
    )
    assert [s.sleeve for s in sleeves] == ["bond", "equity"]
    assert sleeves[0].portfolio_weight == Weight("0")


# ============================================================
# RISK STATISTICS
# ============================================================


def test_volatility_is_annualised() -> None:
    returns = [0.01, -0.01] * 50
    stats = risk_statistics(returns)
    assert stats.volatility > 0
    assert stats.observations == 100


def test_benchmark_relative_stats_are_none_without_a_benchmark() -> None:
    """A missing measurement is not a measurement of zero."""
    stats = risk_statistics([0.01, -0.01, 0.02, -0.005])
    assert stats.tracking_error is None
    assert stats.beta is None
    assert stats.information_ratio is None


def test_beta_of_one_against_itself() -> None:
    returns = [0.01, -0.02, 0.015, 0.003, -0.008]
    stats = risk_statistics(returns, benchmark_returns=returns)
    assert stats.beta is not None
    assert abs(stats.beta - 1.0) < 1e-9
    assert stats.tracking_error == 0.0


def test_a_misaligned_benchmark_is_refused() -> None:
    with pytest.raises(PerformanceError, match="period for period"):
        risk_statistics([0.01, 0.02], benchmark_returns=[0.01])


def test_risk_stats_need_two_observations() -> None:
    with pytest.raises(PerformanceError, match="at least two"):
        risk_statistics([0.01])


def test_max_drawdown_is_positive() -> None:
    """'A 32% drawdown' is how anyone says it. A sign convention nobody
    uses in speech is one someone will misread."""
    assert abs(max_drawdown([100.0, 120.0, 80.0, 90.0]) - (40.0 / 120.0)) < 1e-9


def test_a_monotonic_rise_has_no_drawdown() -> None:
    assert max_drawdown([100.0, 110.0, 120.0]) == 0.0


# ============================================================
# PROPERTIES
# ============================================================

returns_strategy = st.lists(
    st.decimals(
        min_value=Decimal("-0.5"),
        max_value=Decimal("0.5"),
        places=4,
        allow_nan=False,
        allow_infinity=False,
    ),
    min_size=1,
    max_size=8,
)


@st.composite
def attribution_inputs(draw: st.DrawFn) -> list[SleeveReturn]:
    """Sleeves whose weights sum to exactly 1 on both sides.

    The same stick-cutting trick as the allocator's weight strategy —
    drawing weights and normalising by division would reintroduce the
    rounding the identity is supposed to survive.
    """
    units = 1_000_000
    n = draw(st.integers(min_value=1, max_value=6))
    names = [f"s{i}" for i in range(n)]

    def cut() -> list[Decimal]:
        cuts = sorted(
            draw(st.lists(st.integers(0, units), min_size=n - 1, max_size=n - 1))
        )
        bounds = [0, *cuts, units]
        return [Decimal(bounds[i + 1] - bounds[i]) / units for i in range(n)]

    pw, bw = cut(), cut()
    pr = [draw(st.integers(-5000, 5000)) for _ in names]
    br = [draw(st.integers(-5000, 5000)) for _ in names]

    return [
        SleeveReturn(
            sleeve=name,
            portfolio_weight=Weight(pw[i]),
            portfolio_return=Decimal(pr[i]) / 10000,
            benchmark_weight=Weight(bw[i]),
            benchmark_return=Decimal(br[i]) / 10000,
        )
        for i, name in enumerate(names)
    ]


@given(sleeves=attribution_inputs())
@settings(max_examples=500)
def test_the_identity_holds_for_any_inputs(sleeves: list[SleeveReturn]) -> None:
    """Allocation + selection + interaction == excess return.

    Five hundred generated portfolios and benchmarks. If this ever
    fails, one of the three effects has the wrong formula — which is the
    single most likely bug in an attribution engine, and otherwise
    invisible because every number still looks plausible.
    """
    result = attribute(sleeves)
    total = result.total_allocation + result.total_selection + result.total_interaction
    assert abs(total - result.excess_return) < EPSILON


@given(sleeves=attribution_inputs())
@settings(max_examples=200)
def test_attribution_is_deterministic(sleeves: list[SleeveReturn]) -> None:
    first, second = attribute(sleeves), attribute(list(reversed(sleeves)))
    assert [r.sleeve for r in first.rows] == [r.sleeve for r in second.rows]
    assert first.excess_return == second.excess_return


@given(
    start=st.integers(min_value=1000, max_value=1_000_000),
    growth=st.integers(min_value=-9000, max_value=50000),
)
@settings(max_examples=300)
def test_twr_matches_the_direct_ratio_when_there_are_no_flows(
    start: int, growth: int
) -> None:
    """With no cash flows, the time-weighted return is just the ratio.
    Anything more elaborate is a bug."""
    begin = Money.from_cents(start * 100)
    end = Money.from_cents(start * 100 + growth * 100)

    result = time_weighted_return(
        [Valuation(date(2026, 1, 1), begin), Valuation(date(2026, 12, 31), end)], []
    )
    expected = (end.amount - begin.amount) / begin.amount
    assert abs(result - expected) < EPSILON


@given(rate=st.integers(min_value=-500, max_value=2000))
@settings(max_examples=200)
def test_annualising_a_year_is_the_identity(rate: int) -> None:
    total = Decimal(rate) / 1000
    assert abs(annualize(total, days=365) - total) < Decimal("1e-9")


@given(levels=st.lists(st.floats(1.0, 1000.0), min_size=2, max_size=50))
@settings(max_examples=300)
def test_drawdown_is_between_zero_and_one(levels: list[float]) -> None:
    assert 0.0 <= max_drawdown(levels) < 1.0
