"""Performance — what the return actually was.

============================================================
THERE ARE TWO RIGHT ANSWERS
============================================================

A client puts in $1,000 in January and $50,000 in November, right before
a crash. Did the manager do badly, or did the client pick a bad moment?

TIME-WEIGHTED RETURN answers "was the chef any good?" It cuts the period
at every client deposit and withdrawal and links the pieces
geometrically, so the manager is neither blamed nor credited for WHEN
money arrived — something they do not control.

MONEY-WEIGHTED RETURN answers "how was the meal I actually ate?" It is
the internal rate of return, and it does reflect the client's timing.

Both are correct. Using the wrong one is a compliance problem, not
merely an inaccuracy: the GIPS standards require time-weighted returns
for most composites, permitting money-weighted only where the firm
controls the cash flows and the vehicle is closed-end, fixed-life, or
materially illiquid.

============================================================
GROSS AND NET TRAVEL TOGETHER
============================================================

SEC Marketing Rule 206(4)-1 requires net-of-fee performance wherever
gross is shown, with equal prominence.

So this module does not offer a `return_of(...)` that a caller might
use for gross and forget for net. `PerformanceResult` carries BOTH,
computed in the same call from the same inputs. A net figure that has to
be produced separately is a net figure that will eventually be
forgotten, and the omission is a rule violation rather than an
oversight.

============================================================
WHERE EXACTNESS STOPS, AGAIN
============================================================

Same boundary as `drift.py`, drawn one step further.

  MONEY is exact. Decimal, always.

  RETURNS are ratios of money. Division is not closed over decimals, so
  they carry about 1e-28 of rounding. Still Decimal — they are close to
  money and the precision costs nothing.

  IRR AND RISK STATISTICS are float. An internal rate of return is found
  by iteration, and a standard deviation needs a square root; neither is
  a claim about a client's money. They are estimates of a distribution,
  and pretending otherwise by wrapping them in Decimal would imply a
  precision that does not exist.

The boundary is deliberate. `Decimal` for anything that appears on a
statement, `float` for anything that appears on a risk report.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum
from itertools import pairwise

from meridian.money import Money, Weight

__all__ = [
    "Attribution",
    "AttributionRow",
    "ExternalFlow",
    "PerformanceError",
    "PerformanceResult",
    "RiskStatistics",
    "SleeveReturn",
    "Valuation",
    "annualize",
    "attribute",
    "compute_performance",
    "max_drawdown",
    "modified_dietz",
    "money_weighted_return",
    "risk_statistics",
    "time_weighted_return",
]

TRADING_DAYS_PER_YEAR = 252
"""The conventional count for annualising a daily volatility. A real
NYSE calendar would give the exact figure for a given period; using the
convention is fine as long as the report SAYS which it used, which is
why this is a named constant rather than a bare 252 in a formula."""

DAYS_PER_YEAR = Decimal(365)


class PerformanceError(ValueError):
    """Raised when a return cannot be computed from the inputs given."""


# ============================================================
# Inputs
# ============================================================


@dataclass(frozen=True, slots=True)
class Valuation:
    """The portfolio's market value at the end of a day.

    Taken AFTER any cash flow that day, which is the convention the
    sub-period formula below assumes. Getting that backwards shifts
    every flow into the wrong period and quietly biases the whole
    series.
    """

    on: date
    value: Money


class FlowKind(Enum):
    EXTERNAL = "external"
    """Client contributions and withdrawals. Outside the manager's
    control, so time-weighted return removes their effect."""

    FEE = "fee"
    """Advisory fees. What separates gross from net: a fee is an
    expense of the strategy for NET purposes, and is added back for
    GROSS."""


@dataclass(frozen=True, slots=True)
class ExternalFlow:
    """Money crossing the portfolio boundary.

    Positive is money IN (a contribution), negative is money OUT (a
    withdrawal or a fee).
    """

    on: date
    amount: Money
    kind: FlowKind = FlowKind.EXTERNAL

    def __post_init__(self) -> None:
        if self.kind is FlowKind.FEE and not self.amount.is_negative:
            raise PerformanceError(
                f"a fee must be negative (money leaving), got {self.amount}"
            )


# ============================================================
# Time-weighted return
# ============================================================


def time_weighted_return(
    valuations: Sequence[Valuation],
    flows: Sequence[ExternalFlow],
) -> Decimal:
    """True daily TWR, geometrically linked.

    For consecutive valuations V0 and V1, with net external flow F
    between them:

        r = (V1 - F - V0) / V0

    subtracting the flow because money the client added is not return
    the manager produced. Then:

        TWR = product of (1 + r) - 1

    Geometric linking, not addition: a 50% gain followed by a 50% loss
    is a 25% LOSS, not break-even. Adding period returns is the single
    most common performance bug and it always flatters the manager in
    volatile periods.

    Raises:
        PerformanceError: on a zero starting value with a non-trivial
            period, which has no defined return.
    """
    if len(valuations) < 2:
        raise PerformanceError(
            "a time-weighted return needs at least a start and an end valuation"
        )

    ordered = sorted(valuations, key=lambda v: v.on)
    compound = Decimal(1)

    for previous, current in pairwise(ordered):
        flow = sum(
            (f.amount for f in flows if previous.on < f.on <= current.on),
            Money.zero(),
        )

        if previous.value.is_zero:
            # A period that starts from nothing has no return to speak
            # of — any gain is infinite in percentage terms. Funding a
            # new account is the common case, so treat the sub-period as
            # flat rather than refusing outright.
            if (current.value - flow).is_zero:
                continue
            raise PerformanceError(
                f"period starting {previous.on} opens at zero but ends at "
                f"{current.value}; a return from nothing is undefined"
            )

        sub_period = (current.value - flow - previous.value).amount / (
            previous.value.amount
        )
        compound *= Decimal(1) + sub_period

    return compound - 1


def modified_dietz(
    begin: Money,
    end: Money,
    flows: Sequence[ExternalFlow],
    start: date,
    finish: date,
) -> Decimal:
    """TWR approximated without daily valuations.

        R = (EMV - BMV - F) / (BMV + sum(w_i * F_i))
        w_i = (D - d_i) / D

    where D is the length of the period in days and d_i is the day each
    flow landed. Each flow is weighted by the FRACTION OF THE PERIOD IT
    WAS INVESTED, which is the whole idea: $10,000 arriving on day 2
    should carry nearly full weight, and the same $10,000 arriving on
    day 29 should carry almost none.

    GIPS accepts this for periods without large flows, which is why it
    is here at all — you will meet it in any legacy book of business.
    With daily valuations available, use `time_weighted_return`
    instead. Compute Dietz monthly and link the months geometrically.
    """
    if finish <= start:
        raise PerformanceError(f"period must be positive: {start} to {finish}")

    days = Decimal((finish - start).days)
    net_flow = sum((f.amount for f in flows), Money.zero())

    weighted = Decimal(0)
    for flow in flows:
        if not start <= flow.on <= finish:
            raise PerformanceError(
                f"flow on {flow.on} falls outside the period {start}..{finish}"
            )
        day = Decimal((flow.on - start).days)
        weighted += ((days - day) / days) * flow.amount.amount

    denominator = begin.amount + weighted
    if denominator == 0:
        raise PerformanceError(
            "average invested capital is zero; the return is undefined"
        )

    return (end.amount - begin.amount - net_flow.amount) / denominator


# ============================================================
# Money-weighted return
# ============================================================


def money_weighted_return(
    flows: Sequence[tuple[date, Money]],
    *,
    guess: float = 0.1,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> float:
    """The internal rate of return — XIRR.

        sum( CF_i / (1 + r) ** (d_i / 365) ) = 0

    Solved numerically, and returned as a FLOAT on purpose: an IRR is
    found by iteration and has no exact decimal form, so dressing it in
    Decimal would imply a precision it does not have.

    Newton-Raphson converges fast but can wander off on sign-alternating
    flows, so it is bracketed by bisection and falls back to it. A
    solver that silently returns a wrong root is worse than one that
    says it could not find one.

    Args:
        flows: (date, amount) pairs. Contributions negative, withdrawals
            and the final value positive — the sign convention of a cash
            flow to the INVESTOR.
    """
    if len(flows) < 2:
        raise PerformanceError("an IRR needs at least two cash flows")

    ordered = sorted(flows, key=lambda f: f[0])
    origin = ordered[0][0]

    amounts = [float(amount.amount) for _, amount in ordered]
    years = [(when - origin).days / 365.0 for when, _ in ordered]

    if not (any(a > 0 for a in amounts) and any(a < 0 for a in amounts)):
        raise PerformanceError(
            "an IRR needs both positive and negative flows; "
            "money must go in and come out"
        )

    def npv(rate: float) -> float:
        total = 0.0
        for amount, t in zip(amounts, years, strict=True):
            total += amount / (1.0 + rate) ** t
        return total

    def derivative(rate: float) -> float:
        total = 0.0
        for amount, t in zip(amounts, years, strict=True):
            if t > 0:
                total += -t * amount / (1.0 + rate) ** (t + 1)
        return total

    # --- Newton-Raphson ---
    rate = guess
    for _ in range(max_iterations):
        try:
            value = npv(rate)
        except (OverflowError, ZeroDivisionError):
            break
        if abs(value) < tolerance:
            return rate
        slope = derivative(rate)
        if slope == 0:
            break
        step = value / slope
        rate -= step
        if rate <= -1.0:
            # Below -100% the discount factor is undefined. Reset into
            # the valid range rather than chasing a root that cannot
            # exist.
            rate = -0.99
            break
        if abs(step) < tolerance:
            return rate

    # --- Bisection fallback ---
    # Slower, but it cannot diverge: if a sign change exists in the
    # bracket, it will be found.
    low, high = -0.9999, 10.0
    try:
        low_value, high_value = npv(low), npv(high)
    except (OverflowError, ZeroDivisionError) as exc:
        raise PerformanceError("could not evaluate the IRR bracket") from exc

    if low_value * high_value > 0:
        raise PerformanceError(
            "no internal rate of return exists between -99.99% and 1000% "
            "for these flows"
        )

    for _ in range(200):
        middle = (low + high) / 2
        value = npv(middle)
        if abs(value) < tolerance:
            return middle
        if low_value * value < 0:
            high = middle
        else:
            low, low_value = middle, value

    return (low + high) / 2


# ============================================================
# Annualisation — the GIPS refusal
# ============================================================


def annualize(total_return: Decimal, days: int) -> Decimal:
    """Convert a cumulative return to an annual rate.

        annual = (1 + R) ** (365 / days) - 1

    REFUSES periods shorter than a year. This is not a stylistic
    preference: the GIPS standards prohibit annualising a return for a
    period of less than one year, and presenting one in an advertisement
    is a Marketing Rule problem.

    A 4% quarter is a 4% quarter. Extrapolating it to 17% invents
    performance for nine months that have not happened.

    The function raises rather than warns, because a warning in a log is
    not a control — it is a thing nobody reads until after the
    advertisement went out.
    """
    if days <= 0:
        raise PerformanceError("cannot annualise a period of zero days")

    if days < 365:
        raise PerformanceError(
            f"refusing to annualise a {days}-day period. The GIPS standards "
            "prohibit annualising returns for periods of less than one year — "
            "report the cumulative return for the period instead."
        )

    exponent = float(DAYS_PER_YEAR) / days
    return Decimal(str((1 + float(total_return)) ** exponent - 1))


# ============================================================
# The result — gross and net together
# ============================================================


@dataclass(frozen=True, slots=True)
class PerformanceResult:
    """Gross and net, computed together so neither can be reported alone.

    SEC Marketing Rule 206(4)-1 requires net performance wherever gross
    is shown, with equal prominence. Making them one object rather than
    two function calls turns that from a rule someone must remember into
    a shape the code enforces.
    """

    start: date
    finish: date
    beginning_value: Money
    ending_value: Money

    twr_gross: Decimal
    twr_net: Decimal
    net_external_flow: Money
    fees_paid: Money

    @property
    def days(self) -> int:
        return (self.finish - self.start).days

    @property
    def fee_drag(self) -> Decimal:
        """How much of the gross return the fee consumed."""
        return self.twr_gross - self.twr_net

    @property
    def can_annualize(self) -> bool:
        return self.days >= 365

    def annualized(self) -> tuple[Decimal, Decimal]:
        """(gross, net), or a refusal for a sub-year period."""
        return annualize(self.twr_gross, self.days), annualize(self.twr_net, self.days)

    def __str__(self) -> str:
        gross = Weight(self.twr_gross).as_percent(2)
        net = Weight(self.twr_net).as_percent(2)
        period = "annualized" if self.can_annualize else f"{self.days}-day cumulative"
        return (
            f"{self.start} to {self.finish} ({period}): "
            f"{gross} gross / {net} net "
            f"— fees {self.fees_paid}"
        )


def compute_performance(
    valuations: Sequence[Valuation],
    flows: Sequence[ExternalFlow],
) -> PerformanceResult:
    """Time-weighted return, gross and net, in one pass.

    The difference between the two is entirely in how fees are treated:

      NET   — the fee is an expense of the strategy. It reduces the
              portfolio's value and therefore its return, exactly as the
              client experienced it.

      GROSS — the fee is added back, treated as an external outflow, so
              it does not count against the manager's result.

    Everything else is identical, which is why both come out of the same
    inputs and neither can be quietly computed with different data.
    """
    if len(valuations) < 2:
        raise PerformanceError("performance needs a start and an end valuation")

    ordered = sorted(valuations, key=lambda v: v.on)

    external = [f for f in flows if f.kind is FlowKind.EXTERNAL]
    fees = [f for f in flows if f.kind is FlowKind.FEE]

    return PerformanceResult(
        start=ordered[0].on,
        finish=ordered[-1].on,
        beginning_value=ordered[0].value,
        ending_value=ordered[-1].value,
        # Net: fees stay inside the portfolio, dragging the return down.
        twr_net=time_weighted_return(ordered, external),
        # Gross: fees are treated as external, so they do not count
        # against the manager.
        twr_gross=time_weighted_return(ordered, [*external, *fees]),
        net_external_flow=sum((f.amount for f in external), Money.zero()),
        fees_paid=abs(sum((f.amount for f in fees), Money.zero())),
    )


# ============================================================
# Attribution — Brinson-Fachler
# ============================================================


@dataclass(frozen=True, slots=True)
class SleeveReturn:
    """One sleeve's weight and return, in the portfolio and the benchmark."""

    sleeve: str
    portfolio_weight: Weight
    portfolio_return: Decimal
    benchmark_weight: Weight
    benchmark_return: Decimal


@dataclass(frozen=True, slots=True)
class AttributionRow:
    """Why one sleeve helped or hurt."""

    sleeve: str
    allocation: Decimal
    """Did we hold the right sleeves? Positive when we overweighted a
    sleeve that beat the overall benchmark."""

    selection: Decimal
    """Did we pick the right securities within the sleeve?"""

    interaction: Decimal
    """The cross term — overweighting a sleeve we also picked well in.
    Small, and awkward to explain to clients, which is why some firms
    fold it into selection."""

    @property
    def total(self) -> Decimal:
        return self.allocation + self.selection + self.interaction

    def __str__(self) -> str:
        return (
            f"{self.sleeve}: allocation {Weight(self.allocation).as_percent(2)}, "
            f"selection {Weight(self.selection).as_percent(2)}, "
            f"total {Weight(self.total).as_percent(2)}"
        )


@dataclass(frozen=True, slots=True)
class Attribution:
    rows: tuple[AttributionRow, ...]
    portfolio_return: Decimal
    benchmark_return: Decimal

    @property
    def excess_return(self) -> Decimal:
        return self.portfolio_return - self.benchmark_return

    @property
    def total_allocation(self) -> Decimal:
        return sum((r.allocation for r in self.rows), Decimal(0))

    @property
    def total_selection(self) -> Decimal:
        return sum((r.selection for r in self.rows), Decimal(0))

    @property
    def total_interaction(self) -> Decimal:
        return sum((r.interaction for r in self.rows), Decimal(0))

    def __str__(self) -> str:
        header = (
            f"excess {Weight(self.excess_return).as_percent(2)} = "
            f"allocation {Weight(self.total_allocation).as_percent(2)} + "
            f"selection {Weight(self.total_selection).as_percent(2)} + "
            f"interaction {Weight(self.total_interaction).as_percent(2)}"
        )
        return "\n".join([header, *(f"  {r}" for r in self.rows)])


def attribute(sleeves: Sequence[SleeveReturn]) -> Attribution:
    """Brinson-Fachler attribution.

        Allocation_i  = (wp_i - wb_i) * (Rb_i - Rb)
        Selection_i   = wb_i * (Rp_i - Rb_i)
        Interaction_i = (wp_i - wb_i) * (Rp_i - Rb_i)

    Note the `- Rb` in the allocation term. That is the FACHLER
    refinement, and it is not cosmetic.

    Brinson-Hood-Beebower uses the raw sector return, which credits you
    for overweighting any sector with a positive return — even one that
    trailed the benchmark badly. Subtracting the total benchmark return
    means you are credited only for overweighting a sector that ACTUALLY
    BEAT the benchmark, which is what an advisor means when they say the
    allocation decision worked.

    The three effects sum exactly to the excess return. That identity is
    a free property test, and `test_the_effects_sum_to_the_excess_return`
    asserts it over generated inputs.
    """
    if not sleeves:
        raise PerformanceError("attribution needs at least one sleeve")

    benchmark_return = sum(
        (s.benchmark_weight.value * s.benchmark_return for s in sleeves), Decimal(0)
    )
    portfolio_return = sum(
        (s.portfolio_weight.value * s.portfolio_return for s in sleeves), Decimal(0)
    )

    rows = tuple(
        AttributionRow(
            sleeve=s.sleeve,
            allocation=(s.portfolio_weight.value - s.benchmark_weight.value)
            * (s.benchmark_return - benchmark_return),
            selection=s.benchmark_weight.value
            * (s.portfolio_return - s.benchmark_return),
            interaction=(s.portfolio_weight.value - s.benchmark_weight.value)
            * (s.portfolio_return - s.benchmark_return),
        )
        for s in sorted(sleeves, key=lambda s: s.sleeve)
    )

    return Attribution(
        rows=rows,
        portfolio_return=portfolio_return,
        benchmark_return=benchmark_return,
    )


# ============================================================
# Risk statistics — float territory
# ============================================================


@dataclass(frozen=True, slots=True)
class RiskStatistics:
    """Estimates of a distribution, not claims about money.

    All floats, deliberately. See the module docstring: a standard
    deviation needs a square root and an annualisation factor of
    sqrt(252); wrapping that in Decimal would imply a precision that
    does not exist.
    """

    volatility: float
    """Annualised standard deviation of returns."""

    sharpe: float | None
    tracking_error: float | None
    information_ratio: float | None
    beta: float | None
    max_drawdown: float

    observations: int
    periods_per_year: int = TRADING_DAYS_PER_YEAR


def risk_statistics(
    portfolio_returns: Sequence[float],
    *,
    benchmark_returns: Sequence[float] | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> RiskStatistics:
    """Volatility, Sharpe, tracking error, information ratio, beta, drawdown.

    Args:
        portfolio_returns: PERIODIC returns (daily, usually), not
            cumulative.
        benchmark_returns: Same length as the portfolio series. Without
            it, the benchmark-relative statistics are None rather than
            zero — a missing measurement is not a measurement of zero.
        risk_free_rate: Annual rate, de-annualised internally.
    """
    n = len(portfolio_returns)
    if n < 2:
        raise PerformanceError("risk statistics need at least two observations")

    mean = sum(portfolio_returns) / n
    variance = sum((r - mean) ** 2 for r in portfolio_returns) / (n - 1)
    volatility = math.sqrt(variance) * math.sqrt(periods_per_year)

    annualized_mean = mean * periods_per_year
    sharpe = (annualized_mean - risk_free_rate) / volatility if volatility > 0 else None

    tracking_error: float | None = None
    information_ratio: float | None = None
    beta: float | None = None

    if benchmark_returns is not None:
        if len(benchmark_returns) != n:
            raise PerformanceError(
                f"benchmark has {len(benchmark_returns)} observations but the "
                f"portfolio has {n}; they must align period for period"
            )

        active = [
            p - b for p, b in zip(portfolio_returns, benchmark_returns, strict=True)
        ]
        active_mean = sum(active) / n
        active_variance = sum((a - active_mean) ** 2 for a in active) / (n - 1)
        tracking_error = math.sqrt(active_variance) * math.sqrt(periods_per_year)

        if tracking_error > 0:
            information_ratio = (active_mean * periods_per_year) / tracking_error

        benchmark_mean = sum(benchmark_returns) / n
        benchmark_variance = sum(
            (b - benchmark_mean) ** 2 for b in benchmark_returns
        ) / (n - 1)
        if benchmark_variance > 0:
            covariance = sum(
                (p - mean) * (b - benchmark_mean)
                for p, b in zip(portfolio_returns, benchmark_returns, strict=True)
            ) / (n - 1)
            beta = covariance / benchmark_variance

    return RiskStatistics(
        volatility=volatility,
        sharpe=sharpe,
        tracking_error=tracking_error,
        information_ratio=information_ratio,
        beta=beta,
        max_drawdown=max_drawdown(_to_levels(portfolio_returns)),
        observations=n,
        periods_per_year=periods_per_year,
    )


def max_drawdown(levels: Sequence[float]) -> float:
    """The worst peak-to-trough fall, as a positive fraction.

    Returned positive because "a 32% drawdown" is how anyone says it,
    and a sign convention nobody uses in speech is a sign convention
    someone will misread.
    """
    if len(levels) < 2:
        return 0.0

    peak = levels[0]
    worst = 0.0
    for level in levels:
        peak = max(peak, level)
        if peak > 0:
            worst = max(worst, (peak - level) / peak)
    return worst


def _to_levels(returns: Sequence[float]) -> list[float]:
    """Turn a return series into an index starting at 1."""
    level = 1.0
    levels = [level]
    for r in returns:
        level *= 1.0 + r
        levels.append(level)
    return levels


def sleeve_returns_from(
    portfolio_weights: Mapping[str, Weight],
    portfolio_returns: Mapping[str, Decimal],
    benchmark_weights: Mapping[str, Weight],
    benchmark_returns: Mapping[str, Decimal],
) -> list[SleeveReturn]:
    """Assemble attribution inputs over the UNION of sleeve names.

    Same union-of-keys discipline as the drift report, for the same
    reason: a sleeve the benchmark holds and the portfolio does not is
    an allocation decision worth attributing, and it would vanish from a
    report that iterated over the portfolio alone.
    """
    zero = Weight("0")
    names = set(portfolio_weights) | set(benchmark_weights)
    return [
        SleeveReturn(
            sleeve=name,
            portfolio_weight=portfolio_weights.get(name, zero),
            portfolio_return=portfolio_returns.get(name, Decimal(0)),
            benchmark_weight=benchmark_weights.get(name, zero),
            benchmark_return=benchmark_returns.get(name, Decimal(0)),
        )
        for name in sorted(names)
    ]
