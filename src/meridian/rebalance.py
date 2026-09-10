"""Turning drift into orders.

============================================================
WHAT THIS MODULE ANSWERS
============================================================

Drift says "you are ten points light on bonds." That is a finding, not
an instruction. This module answers the next question — WHAT TRADES DO I
PLACE? — and produces a Proposal: an immutable record of the trades, the
reasoning, and everything the engine could NOT do and why.

============================================================
FOUR DECISIONS, EACH WITH A COST
============================================================

1. HOW FAR TO TRADE — to the target, or only to the near edge of the
   band. Trading to the band edge moves less, so it costs less in
   spread, commission, and realised gains. Trading to target uses the
   whole budget to buy the least tracking error. The policy chooses;
   neither is universally right.

2. CASH FIRST. The cheapest rebalance places no sell orders at all. If
   there is uninvested cash, it should close the underweights before
   anything is sold. This falls out of the arithmetic rather than
   needing a special case: targets are computed against total investable
   value, so idle cash automatically appears as a shortfall in the
   sleeves and gets deployed first.

3. WHICH SECURITY WITHIN A SLEEVE. "Buy $12,000 of equity" is not an
   order. The sleeve may hold VTI and AAPL, and something has to decide
   the split. Here it is proportional to what is already held — and the
   split runs through the SAME allocate() the money uses everywhere
   else, so a sleeve's trades sum to the sleeve's delta exactly.

   (Phase 04 replaces this for sells, where tax lots make some shares
   far more expensive to sell than others.)

4. ROUNDING DIRECTION. Always toward zero, never nearest. A buy rounded
   up orders more than the cash can fund; a sell rounded up sells shares
   that are not held. Both bounce at the custodian and both produce a
   statement that does not tie. The leftover simply stays in cash.

============================================================
THE INVARIANT
============================================================

A proposal CONSERVES VALUE. Selling $5,000 of a holding produces $5,000
of cash; the portfolio is worth the same before and after, at the same
prices. A rebalance moves value between buckets — it never creates or
destroys any.

That, and the fact that every generated proposal can actually be applied
to the ledger without overdrawing or shorting, are what the property
tests assert.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Final

from meridian.allocate import allocate, normalize_weights
from meridian.drift import DriftReport, SleeveDrift, compute_drift, group_values
from meridian.ledger import Buy, LedgerEvent, Portfolio, Sell, market_values
from meridian.model import Model
from meridian.money import Money, Price, Shares, Weight

__all__ = [
    "Proposal",
    "RebalancePolicy",
    "Side",
    "TargetMode",
    "Trade",
    "Unplaced",
    "generate_proposal",
]


# Policy defaults, named rather than inlined. A dataclass default that
# is a function CALL is evaluated once at class-definition time and
# shared by every instance — harmless for a frozen type like Money, but
# a trap for anything mutable, so the linter rejects the whole shape.
# Naming them also puts the firm's defaults in one greppable place.
DEFAULT_MIN_TRADE: Final = Money("100.00")
DEFAULT_CASH_BUFFER: Final = Money("0")
DEFAULT_SHARE_INCREMENT: Final = Decimal("0.001")
DEFAULT_BAND_ENTRY: Final = Decimal("0.10")


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


class TargetMode(Enum):
    """How far a rebalance moves a breached sleeve."""

    TO_TARGET = "to_target"
    """All the way back to the model weight. Minimises tracking error."""

    TO_BAND_EDGE = "to_band_edge"
    """Only to the near edge of the tolerance band. Minimises turnover,
    and therefore commission, spread, and realised gains. The usual
    default in a taxable account."""


@dataclass(frozen=True, slots=True)
class RebalancePolicy:
    """The knobs, in one place, so a proposal can record which were set."""

    target_mode: TargetMode = TargetMode.TO_BAND_EDGE

    min_trade: Money = DEFAULT_MIN_TRADE
    """Trades smaller than this are not worth placing. A $3 order costs
    more in spread and operational attention than the tracking error it
    removes."""

    share_increment: Decimal = DEFAULT_SHARE_INCREMENT
    """The smallest quantity the custodian will trade. Whole shares are
    Decimal(1); most custodians now support three decimal places."""

    cash_buffer: Money = DEFAULT_CASH_BUFFER
    """Cash held back from investment — for fees, withdrawals, and
    settlement timing. Never deployed by a rebalance."""

    band_entry: Decimal = DEFAULT_BAND_ENTRY
    """How far INSIDE the band to land, as a fraction of band width.

    Aiming at the exact edge is a trap. Share quantities round DOWN, so
    a rebalance targeting precisely 30.0% lands at something like
    29.99995% — five cents short, and therefore still technically in
    breach. The next review flags the same sleeve again, generates a
    five-cent delta, declines to trade it (below min_trade), and reports
    a breach nobody can close.

    Landing a little way inside the band instead means the rounding has
    somewhere to go. 10% of the band width is enough to absorb any
    realistic share increment while still moving as little as possible.

    Set to 0 to aim exactly at the edge and accept the boundary problem.
    """

    def __post_init__(self) -> None:
        if self.min_trade.is_negative:
            raise ValueError("min_trade cannot be negative")
        if self.cash_buffer.is_negative:
            raise ValueError("cash_buffer cannot be negative")
        if self.share_increment <= 0:
            raise ValueError("share_increment must be positive")
        if not 0 <= self.band_entry <= 1:
            raise ValueError("band_entry must be a fraction between 0 and 1")


@dataclass(frozen=True, slots=True)
class Trade:
    """One order. Immutable, and carries the reason it exists."""

    ticker: str
    side: Side
    quantity: Shares
    price: Price
    sleeve: str
    reason: str

    @property
    def consideration(self) -> Money:
        """What it costs (buy) or raises (sell)."""
        return self.quantity.value_at(self.price)

    @property
    def cash_effect(self) -> Money:
        """Signed effect on cash. Buys consume, sells produce."""
        c = self.consideration
        return -c if self.side is Side.BUY else c

    def __str__(self) -> str:
        return (
            f"{self.side.value.upper():4} {self.quantity} {self.ticker} "
            f"@ {self.price} = {self.consideration}"
        )


@dataclass(frozen=True, slots=True)
class Unplaced:
    """Something the engine could not do, and why.

    A proposal that quietly omits what it failed to do is a proposal
    that lies. Every shortfall gets a row here, so the advisor sees the
    gap rather than discovering it at the next review.
    """

    sleeve: str
    shortfall: Money
    reason: str

    def __str__(self) -> str:
        return f"{self.sleeve}: {self.shortfall} not placed — {self.reason}"


@dataclass(frozen=True, slots=True)
class Proposal:
    """The fiduciary artifact.

    Immutable once generated, and carries everything needed to explain
    itself years later: which model version it was measured against,
    what the drift was, what was traded, what was not, and under which
    policy. This is the object an examiner asks to see.
    """

    on: date
    model_id: str
    model_version: int
    policy: RebalancePolicy
    drift_before: DriftReport
    trades: tuple[Trade, ...]
    unplaced: tuple[Unplaced, ...]
    cash_before: Money
    cash_after: Money

    @property
    def is_empty(self) -> bool:
        return not self.trades

    @property
    def buys(self) -> tuple[Trade, ...]:
        return tuple(t for t in self.trades if t.side is Side.BUY)

    @property
    def sells(self) -> tuple[Trade, ...]:
        return tuple(t for t in self.trades if t.side is Side.SELL)

    @property
    def turnover(self) -> Money:
        """Total value traded. The number to minimise."""
        return sum((t.consideration for t in self.trades), Money.zero())

    def to_events(self, starting_seq: int) -> list[LedgerEvent]:
        """Render as ledger events, sells first.

        Ordering matters and is not cosmetic: sells must settle before
        the buys they fund, or the ledger refuses the buy for want of
        cash. That is the correct refusal — it is exactly what the
        custodian would do.
        """
        events: list[LedgerEvent] = []
        seq = starting_seq
        for trade in (*self.sells, *self.buys):
            if trade.side is Side.SELL:
                events.append(
                    Sell(seq, self.on, trade.ticker, trade.quantity, trade.price)
                )
            else:
                events.append(
                    Buy(seq, self.on, trade.ticker, trade.quantity, trade.price)
                )
            seq += 1
        return events

    def __str__(self) -> str:
        if self.is_empty:
            return f"{self.model_id} v{self.model_version} — no trades required"
        lines = [
            f"{self.model_id} v{self.model_version} — "
            f"{len(self.trades)} trade(s), turnover {self.turnover}",
            *(f"  {t}" for t in self.trades),
        ]
        if self.unplaced:
            lines += ["  --", *(f"  {u}" for u in self.unplaced)]
        return "\n".join(lines)


# ============================================================
# Generation
# ============================================================


def generate_proposal(
    portfolio: Portfolio,
    prices: Mapping[str, Price],
    model: Model,
    classification: Mapping[str, str],
    *,
    on: date,
    policy: RebalancePolicy | None = None,
) -> Proposal:
    """Produce the trades that bring `portfolio` back toward `model`.

    Returns an empty proposal — not an error — when no band is breached.
    "Nothing to do" is a valid and common answer, and a rebalancer that
    trades on every review is a rebalancer that costs its clients money.
    """
    policy = policy or RebalancePolicy()

    ticker_values = market_values(portfolio, prices)
    sleeve_values = group_values(ticker_values, classification)

    # Cash is part of the portfolio's value but is not a sleeve to trade
    # into, so it is excluded from the drift denominator here and
    # reintroduced as investable capital below.
    drift = compute_drift(sleeve_values, model)

    total_value = portfolio.total_value(prices)
    investable = total_value - policy.cash_buffer

    if not drift.needs_rebalancing:
        return Proposal(
            on=on,
            model_id=model.model_id,
            model_version=model.version,
            policy=policy,
            drift_before=drift,
            trades=(),
            unplaced=(),
            cash_before=portfolio.cash,
            cash_after=portfolio.cash,
        )

    # ---- what each sleeve should be worth -------------------------
    deltas = _sleeve_deltas(drift, model, investable, policy)

    # ---- turn sleeve deltas into orders ---------------------------
    trades: list[Trade] = []
    unplaced: list[Unplaced] = []

    # Sells first: they raise the cash the buys spend.
    for sleeve, delta in sorted(deltas.items()):
        if delta.is_negative:
            _plan_sells(
                sleeve,
                abs(delta),
                portfolio,
                prices,
                classification,
                policy,
                trades,
                unplaced,
            )

    available = (
        portfolio.cash
        + sum((t.consideration for t in trades), Money.zero())
        - policy.cash_buffer
    )

    for sleeve, delta in sorted(deltas.items()):
        if not delta.is_negative and not delta.is_zero:
            spent = _plan_buys(
                sleeve,
                delta,
                available,
                ticker_values,
                prices,
                classification,
                model,
                policy,
                trades,
                unplaced,
            )
            available = available - spent

    cash_after = portfolio.cash + sum((t.cash_effect for t in trades), Money.zero())

    return Proposal(
        on=on,
        model_id=model.model_id,
        model_version=model.version,
        policy=policy,
        drift_before=drift,
        trades=tuple(trades),
        unplaced=tuple(unplaced),
        cash_before=portfolio.cash,
        cash_after=cash_after,
    )


def _sleeve_deltas(
    drift: DriftReport,
    model: Model,
    investable: Money,
    policy: RebalancePolicy,
) -> dict[str, Money]:
    """How much money each sleeve needs, positive to buy, negative to sell.

    ============================================================
    A BAND SAYS WHETHER TO REBALANCE, NOT WHAT TO TRADE
    ============================================================

    An earlier version of this function moved only the sleeves that had
    breached, on the reasoning that a sleeve inside its band should be
    left alone. That is wrong, and the arithmetic says so.

    Take the drill: bond breaches ten points low, alt breaches five
    points high, equity sits five points high and INSIDE its band. Move
    bond up to 30% and alt down to 13%, leave equity at 60%, and the
    weights sum to 103%. There is no such portfolio. The extra three
    points have to come from somewhere, and if nothing gives them up the
    breach simply cannot be closed — which is exactly what the engine
    reported, honestly and uselessly, as $3,000 unplaced.

    So: the band decides WHETHER a rebalance happens. Once it does, the
    whole portfolio participates, because bringing one sleeve back
    necessarily moves the others. That is also what the practitioner
    literature means by tolerance-band rebalancing, and what
    DriftReport.needs_rebalancing already said.

    The two modes now differ in how far, not in which:

      TO_TARGET     every sleeve returns to its model weight.
      TO_BAND_EDGE  breached sleeves move to the near edge of their
                    band; unbreached sleeves absorb the residual in
                    proportion to their targets, moving as little as the
                    arithmetic allows.
    """
    rows = {r.sleeve: r for r in drift.sleeves}

    if policy.target_mode is TargetMode.TO_TARGET:
        goals = {key: model.target_of(key) for key in rows}
    else:
        goals = _band_edge_goals(rows, model, policy.band_entry)

    return {key: (investable * goals[key]) - rows[key].value for key in rows}


def _band_edge_goals(
    rows: Mapping[str, SleeveDrift], model: Model, band_entry: Decimal
) -> dict[str, Weight]:
    """Target weights for a minimum-turnover rebalance.

    Breached sleeves move to the near edge of their band — an overweight
    comes down to the upper bound, not all the way to target. Everything
    else would prefer to stay exactly where it is, but the weights have
    to sum to one, so the unbreached sleeves share out whatever is left
    over in proportion to their targets.

    Sharing by TARGET rather than by current weight is deliberate: it
    nudges the flexible sleeves toward the model rather than entrenching
    wherever they happen to sit.
    """
    goals: dict[str, Weight] = {}
    flexible: list[str] = []

    for key, row in rows.items():
        sleeve = model.sleeve_of(key)
        if sleeve is None:
            # Held but never targeted. There is no band around zero, so
            # the whole position goes.
            goals[key] = Weight("0")
        elif row.breached:
            # Land just INSIDE the edge, not on it. See RebalancePolicy
            # .band_entry — rounding down onto the boundary leaves the
            # sleeve a few cents outside and the breach uncloseable.
            margin = sleeve.band_width.value * band_entry
            if row.is_overweight:
                goal = sleeve.upper_bound.value - margin
                goals[key] = Weight(max(goal, sleeve.target.value))
            else:
                goal = sleeve.lower_bound.value + margin
                goals[key] = Weight(min(goal, sleeve.target.value))
        else:
            goals[key] = row.actual
            flexible.append(key)

    residual = Decimal(1) - sum((g.value for g in goals.values()), Decimal(0))
    if residual == 0:
        return goals

    # Prefer to push the residual onto the sleeves that have not
    # breached; if every sleeve has, spread it across all of them.
    absorbers = flexible or [k for k in rows if model.sleeve_of(k) is not None]
    weight_of = {k: model.target_of(k).value for k in absorbers}
    total_weight = sum(weight_of.values(), Decimal(0))

    if not absorbers or total_weight == 0:
        # Nothing can absorb it. The shortfall stays as cash, and the
        # unplaced rows downstream will say so rather than the engine
        # pretending it balanced.
        return goals

    for key in absorbers:
        share = residual * weight_of[key] / total_weight
        adjusted = goals[key].value + share
        # A sleeve cannot hold a negative share of the portfolio.
        goals[key] = Weight(max(adjusted, Decimal(0)))

    return goals


def _record_unplaced(
    unplaced: list[Unplaced],
    sleeve: str,
    shortfall: Money,
    reason: str,
    policy: RebalancePolicy,
) -> None:
    """Record a gap, but only one an advisor could actually act on.

    Rounding leaves residue. Sells round down so they raise slightly
    less than asked; buys round down so they spend slightly less than
    offered. On a $100,000 portfolio that lands about eleven cents short
    — which is correct behaviour, not a failure, and the residue stays
    in cash where it belongs.

    Reporting it as "unplaced" would be technically honest and
    practically useless: nobody places an eleven-cent order. A report
    that cries wolf on rounding dust is a report advisors stop reading,
    and then they miss the row that mattered. So the threshold for
    "worth telling someone about" is the same one used for "worth
    trading" — min_trade.
    """
    if shortfall < policy.min_trade:
        return
    unplaced.append(Unplaced(sleeve, shortfall, reason))


def _plan_sells(
    sleeve: str,
    amount: Money,
    portfolio: Portfolio,
    prices: Mapping[str, Price],
    classification: Mapping[str, str],
    policy: RebalancePolicy,
    trades: list[Trade],
    unplaced: list[Unplaced],
) -> None:
    """Raise `amount` from the holdings classified into `sleeve`."""
    holdings = {
        ticker: quantity
        for ticker, quantity in portfolio.positions.items()
        if classification.get(ticker) == sleeve and not quantity.is_zero
    }

    if not holdings:
        _record_unplaced(
            unplaced,
            sleeve,
            amount,
            "sleeve is overweight but holds nothing to sell",
            policy,
        )
        return

    values = {t: q.value_at(prices[t]) for t, q in holdings.items()}
    sleeve_value = sum(values.values(), Money.zero())

    # Cannot raise more than the sleeve is worth.
    raising = amount if amount <= sleeve_value else sleeve_value
    if raising < sleeve_value:
        proportions = {t: v.amount for t, v in values.items()}
        split = allocate(raising, normalize_weights(proportions))
    else:
        split = values  # selling the whole sleeve

    for ticker in sorted(split):
        target_amount = split[ticker]
        if target_amount < policy.min_trade:
            continue

        price = prices[ticker]
        wanted = Shares(target_amount.amount / price.amount).round_to(
            policy.share_increment
        )
        # Never more than is held, even by a rounding step.
        held = holdings[ticker]
        quantity = wanted if wanted.quantity <= held.quantity else held

        if quantity.is_zero:
            continue

        trades.append(
            Trade(
                ticker=ticker,
                side=Side.SELL,
                quantity=quantity,
                price=price,
                sleeve=sleeve,
                reason=f"{sleeve} overweight; raising {target_amount}",
            )
        )

    if amount > sleeve_value:
        _record_unplaced(
            unplaced,
            sleeve,
            amount - sleeve_value,
            "sleeve does not hold enough to raise the full amount",
            policy,
        )


def _plan_buys(
    sleeve: str,
    amount: Money,
    available: Money,
    ticker_values: Mapping[str, Money],
    prices: Mapping[str, Price],
    classification: Mapping[str, str],
    model: Model,
    policy: RebalancePolicy,
    trades: list[Trade],
    unplaced: list[Unplaced],
) -> Money:
    """Deploy up to `amount` into `sleeve`. Returns what was actually spent."""
    if available <= Money.zero():
        _record_unplaced(unplaced, sleeve, amount, "no cash available", policy)
        return Money.zero()

    spendable = amount if amount <= available else available
    if amount > available:
        # The sleeve wants more than the rebalance could raise. Record
        # the gap rather than letting it disappear — an advisor seeing
        # "still underweight" at the next review with no explanation is
        # the failure this row prevents.
        _record_unplaced(
            unplaced,
            sleeve,
            amount - available,
            "insufficient cash after sells",
            policy,
        )

    existing = {
        ticker: value
        for ticker, value in ticker_values.items()
        if classification.get(ticker) == sleeve and not value.is_zero
    }

    if existing:
        # Top up in proportion to what is already held. Same allocate()
        # the money uses everywhere, so the parts sum to the whole.
        split = allocate(
            spendable,
            normalize_weights({t: v.amount for t, v in existing.items()}),
        )
    else:
        # Empty sleeve. The model must name what to buy; guessing would
        # be the engine making an investment decision it has no
        # authority to make.
        sleeve_def = model.sleeve_of(sleeve)
        security = sleeve_def.security if sleeve_def else None
        if security is None:
            _record_unplaced(
                unplaced,
                sleeve,
                amount,
                "sleeve holds nothing and the model names no security to buy",
                policy,
            )
            return Money.zero()
        if security not in prices:
            _record_unplaced(
                unplaced, sleeve, amount, f"no price for {security}", policy
            )
            return Money.zero()
        split = {security: spendable}

    spent = Money.zero()
    for ticker in sorted(split):
        target_amount = split[ticker]
        if target_amount < policy.min_trade:
            continue
        if ticker not in prices:
            _record_unplaced(
                unplaced, sleeve, target_amount, f"no price for {ticker}", policy
            )
            continue

        price = prices[ticker]
        # Round DOWN so the order can always be funded.
        quantity = Shares(target_amount.amount / price.amount).round_to(
            policy.share_increment
        )
        if quantity.is_zero:
            continue

        cost = quantity.value_at(price)
        if cost > available - spent:
            continue

        spent = spent + cost
        trades.append(
            Trade(
                ticker=ticker,
                side=Side.BUY,
                quantity=quantity,
                price=price,
                sleeve=sleeve,
                reason=f"{sleeve} underweight; deploying {target_amount}",
            )
        )

    return spent
