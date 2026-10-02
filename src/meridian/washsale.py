"""Wash sales — IRC section 1091.

============================================================
THE RULE
============================================================

Sell at a loss, acquire substantially identical stock within 30 days
BEFORE or AFTER, and the loss is disallowed. The window is 61 days: 30
before, the day of sale, 30 after. Calendar days — weekends and holidays
included.

Selling your car to your brother on Tuesday and buying it back Thursday.
The IRS declines to pretend you ever stopped owning it.

Two consequences follow, and they are the reason this is not just a
boolean:

  BASIS ADJUSTMENT (section 1091(d)) — the disallowed loss is ADDED to
  the replacement's cost basis. The loss is deferred, not destroyed; it
  comes back when the replacement is finally sold.

  HOLDING PERIOD TACKING (section 1223(3)) — the replacement inherits
  the holding period of the shares sold. Selling and rebuying does not
  reset your clock, which cuts both ways: it can hand you long-term
  treatment you had not earned on the calendar.

============================================================
THE THREE THINGS ENGINES GET WRONG
============================================================

1. SCOPE. The rule follows the taxpayer, not the account. A sale in a
   taxable account is washed by a purchase in the spouse's account, in
   the client's IRA, or in a corporation they control. Anything scoped
   to a single account reports clean harvests that are not clean.

2. THE IRA TRAP — Rev. Rul. 2008-5. If the replacement is bought inside
   an IRA or Roth, the loss is disallowed AND the IRA's basis is not
   increased. The loss is not deferred. It is PERMANENTLY FORFEITED.

   An engine that treats every wash sale as a deferral is quietly
   telling a client they will get their deduction back later when they
   never will.

3. DIVIDEND REINVESTMENT. An automatic DRIP purchase inside the window
   triggers the rule exactly as a manual buy does. It is the most common
   accidental wash sale in real portfolios, and it is invisible unless
   you are modelling lots — which is why Phase 04 came first.

============================================================
WHAT THIS MODULE REFUSES TO DECIDE
============================================================

"Substantially identical" has never been defined by Congress or the
IRS. It is a facts-and-circumstances test. Two S&P 500 ETFs from
different issuers is a genuinely contested case on which the IRS has not
ruled and practitioners disagree.

So the engine does not decide it. `SubstituteMap` is CURATED POLICY
DATA, supplied by the firm and auditable, and the engine applies
whatever the firm decided. An algorithm that quietly resolved a
contested legal question would be asserting an authority it does not
have — and would be unable to explain itself when asked why.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Final

from meridian.allocate import allocate, normalize_weights
from meridian.household import AccountType
from meridian.money import Money, Price, Shares
from meridian.taxlot import Disposal, HoldingPeriod, TaxLot, TaxRates

__all__ = [
    "Acquisition",
    "HarvestOpportunity",
    "MatchOutcome",
    "ReplacementMatch",
    "SubstituteMap",
    "WashSale",
    "WashSaleReport",
    "apply_basis_adjustment",
    "blackout_tickers",
    "find_harvest_opportunities",
    "find_wash_sales",
    "wash_sale_window",
    "would_trigger_wash_sale",
]

NO_ALTERNATIVES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType({})
"""Empty default for SubstituteMap.alternatives. A read-only proxy, so
a shared default cannot be mutated by one caller and seen by another."""

WINDOW_DAYS = 30
"""Section 1091: 30 days before and 30 days after. With the sale day
itself that is a 61-day window. Calendar days, not trading days —
weekends and market holidays are inside it."""


def wash_sale_window(sale: date) -> tuple[date, date]:
    """The 61-day window around a sale, inclusive at both ends."""
    return (sale - timedelta(days=WINDOW_DAYS), sale + timedelta(days=WINDOW_DAYS))


# ============================================================
# Substantially identical — curated, not computed
# ============================================================


@dataclass(frozen=True, slots=True)
class SubstituteMap:
    """The firm's policy on what counts as substantially identical.

    Two sets per security, and the distinction is the whole point:

      IDENTICAL    — buying this BLOCKS a loss. Same security always;
                     beyond that it is the firm's judgment call.

      ALTERNATIVES — correlated enough to keep the client's exposure,
                     different enough that the firm is willing to defend
                     the position. What you buy INSTEAD when harvesting.

    Both are data. A firm that decides IVV and VOO are substantially
    identical, and a firm that decides they are not, get the same engine
    and different answers — and each can point at the policy that
    produced its answer.
    """

    identical: Mapping[str, frozenset[str]]
    alternatives: Mapping[str, tuple[str, ...]] = NO_ALTERNATIVES

    def __post_init__(self) -> None:
        # "Substantially identical" is symmetric: if A is identical to
        # B then B is identical to A. Storing it one-way invites a map
        # where selling A and buying B is caught but selling B and
        # buying A is not — a bug that would only ever surface as an
        # inconsistency in someone's tax return.
        for ticker, others in self.identical.items():
            for other in others:
                if ticker not in self.identical.get(other, frozenset()):
                    raise ValueError(
                        f"substitute map is asymmetric: {ticker} lists {other} "
                        f"as identical, but not the reverse. "
                        "'Substantially identical' is a symmetric relation."
                    )

    def are_identical(self, a: str, b: str) -> bool:
        """A security is always substantially identical to itself."""
        if a == b:
            return True
        return b in self.identical.get(a, frozenset())

    def alternatives_for(self, ticker: str) -> tuple[str, ...]:
        return tuple(self.alternatives.get(ticker, ()))

    @classmethod
    def symmetric(
        cls,
        groups: Iterable[Iterable[str]],
        alternatives: Mapping[str, tuple[str, ...]] | None = None,
    ) -> SubstituteMap:
        """Build from groups of mutually identical tickers.

        The convenient constructor, because writing a symmetric mapping
        by hand is exactly the sort of thing people get half-right.
        """
        identical: dict[str, set[str]] = {}
        for group in groups:
            members = list(group)
            for ticker in members:
                identical.setdefault(ticker, set()).update(
                    m for m in members if m != ticker
                )
        return cls(
            identical={k: frozenset(v) for k, v in identical.items()},
            alternatives=dict(alternatives or {}),
        )


# ============================================================
# Acquisitions — anything that could serve as a replacement
# ============================================================


@dataclass(frozen=True, slots=True)
class Acquisition:
    """A purchase anywhere in the household, inside any account.

    Deliberately not a TaxLot. A replacement may sit in an IRA, where
    lots are not tracked for basis purposes at all, or in a spouse's
    account this system may only see a summary of. What matters for
    section 1091 is that shares were acquired, by whom, when, and where.
    """

    acquisition_id: str
    account_id: str
    account_type: AccountType
    ticker: str
    on: date
    quantity: Shares
    is_reinvestment: bool = False
    """Automatic DRIP purchases trigger the rule exactly as manual buys
    do, and are the most common accidental wash sale in real
    portfolios. Flagged separately so a report can say WHY the client
    tripped it, which is usually news to them."""


class MatchOutcome(Enum):
    DEFERRED = "deferred"
    """Loss disallowed, added to the replacement's basis. Recoverable
    when the replacement is sold."""

    FORFEITED = "forfeited"
    """Loss disallowed and NOT added to any usable basis, because the
    replacement sits in a retirement account. Rev. Rul. 2008-5. Gone
    permanently."""


@dataclass(frozen=True, slots=True)
class ReplacementMatch:
    """One acquisition, matched against one loss sale."""

    acquisition_id: str
    account_id: str
    account_type: AccountType
    ticker: str
    on: date
    matched: Shares
    disallowed: Money
    outcome: MatchOutcome
    is_reinvestment: bool

    def __str__(self) -> str:
        how = " (dividend reinvestment)" if self.is_reinvestment else ""
        return (
            f"{self.matched} {self.ticker} on {self.on} in {self.account_id} "
            f"-> {self.disallowed} {self.outcome.value}{how}"
        )


@dataclass(frozen=True, slots=True)
class WashSale:
    """One loss disposal, and what section 1091 does to it."""

    disposal: Disposal
    matched: Shares
    disallowed: Money
    allowed: Money
    """The part of the loss that survives — when fewer shares were
    replaced than sold, only the matched portion is disallowed."""
    replacements: tuple[ReplacementMatch, ...]

    @property
    def forfeited(self) -> Money:
        """The part that is gone permanently rather than deferred."""
        return sum(
            (
                r.disallowed
                for r in self.replacements
                if r.outcome is MatchOutcome.FORFEITED
            ),
            Money.zero(),
        )

    @property
    def deferred(self) -> Money:
        return sum(
            (
                r.disallowed
                for r in self.replacements
                if r.outcome is MatchOutcome.DEFERRED
            ),
            Money.zero(),
        )

    @property
    def is_fully_disallowed(self) -> bool:
        return self.allowed.is_zero

    def __str__(self) -> str:
        parts = [
            f"{self.disposal.quantity} {self.disposal.ticker} sold "
            f"{self.disposal.disposed}: loss {abs(self.disposal.gain)}, "
            f"{abs(self.disallowed)} disallowed"
        ]
        if not self.forfeited.is_zero:
            parts.append(f"  {abs(self.forfeited)} PERMANENTLY FORFEITED (IRA)")
        return "\n".join(parts)


@dataclass(frozen=True, slots=True)
class WashSaleReport:
    findings: tuple[WashSale, ...]

    @property
    def total_disallowed(self) -> Money:
        return sum((f.disallowed for f in self.findings), Money.zero())

    @property
    def total_forfeited(self) -> Money:
        """The number worth putting at the top of any report. This is
        client money destroyed, not deferred."""
        return sum((f.forfeited for f in self.findings), Money.zero())

    @property
    def total_deferred(self) -> Money:
        return sum((f.deferred for f in self.findings), Money.zero())

    def __bool__(self) -> bool:
        return bool(self.findings)


# ============================================================
# Detection
# ============================================================


def find_wash_sales(
    disposals: Sequence[Disposal],
    acquisitions: Sequence[Acquisition],
    substitutes: SubstituteMap,
) -> WashSaleReport:
    """Find every wash sale across a household's activity.

    Args:
        disposals: Every sale in the period. Gains are ignored — section
            1091 applies only to losses.
        acquisitions: Every purchase ANYWHERE in the household, in any
            account type, including reinvestments.
        substitutes: The firm's policy on substantial identity.

    Each replacement share can only absorb one loss, so disposals are
    processed in date order and replacement quantity is consumed as it
    is matched. Without that bookkeeping a single small repurchase would
    appear to disallow several different losses in full.

    A LOT IS NOT ITS OWN REPLACEMENT. Shares bought on 20 November and
    sold on 15 December were acquired inside the window, but they are
    the shares being sold, not shares acquired to replace them. So the
    purchase that created the disposed lot is netted out, up to the
    quantity disposed, before anything is matched. Other shares bought
    the same day — a larger purchase than the sale — can still be
    replacements, which is why this is a quantity and not a flag.

    SHARES SOLD IN THE SAME TRANSACTION ARE NOT REPLACEMENTS EITHER.
    Rev. Rul. 56-602: a taxpayer who buys shares and then sells the
    whole position, old and new, within the window has not replaced
    anything — nothing is held afterwards for the loss to defer into.
    So every disposal on a given day is netted against its own purchase
    before any loss on that day is matched. Without this, selling an
    old lot and a recent small lot together would wash the old lot's
    loss against the recent lot's purchase, which the sale itself
    disposed of.
    """
    findings: list[WashSale] = []

    # How much of each acquisition is still available to serve as a
    # replacement. Consumed as matches are made.
    remaining: dict[str, Decimal] = {
        a.acquisition_id: a.quantity.quantity for a in acquisitions
    }
    by_id = {a.acquisition_id: a for a in acquisitions}

    ordered = sorted(disposals, key=lambda d: (d.disposed, d.lot_id))
    netted_through: date | None = None

    for disposal in ordered:
        # Net every disposal on this day against the purchase that
        # created it, before matching any of them. Shares sold today
        # are gone; they replace nothing, loss or gain.
        if netted_through != disposal.disposed:
            for same_day in ordered:
                if same_day.disposed == disposal.disposed:
                    _consume_own_purchase(
                        remaining,
                        acquisitions,
                        same_day.ticker,
                        same_day.acquired,
                        same_day.quantity.quantity,
                    )
            netted_through = disposal.disposed

        if not disposal.is_loss:
            # Section 1091 disallows LOSSES. A gain is taxable now
            # whatever you buy afterwards.
            continue

        start, end = wash_sale_window(disposal.disposed)

        candidates = [
            a
            for a in acquisitions
            if start <= a.on <= end
            and substitutes.are_identical(disposal.ticker, a.ticker)
            and remaining[a.acquisition_id] > 0
        ]
        # Chronological, with the id as a total order so the same inputs
        # always match the same way.
        candidates.sort(key=lambda a: (a.on, a.acquisition_id))

        if not candidates:
            continue

        matched_by_id: dict[str, Decimal] = {}
        still_to_match = disposal.quantity.quantity

        for candidate in candidates:
            if still_to_match <= 0:
                break
            take = min(still_to_match, remaining[candidate.acquisition_id])
            matched_by_id[candidate.acquisition_id] = take
            remaining[candidate.acquisition_id] -= take
            still_to_match -= take

        matched_quantity = sum(matched_by_id.values(), Decimal(0))
        if matched_quantity <= 0:
            continue

        # Proportional disallowance: replace half the shares and half
        # the loss is disallowed. Section 1091(b).
        loss = disposal.gain  # negative
        proportion = matched_quantity / disposal.quantity.quantity
        disallowed = Money(loss.amount * proportion).quantize()
        allowed = loss - disallowed

        # Split the disallowed amount across the matched acquisitions —
        # the fourth caller of the same largest remainder method, so the
        # parts sum to the whole exactly. They have to: each part
        # becomes a basis adjustment on a different lot, and a cent lost
        # here is a cent of phantom gain later.
        shares = allocate(
            disallowed,
            normalize_weights({k: v for k, v in matched_by_id.items()}),
        )

        matches = tuple(
            ReplacementMatch(
                acquisition_id=acq_id,
                account_id=by_id[acq_id].account_id,
                account_type=by_id[acq_id].account_type,
                ticker=by_id[acq_id].ticker,
                on=by_id[acq_id].on,
                matched=Shares(matched_by_id[acq_id]),
                disallowed=shares[acq_id],
                outcome=(
                    MatchOutcome.FORFEITED
                    if by_id[acq_id].account_type.forfeits_wash_sale_basis
                    else MatchOutcome.DEFERRED
                ),
                is_reinvestment=by_id[acq_id].is_reinvestment,
            )
            for acq_id in sorted(matched_by_id)
        )

        findings.append(
            WashSale(
                disposal=disposal,
                matched=Shares(matched_quantity),
                disallowed=disallowed,
                allowed=allowed,
                replacements=matches,
            )
        )

    return WashSaleReport(findings=tuple(findings))


# ============================================================
# The consequence: adjusting the replacement lot
# ============================================================


def apply_basis_adjustment(
    lot: TaxLot,
    disallowed: Money,
    original_acquired: date,
) -> TaxLot:
    """Apply section 1091(d) and section 1223(3) to a replacement lot.

    Two changes, and the second is the one people forget:

      BASIS goes UP by the disallowed loss. The deduction comes back
      when this lot is eventually sold.

      ACQUIRED DATE moves BACK to the original lot's, because section
      1223(3) starts the replacement's holding period on the same day as
      the shares that were sold. Selling and rebuying does not reset the
      clock.

    `disallowed` arrives as a NEGATIVE number (it is a loss), so it is
    subtracted to increase the basis. Getting that sign backwards halves
    the client's basis instead of raising it, which is the kind of error
    that looks plausible on a screen.
    """
    if not disallowed.is_negative and not disallowed.is_zero:
        raise ValueError(
            f"disallowed loss should be negative or zero, got {disallowed}. "
            "A wash sale defers a LOSS; a positive value here means a sign "
            "was flipped upstream."
        )

    already = lot.disallowed_loss_added or Money.zero()

    return replace(
        lot,
        cost_basis=lot.cost_basis - disallowed,
        acquired=min(lot.acquired, original_acquired),
        disallowed_loss_added=already + disallowed,
    )


# ============================================================
# Pre-trade screen
# ============================================================


def would_trigger_wash_sale(
    ticker: str,
    sale_date: date,
    acquisitions: Sequence[Acquisition],
    substitutes: SubstituteMap,
    *,
    account_id: str | None = None,
    selling: Sequence[tuple[date, Shares]] = (),
) -> tuple[Acquisition, ...]:
    """Which existing acquisitions would wash a loss sold on this date.

    Runs BEFORE the trade, so the order can be blocked or re-timed
    rather than discovered in April. Returns the offending acquisitions
    — an empty tuple means the sale is clear.

    Note this covers the 30 days AFTER the sale as well as before: a
    purchase already scheduled inside the forward window (a DRIP date, a
    recurring contribution) washes a sale that has not happened yet.

    Args:
        account_id: The account the sale is in.
        selling: (acquired, quantity) for each lot being sold. The
            purchase that created each of those lots — same account,
            same ticker, same day — is netted out up to that quantity,
            because the shares being sold are not replacements for
            themselves. Any excess bought that day still counts, and an
            acquisition that is only partly netted is returned with the
            quantity that remains.
    """
    start, end = wash_sale_window(sale_date)

    remaining: dict[str, Decimal] = {
        a.acquisition_id: a.quantity.quantity for a in acquisitions
    }
    if account_id is not None:
        own = [a for a in acquisitions if a.account_id == account_id]
        for acquired, quantity in selling:
            _consume_own_purchase(remaining, own, ticker, acquired, quantity.quantity)

    offenders = []
    for a in sorted(acquisitions, key=lambda a: (a.on, a.acquisition_id)):
        if not (start <= a.on <= end):
            continue
        if not substitutes.are_identical(ticker, a.ticker):
            continue
        left = remaining[a.acquisition_id]
        if left <= 0:
            continue
        offenders.append(
            a if left == a.quantity.quantity else replace(a, quantity=Shares(left))
        )
    return tuple(offenders)


def _consume_own_purchase(
    remaining: dict[str, Decimal],
    acquisitions: Sequence[Acquisition],
    ticker: str,
    acquired: date,
    quantity: Decimal,
) -> None:
    """Net `quantity` of shares sold out of the purchase(s) that created
    them: same ticker, acquired the same day. Oldest id first, so the
    netting is reproducible when several purchases share a day."""
    own = sorted(
        (a for a in acquisitions if a.ticker == ticker and a.on == acquired),
        key=lambda a: a.acquisition_id,
    )
    for a in own:
        if quantity <= 0:
            break
        take = min(quantity, remaining[a.acquisition_id])
        remaining[a.acquisition_id] -= take
        quantity -= take


def blackout_tickers(
    disposals: Sequence[Disposal],
    substitutes: SubstituteMap,
    *,
    on: date,
) -> frozenset[str]:
    """Securities that must not be BOUGHT on `on`, because buying them
    would wash a loss realised in the last 30 days.

    The other half of section 1091. The pre-trade screen stops a sale
    that an earlier purchase would wash; this stops a purchase that
    would wash an earlier sale — the way a harvest is most often undone,
    by the next rebalance quietly buying back what was just sold.

    Returns the tickers sold at a loss within the window, plus every
    ticker the firm's policy treats as identical to them. Gains impose
    no blackout: section 1091 disallows losses only.
    """
    found: set[str] = set()
    for d in disposals:
        if not d.is_loss:
            continue
        if not (on - timedelta(days=WINDOW_DAYS) <= d.disposed <= on):
            continue
        found.add(d.ticker)
        found.update(substitutes.identical.get(d.ticker, frozenset()))
    return frozenset(found)


# ============================================================
# Harvesting
# ============================================================


@dataclass(frozen=True, slots=True)
class HarvestOpportunity:
    """A lot sitting at a loss, and whether it can actually be taken."""

    lot: TaxLot
    account_id: str
    account_type: AccountType
    market_value: Money
    unrealized_loss: Money
    period: HoldingPeriod
    tax_benefit: Money
    """What the deduction is worth at the client's rates. Positive."""

    blocked_by: tuple[Acquisition, ...] = ()
    alternatives: tuple[str, ...] = ()
    """What to buy instead to keep the exposure without tripping the
    rule. Empty means the firm has not designated one, and the position
    would have to go uninvested for 31 days."""

    @property
    def is_blocked(self) -> bool:
        return bool(self.blocked_by)

    @property
    def block_reason(self) -> str:
        """Why the harvest cannot be taken, naming the worst blocker.

        Worst, not first. When several purchases fall inside the window
        the outcome is decided by the most damaging one: a single
        replacement inside an IRA forfeits the loss no matter how many
        taxable purchases also sit in the window. Reporting the earliest
        purchase instead would describe a deferral the client is not
        going to get.
        """
        if not self.blocked_by:
            return ""
        forfeiting = [
            a for a in self.blocked_by if a.account_type.forfeits_wash_sale_basis
        ]
        if forfeiting:
            worst = forfeiting[0]
            return (
                f"would be washed by the {worst.on} purchase in "
                f"{worst.account_id} — and because that is a retirement "
                "account, the loss would be PERMANENTLY FORFEITED "
                "(Rev. Rul. 2008-5), not deferred"
            )
        first = self.blocked_by[0]
        return (
            f"would be washed by the {first.on} purchase in {first.account_id}; "
            "the loss would be deferred into that lot's basis"
        )

    def __str__(self) -> str:
        flag = " [BLOCKED]" if self.is_blocked else ""
        return (
            f"{self.lot.ticker} lot {self.lot.lot_id}: loss "
            f"{abs(self.unrealized_loss)} ({self.period.value}), worth "
            f"{self.tax_benefit}{flag}"
        )


def find_harvest_opportunities(
    lots: Sequence[TaxLot],
    prices: Mapping[str, Price],
    acquisitions: Sequence[Acquisition],
    substitutes: SubstituteMap,
    rates: TaxRates,
    *,
    on: date,
    account_id: str,
    account_type: AccountType,
    minimum_loss: Money | None = None,
) -> tuple[HarvestOpportunity, ...]:
    """Find losses worth taking, and say which ones are blocked.

    Ordered by tax benefit, largest first — the row an advisor acts on.

    Harvesting inside a tax-advantaged account accomplishes nothing:
    there is no loss to deduct because there was never a gain to tax. So
    those accounts return nothing rather than a list of illusory
    opportunities.
    """
    if account_type.is_tax_advantaged:
        return ()

    floor = minimum_loss or Money.zero()
    found: list[HarvestOpportunity] = []

    for lot in lots:
        if lot.ticker not in prices:
            continue

        value = lot.market_value(prices[lot.ticker])
        unrealized = value - lot.cost_basis

        if not unrealized.is_negative:
            continue
        if abs(unrealized) < floor:
            continue

        period = lot.period_at(on)
        benefit = abs(unrealized) * rates.rate_for(period)

        found.append(
            HarvestOpportunity(
                lot=lot,
                account_id=account_id,
                account_type=account_type,
                market_value=value,
                unrealized_loss=unrealized,
                period=period,
                tax_benefit=benefit,
                blocked_by=would_trigger_wash_sale(
                    lot.ticker,
                    on,
                    acquisitions,
                    substitutes,
                    account_id=account_id,
                    selling=[(lot.acquired, lot.quantity)],
                ),
                alternatives=substitutes.alternatives_for(lot.ticker),
            )
        )

    # Largest benefit first, lot id as the total order so two identical
    # opportunities always appear in the same sequence.
    found.sort(key=lambda o: (-o.tax_benefit.amount, o.lot.lot_id))
    return tuple(found)
