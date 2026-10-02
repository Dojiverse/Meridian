"""Two years of a household, with the engine making every decision.

============================================================
THE SCRIPT AND THE ENGINE
============================================================

Two kinds of thing happen in this history, and the line between them
is the whole point.

  SCRIPTED — client activity. The opening deposits, quarterly
  contributions, a withdrawal for a car, the dividends the funds pay,
  the firm's quarterly fee, and two purchases the client directed
  themselves: gold in the taxable account at the top of a spike, and
  gold in the Roth a few weeks later. These are inputs. A real firm
  gets them from the custodian.

  ENGINE — everything else. On the fifteenth of every month each
  account is reviewed: drift is computed, a proposal is generated with
  min-tax lot selection, the compliance gate runs the wash-sale screen
  across the whole household, and the proposal is either approved and
  applied to the ledger or rejected with the gate's reason. Every one
  of those decisions is appended to the hash-chained audit log at the
  moment it is made.

No trade in this history was typed in. If the engine would not have
made it, it did not happen.

============================================================
THE STORY IT TELLS
============================================================

The client buys $40,000 of gold in November 2025, near the top. Gold
collapses. In December the rebalancer wants to trim the overweight
and, being tax-aware, picks the lot sitting at a loss. The gate stops
that one trade: the client bought gold in their Roth on 1 December, so
under Rev. Rul. 2008-5 the loss would be permanently forfeited rather
than deferred. The rest of the review proceeds. In January the window
has closed, the same sale goes through clean, and the loss is
harvested.

That sequence is not scripted. It falls out of the inputs and the
rules, which is the only reason it is worth showing.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal

from meridian.audit import Action, AuditLog
from meridian.compliance import (
    ComplianceContext,
    ComplianceGate,
    ComplianceResult,
    ConcentrationLimit,
    MinimumCash,
    RestrictedSecurity,
    Severity,
    ShortTermGainLimit,
    WashSaleBlock,
)
from meridian.household import Account, AccountType, Household
from meridian.ledger import (
    Buy,
    Deposit,
    Dividend,
    Fee,
    LedgerEvent,
    Portfolio,
    Sell,
    Withdrawal,
    fold,
)
from meridian.model import BandKind, Model, Sleeve
from meridian.money import Money, Price, Shares, Weight
from meridian.rebalance import (
    Proposal,
    RebalancePolicy,
    Side,
    Trade,
    generate_proposal,
)
from meridian.showcase.prices import PriceBook, build_prices
from meridian.taxlot import TaxLot, TaxRates, build_lots
from meridian.washsale import Acquisition, SubstituteMap

__all__ = [
    "AccountHistory",
    "History",
    "Review",
    "build_history",
]

# ============================================================
# The cast
# ============================================================

START = date(2024, 9, 3)
"""First trading day after Labor Day 2024. Everything begins here."""

TODAY = date(2026, 9, 8)
"""Where the history stops — the same fixed 'today' the demo uses."""

CLASSIFICATION: Mapping[str, str] = {
    "VTI": "equity",
    "AAPL": "equity",
    "BND": "bond",
    "GLD": "alt",
    "IAU": "alt",
}

CLASSIC = Model(
    model_id="classic-60-40",
    version=1,
    name="Classic balanced",
    sleeves={
        "equity": Sleeve(Weight("0.55"), Weight("0.05"), security="VTI"),
        "bond": Sleeve(Weight("0.35"), Weight("0.05"), security="BND"),
        "alt": Sleeve(
            Weight("0.10"), Weight("0.30"), BandKind.RELATIVE, security="GLD"
        ),
    },
)

CONSERVATIVE = Model(
    model_id="conservative-40-60",
    version=1,
    name="Conservative",
    sleeves={
        "equity": Sleeve(Weight("0.40"), Weight("0.05"), security="VTI"),
        "bond": Sleeve(Weight("0.60"), Weight("0.05"), security="BND"),
    },
)
"""No gold sleeve. When the client buys gold here anyway, the engine
sees an untargeted holding — any amount of it is a breach — and the
purchase still counts as a replacement for section 1091."""

SUBSTITUTES = SubstituteMap.symmetric(
    groups=[["GLD", "IAU"], ["VTI", "ITOT"]],
    alternatives={"GLD": ("SLV",), "VTI": ("SCHB",)},
)

HIGH_BRACKET = TaxRates(
    short_term=Weight("0.37"), long_term=Weight("0.20"), niit=Weight("0.038")
)

TAXABLE = Account("taxable-1", AccountType.TAXABLE, "Schwab", "client")
ROTH = Account("roth-1", AccountType.ROTH_IRA, "Schwab", "client")
HOUSEHOLD = Household.of("household-1", [TAXABLE, ROTH])

ADVISOR = "a.advisor"

POLICY = RebalancePolicy(cash_trigger=Weight("0.02"))
"""Band-edge targets, min-tax lots, and act when idle cash passes 2%.
The same policy at every review, so a difference between two reviews
is a difference in the account, not in the knobs."""
REVIEW_TIME = time(20, 30)
"""15:30 Eastern, stored as UTC. The audit log refuses naive timestamps."""


# ============================================================
# Records
# ============================================================


@dataclass(frozen=True, slots=True)
class Review:
    """One monthly review of one account: what the engine proposed, what
    the gate said, and what the advisor did about it."""

    review_id: str
    on: date
    account_id: str
    proposal: Proposal
    compliance: ComplianceResult
    status: str
    """'approved' (possibly as gated, with blocked trades removed),
    'rejected' when nothing survived the gate, or 'no_action' when no
    band was breached."""
    note: str
    applied_seqs: tuple[int, ...]
    """Sequence numbers of the ledger events this review produced."""


@dataclass
class AccountHistory:
    account: Account
    model: Model
    gate: ComplianceGate
    rates: TaxRates | None
    events: list[LedgerEvent] = field(default_factory=list)

    def next_seq(self) -> int:
        return len(self.events) + 1

    def events_through(self, day: date) -> list[LedgerEvent]:
        return [e for e in self.events if e.on <= day]

    def portfolio_on(self, day: date) -> Portfolio:
        return fold(self.events_through(day))

    def lots_on(self, day: date) -> dict[str, list[TaxLot]]:
        return build_lots(self.events_through(day))

    def acquisitions_through(self, day: date) -> list[Acquisition]:
        return [
            Acquisition(
                acquisition_id=f"{self.account.account_id}:{e.ticker}-{e.seq}",
                account_id=self.account.account_id,
                account_type=self.account.account_type,
                ticker=e.ticker,
                on=e.on,
                quantity=e.quantity,
            )
            for e in self.events_through(day)
            if isinstance(e, Buy)
        ]


@dataclass
class History:
    """Everything the showcase needs, as domain objects."""

    start: date
    today: date
    prices: PriceBook
    household: Household
    accounts: dict[str, AccountHistory]
    reviews: list[Review]
    audit: AuditLog
    substitutes: SubstituteMap
    classification: Mapping[str, str]

    def household_acquisitions_through(self, day: date) -> list[Acquisition]:
        found: list[Acquisition] = []
        for state in self.accounts.values():
            found.extend(state.acquisitions_through(day))
        return found

    def review_dates(self) -> list[date]:
        return sorted({r.on for r in self.reviews})


# ============================================================
# Calendar helpers
# ============================================================


def _first_trading_day(prices: PriceBook, year: int, month: int) -> date:
    day = date(year, month, 1)
    while not prices.is_trading_day(day):
        day += timedelta(days=1)
    return day


def _on_or_after(prices: PriceBook, day: date) -> date:
    """The first trading day on or after `day`. Past the end of the
    priced range the date is returned as is, and filtered out later."""
    last = prices.days[-1]
    while day <= last and not prices.is_trading_day(day):
        day += timedelta(days=1)
    return day


def _months(start: date, end: date) -> Iterator[tuple[int, int]]:
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        month += 1
        if month == 13:
            year, month = year + 1, 1


def _at(day: date, when: time = REVIEW_TIME) -> datetime:
    return datetime.combine(day, when, tzinfo=UTC)


# ============================================================
# Scripted client activity
# ============================================================


def _whole_shares(amount: Money, price: Price) -> Shares:
    return Shares(int(amount.amount / price.amount))


INCEPTION_RESERVE = Decimal("0.985")
"""Invest 98.5% on day one. The first quarterly fee lands before the
first contribution, and a fee may overdraw — a custodian charges the
account whether the cash is there or not — so a little is kept back
rather than opening the history with a debit balance."""


def _buy_by_weight(
    state: AccountHistory,
    day: date,
    budget: Money,
    weights: Mapping[str, Decimal],
    prices: Mapping[str, Price],
) -> None:
    """Open a position set in whole shares, by target weight."""
    for ticker in sorted(weights):
        slice_ = Money(budget.amount * weights[ticker] * INCEPTION_RESERVE)
        quantity = _whole_shares(slice_, prices[ticker])
        if quantity.is_zero:
            continue
        state.events.append(
            Buy(state.next_seq(), day, ticker, quantity, prices[ticker])
        )


def _dividend(
    state: AccountHistory,
    day: date,
    ticker: str,
    rate: Decimal,
    prices: Mapping[str, Price],
) -> None:
    """A cash dividend as a fraction of the position's value that day."""
    held = state.portfolio_on(day).shares_of(ticker)
    if held.is_zero:
        return
    amount = Money(held.value_at(prices[ticker]).amount * rate).quantize()
    if amount.is_zero:
        return
    state.events.append(Dividend(state.next_seq(), day, ticker, amount))


def _fee(
    state: AccountHistory, day: date, prices: Mapping[str, Price], label: str
) -> None:
    """The firm's quarterly advisory fee: 0.25% of value, 1% a year."""
    value = state.portfolio_on(day).total_value(prices)
    amount = Money(value.amount * Decimal("0.0025")).quantize()
    state.events.append(Fee(state.next_seq(), day, amount, label))


# ============================================================
# The review — where the engine runs
# ============================================================


class _Reviewer:
    def __init__(self, history: History) -> None:
        self.history = history
        self.count = 0

    def review(self, state: AccountHistory, day: date) -> Review:
        history = self.history
        prices = history.prices.on(day)
        portfolio = state.portfolio_on(day)
        lots = state.lots_on(day)
        taxable = not state.account.account_type.is_tax_advantaged

        proposal = generate_proposal(
            portfolio,
            prices,
            state.model,
            history.classification,
            on=day,
            policy=POLICY,
            lots=lots if taxable else None,
            rates=state.rates if taxable else None,
        )
        compliance = state.gate.evaluate(
            ComplianceContext(
                proposal=proposal,
                portfolio=portfolio,
                prices=prices,
                on=day,
                account_id=state.account.account_id,
                account_type=state.account.account_type,
                lots=lots,
                acquisitions=history.household_acquisitions_through(day),
                substitutes=history.substitutes,
            )
        )

        self.count += 1
        review_id = f"rev-{self.count:03d}"

        if proposal.is_empty:
            return Review(
                review_id=review_id,
                on=day,
                account_id=state.account.account_id,
                proposal=proposal,
                compliance=compliance,
                status="no_action",
                note="every sleeve inside its band",
                applied_seqs=(),
            )

        self._audit(
            "rebalancer",
            Action.PROPOSAL_GENERATED,
            review_id,
            day,
            account_id=state.account.account_id,
            model=f"{state.model.model_id} v{state.model.version}",
            trades=str(len(compliance.passed)),
            turnover=str(proposal.turnover.amount),
        )
        for violation in compliance.violations:
            if violation.severity is Severity.BLOCK:
                self._audit(
                    "compliance-gate",
                    Action.CONSTRAINT_VIOLATED,
                    review_id,
                    day,
                    constraint=violation.constraint_id,
                    ticker=violation.subject,
                    authority=violation.authority,
                )

        if not compliance.passed:
            # Nothing survived the gate. There is no approvable subset,
            # so the review is rejected with the gate's own reason.
            reason = compliance.blocks[0].message
            self._audit(
                ADVISOR,
                Action.PROPOSAL_REJECTED,
                review_id,
                day,
                reason=reason,
            )
            return Review(
                review_id=review_id,
                on=day,
                account_id=state.account.account_id,
                proposal=proposal,
                compliance=compliance,
                status="rejected",
                note=reason,
                applied_seqs=(),
            )

        # Approved AS GATED. The gate has already removed every blocked
        # trade and re-checked that the survivors are fundable, so what
        # executes is exactly what it allowed — not an override of
        # anything. The blocked trades stay on the record with their
        # reasons, and each one was audited above.
        if compliance.blocked:
            note = (
                f"approved as gated: {len(compliance.blocked)} trade(s) removed "
                f"by the compliance gate, {len(compliance.passed)} executed"
            )
        else:
            note = "reviewed and approved; sells first, then buys"
        self._audit(
            ADVISOR,
            Action.PROPOSAL_APPROVED,
            review_id,
            day,
            account_id=state.account.account_id,
            executed=str(len(compliance.passed)),
            removed=str(len(compliance.blocked)),
            note=note,
        )

        events = _events_for(compliance.passed, day, state.next_seq())
        seqs: list[int] = []
        for event in events:
            state.events.append(event)
            seqs.append(event.seq)
            ticker = getattr(event, "ticker", "")
            self._audit(
                ADVISOR,
                Action.ORDER_SUBMITTED,
                review_id,
                day,
                seq=str(event.seq),
                ticker=ticker,
                side=type(event).__name__.lower(),
                when=time(20, 31),
            )
            self._audit(
                "custodian",
                Action.ORDER_FILLED,
                review_id,
                day,
                seq=str(event.seq),
                ticker=ticker,
                price=str(prices[ticker].amount),
                when=time(20, 45),
            )

        return Review(
            review_id=review_id,
            on=day,
            account_id=state.account.account_id,
            proposal=proposal,
            compliance=compliance,
            status="approved",
            note=note,
            applied_seqs=tuple(seqs),
        )

    def _audit(
        self,
        actor: str,
        action: Action,
        subject: str,
        day: date,
        *,
        when: time = REVIEW_TIME,
        **payload: str,
    ) -> None:
        self.history.audit = self.history.audit.append(
            actor=actor,
            action=action,
            subject=subject,
            payload=payload,
            occurred_at=_at(day, when),
        )


def _events_for(
    trades: Sequence[Trade], on: date, starting_seq: int
) -> list[LedgerEvent]:
    """Ledger events for the trades the gate let through, sells first.

    The same rendering as `Proposal.to_events`, applied to the gated
    subset. Sells must settle before the buys they fund or the ledger
    refuses the buy — which is the correct refusal, and exactly what the
    custodian would do.
    """
    events: list[LedgerEvent] = []
    seq = starting_seq
    ordered = [t for t in trades if t.side is Side.SELL] + [
        t for t in trades if t.side is Side.BUY
    ]
    for trade in ordered:
        if trade.side is Side.SELL:
            events.append(
                Sell(
                    seq,
                    on,
                    trade.ticker,
                    trade.quantity,
                    trade.price,
                    lot_ids=tuple(sel.lot_id for sel in trade.lots),
                )
            )
        else:
            events.append(Buy(seq, on, trade.ticker, trade.quantity, trade.price))
        seq += 1
    return events


# ============================================================
# Building the history
# ============================================================


def build_history(start: date = START, today: date = TODAY) -> History:
    prices = build_prices(start, today)

    taxable = AccountHistory(
        account=TAXABLE,
        model=CLASSIC,
        rates=HIGH_BRACKET,
        gate=ComplianceGate(
            constraints=(
                RestrictedSecurity(
                    constraint_id="ips-4.2",
                    tickers=frozenset({"AAPL"}),
                    reason="client's former employer; may exit but not add",
                    authority="IPS clause 4.2",
                    sell_only=True,
                ),
                ConcentrationLimit(
                    constraint_id="ips-3.1",
                    limit=Weight("0.40"),
                    authority="IPS clause 3.1",
                    exempt=frozenset({"VTI", "BND"}),
                ),
                MinimumCash(
                    constraint_id="ips-6",
                    minimum=Money("0.00"),
                    authority="IPS clause 6",
                ),
                ShortTermGainLimit(
                    constraint_id="tax-stcg",
                    threshold=Money("250.00"),
                    authority="firm tax policy",
                ),
                WashSaleBlock(constraint_id="wash-1091"),
            )
        ),
    )
    roth = AccountHistory(
        account=ROTH,
        model=CONSERVATIVE,
        rates=None,
        gate=ComplianceGate(
            constraints=(
                MinimumCash("ips-6", Money("0.00"), authority="IPS clause 6"),
                ShortTermGainLimit("tax-stcg", Money("250.00")),
            )
        ),
    )

    history = History(
        start=start,
        today=today,
        prices=prices,
        household=HOUSEHOLD,
        accounts={TAXABLE.account_id: taxable, ROTH.account_id: roth},
        reviews=[],
        audit=AuditLog(),
        substitutes=SUBSTITUTES,
        classification=CLASSIFICATION,
    )
    reviewer = _Reviewer(history)

    # ---- the models go on record first ----------------------------
    for model in (CLASSIC, CONSERVATIVE):
        reviewer._audit(
            "investment-committee",
            Action.MODEL_PUBLISHED,
            model.model_id,
            start,
            when=time(14, 0),
            version=str(model.version),
            sleeves=", ".join(
                f"{k} {s.target.as_percent(0)}" for k, s in model.sleeves.items()
            ),
        )

    # ---- day one: funding and initial positions -------------------
    opening = prices.on(start)
    taxable.events.append(Deposit(taxable.next_seq(), start, Money("200000.00")))
    _buy_by_weight(
        taxable,
        start,
        Money("200000.00"),
        # The client arrives holding former-employer stock they may keep
        # but not add to. It sits inside the equity sleeve.
        {
            "VTI": Decimal("0.45"),
            "AAPL": Decimal("0.10"),
            "BND": Decimal("0.35"),
            "GLD": Decimal("0.10"),
        },
        opening,
    )
    roth.events.append(Deposit(roth.next_seq(), start, Money("60000.00")))
    _buy_by_weight(
        roth,
        start,
        Money("60000.00"),
        {"VTI": Decimal("0.40"), "BND": Decimal("0.60")},
        opening,
    )

    # ---- the calendar ----------------------------------------------
    scripted: list[tuple[date, str]] = []
    for year, month in _months(start, today):
        first = _first_trading_day(prices, year, month)
        if first <= start:
            continue
        # Quarterly contribution to the taxable account, then the
        # quarterly fee charged against the previous quarter.
        if month in (1, 4, 7, 10):
            scripted.append((first, "contribution"))
            scripted.append((first, "fee"))
        # Annual Roth contribution, January.
        if month == 1:
            scripted.append((first, "roth-contribution"))
        # BND pays monthly; VTI and AAPL quarterly.
        scripted.append((_on_or_after(prices, date(year, month, 5)), "bnd-dividend"))
        if month in (3, 6, 9, 12):
            scripted.append(
                (_on_or_after(prices, date(year, month, 25)), "vti-dividend")
            )
        if month in (2, 5, 8, 11):
            scripted.append(
                (_on_or_after(prices, date(year, month, 14)), "aapl-dividend")
            )
        # The monthly review.
        review_day = _on_or_after(prices, date(year, month, 15))
        if start < review_day <= today:
            scripted.append((review_day, "review"))

    # One-off client decisions.
    scripted.append((_on_or_after(prices, date(2025, 7, 10)), "withdrawal"))
    # Three days after the November review, so the engine does not get
    # to trim it before it falls.
    scripted.append((_on_or_after(prices, date(2025, 11, 20)), "client-buys-gold"))
    scripted.append(
        (_on_or_after(prices, date(2025, 12, 1)), "client-buys-gold-in-roth")
    )

    # Only what falls inside the history. The calendar above is generous
    # at both ends so it can be reasoned about month by month.
    scripted = [(day, what) for day, what in scripted if start < day <= today]

    # Same-day ordering: cash arrives before anything is charged or
    # reviewed; reviews run last so they see the day's other activity.
    rank = {
        "contribution": 0,
        "roth-contribution": 0,
        "withdrawal": 1,
        "bnd-dividend": 2,
        "vti-dividend": 2,
        "aapl-dividend": 2,
        "fee": 3,
        "client-buys-gold": 4,
        "client-buys-gold-in-roth": 4,
        "review": 9,
    }
    scripted.sort(key=lambda item: (item[0], rank[item[1]], item[1]))

    for day, what in scripted:
        on_day = prices.on(day)
        match what:
            case "contribution":
                taxable.events.append(
                    Deposit(taxable.next_seq(), day, Money("6000.00"))
                )
            case "roth-contribution":
                roth.events.append(Deposit(roth.next_seq(), day, Money("7000.00")))
            case "fee":
                quarter = (day.month - 2) // 3 % 4 + 1
                year = day.year if day.month > 1 else day.year - 1
                label = f"advisory fee Q{quarter} {year}"
                _fee(taxable, day, on_day, label)
                _fee(roth, day, on_day, label)
            case "bnd-dividend":
                _dividend(taxable, day, "BND", Decimal("0.0030"), on_day)
                _dividend(roth, day, "BND", Decimal("0.0030"), on_day)
            case "vti-dividend":
                _dividend(taxable, day, "VTI", Decimal("0.0032"), on_day)
                _dividend(roth, day, "VTI", Decimal("0.0032"), on_day)
            case "aapl-dividend":
                _dividend(taxable, day, "AAPL", Decimal("0.0012"), on_day)
            case "withdrawal":
                taxable.events.append(
                    Withdrawal(taxable.next_seq(), day, Money("4000.00"))
                )
            case "client-buys-gold":
                # Client-directed, at the top of the spike. Funded by a
                # deposit made for the purpose.
                taxable.events.append(
                    Deposit(taxable.next_seq(), day, Money("40000.00"))
                )
                quantity = _whole_shares(Money("40000.00"), on_day["GLD"])
                taxable.events.append(
                    Buy(taxable.next_seq(), day, "GLD", quantity, on_day["GLD"])
                )
            case "client-buys-gold-in-roth":
                # The 2025 Roth contribution, which the client puts
                # straight into gold — a different ticker from the one
                # in the taxable account, and the firm's policy says
                # that does not matter.
                roth.events.append(Deposit(roth.next_seq(), day, Money("7000.00")))
                quantity = _whole_shares(Money("7000.00"), on_day["IAU"])
                roth.events.append(
                    Buy(roth.next_seq(), day, "IAU", quantity, on_day["IAU"])
                )
            case "review":
                for state in (taxable, roth):
                    history.reviews.append(reviewer.review(state, day))
            case _:
                raise AssertionError(f"unknown scripted item {what!r}")

    return history


def event_dates(history: History) -> list[date]:
    """Every day on which anything at all happened, in order."""
    days: set[date] = set()
    for state in history.accounts.values():
        days.update(e.on for e in state.events)
    days.update(r.on for r in history.reviews)
    return sorted(days)


def positions_summary(portfolio: Portfolio) -> Sequence[tuple[str, Shares]]:
    return tuple(sorted(portfolio.positions.items()))
