"""The compliance gate — the last thing between the engine and an order.

============================================================
WHY THIS IS A SEPARATE LAYER
============================================================

The rebalancer's job is to produce the trades that best track the model.
That is an optimisation. Whether those trades are ALLOWED is a different
question with a different answer, governed by the client's Investment
Policy Statement, the firm's restricted list, and the Advisers Act.

Mixing the two produces an engine that quietly declines to consider
good trades for compliance reasons it cannot articulate. Separating them
means every rejection has a named constraint attached to it, and the
reason travels with the proposal into the audit log.

============================================================
THE IPS IS EXECUTABLE DATA, NOT A PDF
============================================================

"No tobacco stocks. Nothing from my former employer. Keep 2% in cash."

In most firms that lives in a Word document, and the software has no
idea. Here each line is a CONSTRAINT — a predicate over a proposal that
returns violations naming itself. A rule the system cannot check is a
rule the system will eventually break.

============================================================
BLOCK VERSUS WARN
============================================================

Not every constraint is absolute. A restricted-list security is a hard
stop. Realising $400 of short-term gain to close a genuine breach might
be the right call, and an engine that silently refused would be
substituting its judgment for the advisor's.

So violations carry a severity. BLOCK removes the trade; WARN keeps it
and puts the concern in front of the human who signs. Both are recorded
either way — the distinction is about what happens next, not about what
gets written down.

============================================================
BLOCKING A TRADE CAN BREAK THE PROPOSAL
============================================================

The subtlety worth knowing about. Sells fund buys. Remove a sell for
compliance reasons and the buys it was paying for may no longer be
fundable — so the surviving trade list is re-checked against available
cash, and any buy that no longer fits is dropped too, with its own
reason.

A gate that returned an unfundable trade list would be handing the
custodian a set of orders that bounce.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Final, Protocol

from meridian.household import AccountType
from meridian.ledger import Portfolio
from meridian.money import Money, Price, Weight
from meridian.rebalance import Proposal, Side, Trade
from meridian.taxlot import HoldingPeriod, TaxLot, holding_period
from meridian.washsale import Acquisition, SubstituteMap, would_trigger_wash_sale

__all__ = [
    "ComplianceContext",
    "ComplianceGate",
    "ComplianceResult",
    "ConcentrationLimit",
    "Constraint",
    "MinimumCash",
    "PreclearanceRequired",
    "RestrictedSecurity",
    "Severity",
    "ShortTermGainLimit",
    "Violation",
    "WashSaleBlock",
]


NO_LOTS: Final[Mapping[str, Sequence[TaxLot]]] = MappingProxyType({})
"""Empty default for ComplianceContext.lots. A read-only proxy, so a
shared default cannot be mutated by one caller and seen by another."""


class Severity(Enum):
    BLOCK = "block"
    """The trade does not go. No override at this layer — an override is
    a human decision, recorded separately in the audit log."""

    WARN = "warn"
    """The trade goes, and the concern is put in front of the person who
    signs."""


@dataclass(frozen=True, slots=True)
class Violation:
    """One constraint, unhappy about one thing, for a stated reason."""

    constraint_id: str
    severity: Severity
    subject: str
    """The ticker, sleeve, or account the complaint is about."""

    message: str
    authority: str = ""
    """What makes this a rule — an IPS clause, a rule citation, a firm
    policy id. Blank for constraints that are purely preference.

    Populated because "the system said no" is not an answer anyone can
    give a client or an examiner.
    """

    def __str__(self) -> str:
        tag = f" [{self.authority}]" if self.authority else ""
        return f"{self.severity.value.upper()} {self.subject}: {self.message}{tag}"


@dataclass(frozen=True, slots=True)
class ComplianceContext:
    """Everything a constraint is allowed to look at.

    Passed in rather than fetched, so constraints stay pure and
    testable: no database, no clock, no network. The same context always
    produces the same violations.
    """

    proposal: Proposal
    portfolio: Portfolio
    prices: Mapping[str, Price]
    on: date

    account_id: str = ""
    account_type: AccountType = AccountType.TAXABLE
    lots: Mapping[str, Sequence[TaxLot]] = NO_LOTS
    acquisitions: Sequence[Acquisition] = ()
    substitutes: SubstituteMap | None = None

    def total_value(self) -> Money:
        return self.portfolio.total_value(self.prices)


class Constraint(Protocol):
    """A rule that can inspect a proposal and object to parts of it.

    Structural, not inherited: anything with an id and a `check` is a
    constraint. That keeps the firm's own rules from having to import a
    base class out of this module.

    `constraint_id` is declared as a read-only PROPERTY rather than a
    bare annotation. A bare `constraint_id: str` on a Protocol means a
    SETTABLE attribute, which no frozen dataclass can satisfy — and
    every constraint here is frozen, deliberately. The property form
    asks only that the id be readable, which is all a gate needs.
    """

    @property
    def constraint_id(self) -> str: ...

    def check(self, context: ComplianceContext) -> tuple[Violation, ...]: ...


# ============================================================
# Constraints
# ============================================================


@dataclass(frozen=True, slots=True)
class RestrictedSecurity:
    """Securities that must not be bought, or must not be traded at all.

    Covers two different things that behave the same way:

      The firm's RESTRICTED LIST — securities the firm has material
      non-public information about, or is otherwise walled off from.
      Rule 204A-1 requires a code of ethics; this is the part of it a
      trading system can actually enforce.

      Client PROHIBITIONS from the IPS — "no tobacco", "nothing from my
      former employer".

    `sell_only` is the common shape: the client may not ADD to a
    position but may exit one. Forcing a client to hold something they
    have prohibited would be the opposite of the intent.
    """

    constraint_id: str
    tickers: frozenset[str]
    reason: str
    authority: str = ""
    sell_only: bool = False

    def check(self, context: ComplianceContext) -> tuple[Violation, ...]:
        violations = []
        for trade in context.proposal.trades:
            if trade.ticker not in self.tickers:
                continue
            if self.sell_only and trade.side is Side.SELL:
                continue
            violations.append(
                Violation(
                    constraint_id=self.constraint_id,
                    severity=Severity.BLOCK,
                    subject=trade.ticker,
                    message=(
                        f"{trade.side.value} of {trade.ticker} is restricted: "
                        f"{self.reason}"
                    ),
                    authority=self.authority,
                )
            )
        return tuple(violations)


@dataclass(frozen=True, slots=True)
class ConcentrationLimit:
    """No single security above a share of the account.

    Checked POST-TRADE, on the position the proposal would leave behind
    — which is the only check that means anything. A limit tested
    against the current position would happily approve the trade that
    breaches it.
    """

    constraint_id: str
    limit: Weight
    authority: str = ""
    exempt: frozenset[str] = frozenset()
    """Broad-market funds are usually exempt: a 60% position in a total
    market index is not a concentration risk in the sense the rule is
    about."""

    def check(self, context: ComplianceContext) -> tuple[Violation, ...]:
        total = context.total_value()
        if total.is_zero:
            return ()

        after = _positions_after(context)
        violations = []

        for ticker, quantity in sorted(after.items()):
            if ticker in self.exempt or quantity <= 0:
                continue
            price = context.prices.get(ticker)
            if price is None:
                continue

            value = Money(quantity * price.amount)
            share = value.ratio_to(total)
            if share.value > self.limit.value:
                violations.append(
                    Violation(
                        constraint_id=self.constraint_id,
                        severity=Severity.BLOCK,
                        subject=ticker,
                        message=(
                            f"would leave {ticker} at {share.as_percent()} of the "
                            f"account, above the {self.limit.as_percent()} limit"
                        ),
                        authority=self.authority,
                    )
                )
        return tuple(violations)


@dataclass(frozen=True, slots=True)
class MinimumCash:
    """Cash that must survive the rebalance.

    For fees, withdrawals, and settlement timing. A portfolio invested
    to the last cent is one that has to sell something to pay its next
    advisory fee.
    """

    constraint_id: str
    minimum: Money
    authority: str = ""

    def check(self, context: ComplianceContext) -> tuple[Violation, ...]:
        if context.proposal.cash_after >= self.minimum:
            return ()
        return (
            Violation(
                constraint_id=self.constraint_id,
                severity=Severity.BLOCK,
                subject=context.account_id or "account",
                message=(
                    f"would leave {context.proposal.cash_after} in cash, "
                    f"below the {self.minimum} minimum"
                ),
                authority=self.authority,
            ),
        )


@dataclass(frozen=True, slots=True)
class ShortTermGainLimit:
    """Warn when a sale would realise short-term gain above a threshold.

    A WARNING, not a block, on purpose. Realising $400 of short-term
    gain to close a genuine breach may well be the right call, and an
    engine that silently refused would be substituting its judgment for
    the advisor's.

    Short-term gains are taxed at ordinary income rates — up to 37%
    federal in 2026, plus 3.8% NIIT — against 0/15/20% for long-term. It
    is worth knowing about before the order goes, not after.
    """

    constraint_id: str
    threshold: Money
    authority: str = ""

    def check(self, context: ComplianceContext) -> tuple[Violation, ...]:
        if context.account_type.is_tax_advantaged:
            # No taxable event inside an IRA. Warning about one would be
            # noise that trains people to ignore the warnings.
            return ()

        violations = []
        for trade in context.proposal.sells:
            gain = _short_term_gain_of(trade, context)
            if gain > self.threshold:
                violations.append(
                    Violation(
                        constraint_id=self.constraint_id,
                        severity=Severity.WARN,
                        subject=trade.ticker,
                        message=(
                            f"selling {trade.ticker} would realise about {gain} "
                            "of SHORT-TERM gain, taxed at ordinary income rates"
                        ),
                        authority=self.authority,
                    )
                )
        return tuple(violations)


@dataclass(frozen=True, slots=True)
class WashSaleBlock:
    """Block a sale that would have its loss disallowed.

    Runs the Phase 05 screen across the whole household. Severity
    escalates: a loss that would be DEFERRED is a warning, because the
    deduction comes back eventually. A loss that would be PERMANENTLY
    FORFEITED — a replacement bought inside an IRA, Rev. Rul. 2008-5 —
    is a block, because there is no version of that outcome the client
    wanted.
    """

    constraint_id: str
    authority: str = "IRC 1091"

    def check(self, context: ComplianceContext) -> tuple[Violation, ...]:
        if context.substitutes is None:
            return ()

        violations = []
        for trade in context.proposal.sells:
            if not _would_realise_loss(trade, context):
                continue

            blockers = would_trigger_wash_sale(
                trade.ticker, context.on, context.acquisitions, context.substitutes
            )
            if not blockers:
                continue

            forfeits = [b for b in blockers if b.account_type.forfeits_wash_sale_basis]

            if forfeits:
                violations.append(
                    Violation(
                        constraint_id=self.constraint_id,
                        severity=Severity.BLOCK,
                        subject=trade.ticker,
                        message=(
                            f"selling {trade.ticker} at a loss would be washed by "
                            f"the {forfeits[0].on} purchase in "
                            f"{forfeits[0].account_id}, and because that is a "
                            "retirement account the loss would be PERMANENTLY "
                            "FORFEITED, not deferred"
                        ),
                        authority="Rev. Rul. 2008-5",
                    )
                )
            else:
                violations.append(
                    Violation(
                        constraint_id=self.constraint_id,
                        severity=Severity.WARN,
                        subject=trade.ticker,
                        message=(
                            f"selling {trade.ticker} at a loss would be washed by "
                            f"the {blockers[0].on} purchase in "
                            f"{blockers[0].account_id}; the loss would be "
                            "deferred into that lot's basis"
                        ),
                        authority=self.authority,
                    )
                )
        return tuple(violations)


@dataclass(frozen=True, slots=True)
class PreclearanceRequired:
    """Securities that need documented approval before acquisition.

    Rule 204A-1 requires a code of ethics under which access persons
    obtain approval before acquiring beneficial ownership in an IPO or a
    limited offering, and requires the firm to keep records of those
    approval decisions.

    So the constraint does not merely block — it blocks UNTIL a
    documented approval exists. `granted` is that record, and it belongs
    in the audit log as a PRECLEARANCE_GRANTED entry naming who approved
    it.
    """

    constraint_id: str
    tickers: frozenset[str]
    granted: frozenset[str] = frozenset()
    authority: str = "Advisers Act Rule 204A-1"

    def check(self, context: ComplianceContext) -> tuple[Violation, ...]:
        violations = []
        for trade in context.proposal.buys:
            if trade.ticker not in self.tickers:
                continue
            if trade.ticker in self.granted:
                continue
            violations.append(
                Violation(
                    constraint_id=self.constraint_id,
                    severity=Severity.BLOCK,
                    subject=trade.ticker,
                    message=(
                        f"{trade.ticker} requires documented preclearance before "
                        "acquisition and none is on file"
                    ),
                    authority=self.authority,
                )
            )
        return tuple(violations)


# ============================================================
# The gate
# ============================================================


@dataclass(frozen=True, slots=True)
class ComplianceResult:
    """What survived, what did not, and why."""

    passed: tuple[Trade, ...]
    blocked: tuple[Trade, ...]
    violations: tuple[Violation, ...]

    @property
    def is_clear(self) -> bool:
        """No blocks. Warnings may still be present and still matter."""
        return not self.blocked

    @property
    def blocks(self) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.severity is Severity.BLOCK)

    @property
    def warnings(self) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.severity is Severity.WARN)

    def __str__(self) -> str:
        lines = [
            f"{len(self.passed)} passed, {len(self.blocked)} blocked, "
            f"{len(self.warnings)} warning(s)"
        ]
        lines += [f"  {v}" for v in self.violations]
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ComplianceGate:
    """Runs every constraint and decides what leaves the building."""

    constraints: tuple[Constraint, ...]

    def evaluate(self, context: ComplianceContext) -> ComplianceResult:
        violations: list[Violation] = []
        for constraint in self.constraints:
            violations.extend(constraint.check(context))

        # Deterministic ordering: blocks first, then by constraint and
        # subject. A report whose row order depended on which constraint
        # happened to be registered first could not be diffed between
        # runs.
        violations.sort(
            key=lambda v: (v.severity is not Severity.BLOCK, v.constraint_id, v.subject)
        )

        blocked_subjects = {
            v.subject for v in violations if v.severity is Severity.BLOCK
        }

        passed = []
        blocked = []
        for trade in context.proposal.trades:
            if trade.ticker in blocked_subjects:
                blocked.append(trade)
            else:
                passed.append(trade)

        # An account-level block (minimum cash, say) names the account
        # rather than a ticker, and stops everything.
        account_blocked = any(
            v.severity is Severity.BLOCK
            and v.subject in {context.account_id, "account"}
            for v in violations
        )
        if account_blocked:
            blocked = list(context.proposal.trades)
            passed = []

        passed, dropped, funding_violations = _ensure_fundable(passed, context)
        blocked.extend(dropped)
        violations.extend(funding_violations)

        return ComplianceResult(
            passed=tuple(passed),
            blocked=tuple(blocked),
            violations=tuple(violations),
        )


def _ensure_fundable(
    trades: Sequence[Trade], context: ComplianceContext
) -> tuple[list[Trade], list[Trade], list[Violation]]:
    """Drop buys that the surviving sells can no longer pay for.

    Sells fund buys. Remove a sell for compliance reasons and the buys
    it was paying for may not be fundable any more — and a gate that
    returned an unfundable list would be handing the custodian orders
    that bounce.

    Buys are dropped largest-first, because dropping one big order beats
    dropping several small ones that each still leave the client
    partially rebalanced.
    """
    sells = [t for t in trades if t.side is Side.SELL]
    buys = sorted(
        (t for t in trades if t.side is Side.BUY),
        key=lambda t: (-t.consideration.amount, t.ticker),
    )

    available = context.portfolio.cash + sum(
        (t.consideration for t in sells), Money.zero()
    )

    kept: list[Trade] = []
    dropped: list[Trade] = []
    violations: list[Violation] = []

    for buy in buys:
        if buy.consideration <= available:
            kept.append(buy)
            available = available - buy.consideration
        else:
            dropped.append(buy)
            violations.append(
                Violation(
                    constraint_id="funding",
                    severity=Severity.BLOCK,
                    subject=buy.ticker,
                    message=(
                        f"buying {buy.ticker} needs {buy.consideration} but only "
                        f"{available} remains after blocked sells were removed"
                    ),
                )
            )

    return [*sells, *kept], dropped, violations


# ============================================================
# Helpers
# ============================================================


def _positions_after(context: ComplianceContext) -> dict[str, Decimal]:
    """Share counts the proposal would leave behind."""
    after = {
        ticker: quantity.quantity
        for ticker, quantity in context.portfolio.positions.items()
    }
    for trade in context.proposal.trades:
        delta = trade.quantity.quantity
        if trade.side is Side.SELL:
            delta = -delta
        after[trade.ticker] = after.get(trade.ticker, Decimal(0)) + delta
    return after


def _short_term_gain_of(trade: Trade, context: ComplianceContext) -> Money:
    """Approximate short-term gain from selling `trade`, FIFO.

    An estimate rather than the exact figure: the lot selection method
    the trade will actually settle under is decided later. Naming it an
    estimate matters — a warning that claims precision it does not have
    invites someone to rely on it for a tax filing.
    """
    lots = sorted(
        context.lots.get(trade.ticker, ()), key=lambda lot: (lot.acquired, lot.lot_id)
    )
    remaining = trade.quantity.quantity
    gain = Money.zero()

    for lot in lots:
        if remaining <= 0:
            break
        take = min(remaining, lot.quantity.quantity)
        remaining -= take

        if holding_period(lot.acquired, context.on) is not HoldingPeriod.SHORT:
            continue

        proportion = take / lot.quantity.quantity
        proceeds = Money(take * trade.price.amount)
        basis = Money(lot.cost_basis.amount * proportion)
        realised = proceeds - basis
        if not realised.is_negative:
            gain = gain + realised

    return gain


def _would_realise_loss(trade: Trade, context: ComplianceContext) -> bool:
    """Whether selling would realise a loss on any lot, FIFO."""
    lots = sorted(
        context.lots.get(trade.ticker, ()), key=lambda lot: (lot.acquired, lot.lot_id)
    )
    remaining = trade.quantity.quantity

    for lot in lots:
        if remaining <= 0:
            break
        take = min(remaining, lot.quantity.quantity)
        remaining -= take

        proportion = take / lot.quantity.quantity
        proceeds = Money(take * trade.price.amount)
        basis = Money(lot.cost_basis.amount * proportion)
        if (proceeds - basis).is_negative:
            return True

    return False
