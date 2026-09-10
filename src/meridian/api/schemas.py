"""What crosses the wire.

============================================================
EVERY MONETARY VALUE IS A JSON STRING
============================================================

This is the single most important rule in the API layer, and it looks
like a mistake until you know why.

JSON has one number type. Every JavaScript engine parses it into an
IEEE-754 double — the same binary floating point that cannot represent
0.10. So the moment a browser runs `JSON.parse` on

    {"total": 1234567890123.45}

that value stops being the number the server sent. The server did its
arithmetic in exact Decimal, wrote it correctly, and the client silently
rounded it on arrival. Nothing errors. The books just stop matching.

Sending it as a string instead:

    {"total": "1234567890123.45"}

crosses intact, and the client parses it with `decimal.js` rather than
letting the language do it wrong by default.

`test_no_number_in_any_response_is_money` walks every response and fails
if a monetary field ever comes back as a JSON number. It is a boundary
test, not a unit test, and it is the one that stops the discipline
eroding one endpoint at a time.

============================================================
SHARES AND WEIGHTS TOO
============================================================

Same reasoning, less obviously. A share quantity of 53.353 and a weight
of 0.166667 are both exact decimals the server computed deliberately.
Anything that survives a round trip as a string should.

Percentages that exist ONLY for display — "55.0%" — are strings for a
different reason: they are already formatted text, and formatting is a
server decision so that every client renders the same number the same
way.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer

from meridian.compliance import ComplianceResult, Severity, Violation
from meridian.drift import DriftReport, SleeveDrift
from meridian.money import Money, Price, Shares, Weight
from meridian.rebalance import Proposal, Side, Trade, Unplaced

# ============================================================
# Serialisers
# ============================================================
# Annotated types rather than per-field validators, so a new endpoint
# CANNOT accidentally emit a raw number: the type itself carries the
# rule. Reaching for `float` in a schema below would be a visible
# decision rather than an oversight.

MoneyStr = Annotated[Money, PlainSerializer(lambda m: str(m.amount), return_type=str)]
SharesStr = Annotated[
    Shares, PlainSerializer(lambda s: str(s.quantity), return_type=str)
]
PriceStr = Annotated[Price, PlainSerializer(lambda p: str(p.amount), return_type=str)]
WeightStr = Annotated[Weight, PlainSerializer(lambda w: str(w.value), return_type=str)]


class Schema(BaseModel):
    """Base for everything on the wire.

    `arbitrary_types_allowed` lets the domain types through; the
    serialisers above decide how they are written. The alternative —
    converting to primitives inside every endpoint — is the shape where
    one endpoint eventually forgets.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


# ============================================================
# Drift
# ============================================================


class SleeveDriftOut(Schema):
    sleeve: str
    value: MoneyStr
    actual: WeightStr
    target: WeightStr
    drift: WeightStr
    band_width: WeightStr
    breached: bool
    in_model: bool

    value_display: str = Field(description="Pre-formatted, e.g. '$60,000.00'")
    actual_display: str = Field(description="Pre-formatted, e.g. '60.0%'")
    drift_display: str

    @classmethod
    def of(cls, row: SleeveDrift) -> SleeveDriftOut:
        sign = "+" if row.drift.value > 0 else ""
        return cls(
            sleeve=row.sleeve,
            value=row.value,
            actual=row.actual,
            target=row.target,
            drift=row.drift,
            band_width=row.band_width,
            breached=row.breached,
            in_model=row.in_model,
            value_display=str(row.value),
            actual_display=row.actual.as_percent(),
            drift_display=f"{sign}{row.drift.as_percent()}",
        )


class DriftOut(Schema):
    account_id: str
    total: MoneyStr
    total_display: str
    model_id: str
    model_version: int
    needs_rebalancing: bool
    sleeves: tuple[SleeveDriftOut, ...]

    @classmethod
    def of(cls, account_id: str, report: DriftReport) -> DriftOut:
        return cls(
            account_id=account_id,
            total=report.total,
            total_display=str(report.total),
            model_id=report.model_id,
            model_version=report.model_version,
            needs_rebalancing=report.needs_rebalancing,
            sleeves=tuple(SleeveDriftOut.of(r) for r in report.sleeves),
        )


# ============================================================
# Proposal
# ============================================================


class TradeOut(Schema):
    ticker: str
    side: Side
    quantity: SharesStr
    price: PriceStr
    consideration: MoneyStr
    consideration_display: str
    sleeve: str
    reason: str
    blocked: bool = False

    @classmethod
    def of(cls, trade: Trade, *, blocked: bool = False) -> TradeOut:
        return cls(
            ticker=trade.ticker,
            side=trade.side,
            quantity=trade.quantity,
            price=trade.price,
            consideration=trade.consideration,
            consideration_display=str(trade.consideration),
            sleeve=trade.sleeve,
            reason=trade.reason,
            blocked=blocked,
        )


class UnplacedOut(Schema):
    sleeve: str
    shortfall: MoneyStr
    shortfall_display: str
    reason: str

    @classmethod
    def of(cls, row: Unplaced) -> UnplacedOut:
        return cls(
            sleeve=row.sleeve,
            shortfall=row.shortfall,
            shortfall_display=str(row.shortfall),
            reason=row.reason,
        )


class ViolationOut(Schema):
    constraint_id: str
    severity: Severity
    subject: str
    message: str
    authority: str

    @classmethod
    def of(cls, violation: Violation) -> ViolationOut:
        return cls(
            constraint_id=violation.constraint_id,
            severity=violation.severity,
            subject=violation.subject,
            message=violation.message,
            authority=violation.authority,
        )


class ProposalOut(Schema):
    """A proposal as an advisor sees it, with the compliance verdict
    attached.

    The two are returned TOGETHER on purpose. A UI that fetched trades
    from one endpoint and violations from another could render a trade
    list with the blocks still loading — and an advisor who approves
    what is on screen has approved something the gate rejected.
    """

    proposal_id: str
    account_id: str
    on: date
    model_id: str
    model_version: int
    status: str

    trades: tuple[TradeOut, ...]
    blocked: tuple[TradeOut, ...]
    unplaced: tuple[UnplacedOut, ...]
    violations: tuple[ViolationOut, ...]

    turnover: MoneyStr
    turnover_display: str
    cash_before: MoneyStr
    cash_after: MoneyStr
    cash_after_display: str

    is_clear: bool
    drift_before: DriftOut

    @classmethod
    def of(
        cls,
        proposal_id: str,
        account_id: str,
        proposal: Proposal,
        compliance: ComplianceResult,
        status: str,
    ) -> ProposalOut:
        return cls(
            proposal_id=proposal_id,
            account_id=account_id,
            on=proposal.on,
            model_id=proposal.model_id,
            model_version=proposal.model_version,
            status=status,
            trades=tuple(TradeOut.of(t) for t in compliance.passed),
            blocked=tuple(TradeOut.of(t, blocked=True) for t in compliance.blocked),
            unplaced=tuple(UnplacedOut.of(u) for u in proposal.unplaced),
            violations=tuple(ViolationOut.of(v) for v in compliance.violations),
            turnover=proposal.turnover,
            turnover_display=str(proposal.turnover),
            cash_before=proposal.cash_before,
            cash_after=proposal.cash_after,
            cash_after_display=str(proposal.cash_after),
            is_clear=compliance.is_clear,
            drift_before=DriftOut.of(account_id, proposal.drift_before),
        )


# ============================================================
# Audit
# ============================================================


class AuditEntryOut(Schema):
    seq: int
    occurred_at: datetime
    actor: str
    action: str
    subject: str
    payload: dict[str, str]
    digest: str


class AuditOut(Schema):
    entries: tuple[AuditEntryOut, ...]
    head: str
    is_intact: bool
    breaks: tuple[str, ...]


# ============================================================
# Requests
# ============================================================


class ApproveRequest(Schema):
    """Approving is an act by a PERSON, so the actor is required.

    There is no default of 'system' or 'api'. An approval that cannot
    name who gave it is not an approval, and Rule 204-2 wants the record
    of who recommended what.
    """

    actor: str = Field(min_length=1)
    note: str = ""


class RejectRequest(Schema):
    actor: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    """Required. A rejection with no reason teaches the next reviewer
    nothing and leaves the record unable to explain itself."""
