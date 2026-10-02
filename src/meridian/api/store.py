"""An in-memory store, so the demo has something to talk to.

============================================================
THIS IS NOT THE DATABASE
============================================================

In production every one of these lives in Postgres: the event stream as
an append-only table with UPDATE and DELETE revoked, the audit log with
its chain head anchored elsewhere, models as immutable versioned rows.

What is here is a stand-in with the SAME SHAPE, so the API layer above
it is written against the interface it will actually have. Two
properties are preserved deliberately, because they are the ones the
rest of the system depends on:

  EVENTS ARE APPENDED, NEVER EDITED. `add_events` extends; nothing
  replaces. Positions are folded on read, exactly as they will be from
  the real table.

  THE AUDIT LOG IS REPLACED WHOLE, NOT MUTATED. `AuditLog.append`
  returns a new log, and the store swaps its reference. A store that
  mutated the log in place would let one request's view of history
  depend on another's timing.

Everything else about it — no persistence, no locking, no transactions
— is demo scaffolding and says so.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

from meridian.audit import AuditLog
from meridian.compliance import ComplianceGate, ComplianceResult
from meridian.household import Account, AccountType
from meridian.ledger import LedgerEvent, Portfolio, fold
from meridian.model import Model
from meridian.money import Price
from meridian.rebalance import Proposal
from meridian.taxlot import Disposal, TaxLot, TaxRates, build_lots, replay_disposals
from meridian.washsale import Acquisition, SubstituteMap

__all__ = ["AccountState", "Store", "StoredProposal"]


@dataclass(frozen=True, slots=True)
class StoredProposal:
    """A generated proposal, its compliance verdict, and its status.

    Frozen. Approving does not mutate a proposal — it stores a new one
    with a new status and appends an audit entry. The thing an advisor
    approved must still say the same thing afterwards.
    """

    proposal_id: str
    account_id: str
    proposal: Proposal
    compliance: ComplianceResult
    status: str = "pending"
    decided_by: str = ""
    note: str = ""


@dataclass
class AccountState:
    """Everything known about one account."""

    account: Account
    model: Model
    classification: Mapping[str, str]
    household_id: str = "household-1"
    events: list[LedgerEvent] = field(default_factory=list)
    gate: ComplianceGate = field(default_factory=lambda: ComplianceGate(()))
    substitutes: SubstituteMap | None = None
    rates: TaxRates | None = None
    """The client's marginal rates, for lot selection. None for a
    tax-advantaged account, where no lot is cheaper than another."""

    def portfolio(self) -> Portfolio:
        """Folded on every read, never cached.

        Caching would be an optimisation that quietly reintroduces the
        thing the ledger exists to avoid: a stored balance that can
        disagree with the events it came from.
        """
        return fold(self.events)

    def lots(self) -> dict[str, list[TaxLot]]:
        return build_lots(self.events)

    def disposals(self) -> list[Disposal]:
        """Every realised sale, from the same replay that builds lots."""
        return replay_disposals(self.events)

    def acquisitions(self) -> list[Acquisition]:
        """Purchases in THIS account. See Store.household_acquisitions —
        the wash-sale screen needs the whole household, not this."""
        from meridian.ledger import Buy

        return [
            Acquisition(
                acquisition_id=f"{e.ticker}-{e.seq}",
                account_id=self.account.account_id,
                account_type=self.account.account_type,
                ticker=e.ticker,
                on=e.on,
                quantity=e.quantity,
            )
            for e in self.events
            if isinstance(e, Buy)
        ]

    def next_seq(self) -> int:
        return max((e.seq for e in self.events), default=0) + 1


@dataclass
class Store:
    """The demo's whole world."""

    accounts: dict[str, AccountState] = field(default_factory=dict)
    prices: dict[str, Price] = field(default_factory=dict)
    proposals: dict[str, StoredProposal] = field(default_factory=dict)
    audit: AuditLog = field(default_factory=AuditLog)
    today: date = date(2026, 9, 8)
    """Fixed rather than `date.today()`. The engine reads no clock, and
    neither does the demo — a page whose numbers change overnight cannot
    be compared against the screenshot someone took yesterday."""

    def account(self, account_id: str) -> AccountState:
        state = self.accounts.get(account_id)
        if state is None:
            raise KeyError(account_id)
        return state

    def add_events(self, account_id: str, events: Sequence[LedgerEvent]) -> None:
        self.account(account_id).events.extend(events)

    def next_proposal_id(self) -> str:
        return f"prop-{len(self.proposals) + 1:04d}"

    def account_type(self, account_id: str) -> AccountType:
        return self.account(account_id).account.account_type

    def household_acquisitions(self, account_id: str) -> list[Acquisition]:
        """Every purchase across the whole HOUSEHOLD.

        This is the scope section 1091 actually uses. A sale in a
        taxable account is washed by a purchase in the spouse's account,
        in the client's IRA, or in a corporation they control — so a
        screen that only saw one account would report a clean harvest
        that is not clean.

        It also changes the SEVERITY of what is found. A replacement in
        another taxable account defers the loss; the same replacement
        inside an IRA forfeits it permanently under Rev. Rul. 2008-5.
        Scoped to one account, that distinction is invisible.
        """
        household = self.account(account_id).household_id
        acquisitions: list[Acquisition] = []
        for state in self.accounts.values():
            if state.household_id == household:
                acquisitions.extend(state.acquisitions())
        return acquisitions

    def household_disposals(self, account_id: str) -> list[Disposal]:
        """Every realised sale in the household's TAXABLE accounts.

        The buy-side wash screen: a purchase today, in any account, that
        would wash a loss one of these sales realised. Losses inside an
        IRA are nobody's deduction, so those accounts contribute none.
        """
        household = self.account(account_id).household_id
        disposals: list[Disposal] = []
        for state in self.accounts.values():
            if (
                state.household_id == household
                and not state.account.account_type.is_tax_advantaged
            ):
                disposals.extend(state.disposals())
        return disposals
