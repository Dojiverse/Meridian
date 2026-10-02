"""The HTTP layer.

============================================================
WHAT THIS LAYER IS AND IS NOT
============================================================

It translates. Nothing here computes a dollar amount, decides a lot, or
judges a constraint — every one of those lives in the calculation core,
which holds no connections and reads no clock.

That separation is the whole architecture, and this module is where it
would erode first. The temptation in an endpoint is always to adjust one
number "just for display". Every such adjustment is a number the engine
cannot reproduce and the audit log cannot explain, so formatting is done
by asking the domain object for its own string, never by recomputing.

============================================================
EVERY DECISION IS AUDITED HERE, NOT IN THE UI
============================================================

Generating a proposal, blocking a trade, approving, rejecting — each
appends to the hash-chained log at the point the decision is made, on
the server. A UI that wrote its own audit entries would be a UI that
could forget to.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from meridian.allocate import AllocationError
from meridian.api.schemas import (
    ApproveRequest,
    AuditEntryOut,
    AuditOut,
    DriftOut,
    HarvestOut,
    HarvestScreenOut,
    ProposalOut,
    RejectRequest,
)
from meridian.api.store import AccountState, Store, StoredProposal
from meridian.audit import Action
from meridian.compliance import ComplianceContext, Severity
from meridian.drift import compute_drift, group_values
from meridian.harvest import add_harvest, harvest_trades
from meridian.ledger import LedgerError, market_values
from meridian.model import ModelError
from meridian.rebalance import generate_proposal
from meridian.taxlot import Disposal, LotError
from meridian.washsale import blackout_tickers, find_harvest_opportunities

STATIC = Path(__file__).parent / "static"


def create_app(store: Store) -> FastAPI:
    """Build the app around a store.

    The store is injected rather than global, so tests get a fresh world
    per test instead of sharing state through import order — which is
    the sort of coupling that produces a suite that passes alone and
    fails in CI.
    """
    app = FastAPI(
        title="Meridian",
        version="0.1.0",
        description=(
            "A tax-aware rebalancing engine. Every monetary value in "
            "this API is a JSON STRING — see meridian.api.schemas."
        ),
    )

    # ---- domain errors ------------------------------------------
    # The engine refuses things: a ledger that cannot be folded, an
    # allocation that would lose money, a model whose targets do not sum
    # to one. Those refusals are the point — they are correctness, not
    # crashes — and they carry messages written to be read.
    #
    # Letting them escape as a 500 with a stack trace throws that away
    # and tells the caller nothing. Mapping them to 422 keeps the
    # message and says clearly: the request was well-formed, the DATA
    # behind it will not support what you asked for.
    @app.exception_handler(LedgerError)
    @app.exception_handler(AllocationError)
    @app.exception_handler(ModelError)
    @app.exception_handler(LotError)
    def _domain_error(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "detail": str(exc),
                "kind": type(exc).__name__,
            },
        )

    def _now() -> datetime:
        return datetime.now(UTC)

    def _audit(actor: str, action: Action, subject: str, **payload: str) -> None:
        store.audit = store.audit.append(
            actor=actor,
            action=action,
            subject=subject,
            payload=payload,
            occurred_at=_now(),
        )

    # ---- pages -------------------------------------------------

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    # ---- accounts ----------------------------------------------

    @app.get("/api/accounts")
    def list_accounts() -> list[dict[str, str]]:
        return [
            {
                "account_id": state.account.account_id,
                "account_type": state.account.account_type.value,
                "custodian": state.account.custodian,
                "model_id": state.model.model_id,
                "model_version": str(state.model.version),
            }
            for state in store.accounts.values()
        ]

    @app.get("/api/accounts/{account_id}/drift", response_model=DriftOut)
    def account_drift(account_id: str) -> DriftOut:
        state = _require_account(account_id)
        portfolio = state.portfolio()
        values = group_values(
            market_values(portfolio, store.prices), state.classification
        )
        return DriftOut.of(account_id, compute_drift(values, state.model))

    @app.get("/api/accounts/{account_id}/positions")
    def account_positions(account_id: str) -> dict[str, object]:
        """Holdings with their lots. Money as strings, as everywhere."""
        state = _require_account(account_id)
        portfolio = state.portfolio()
        lots = state.lots()

        rows = []
        for ticker, quantity in sorted(portfolio.positions.items()):
            if quantity.is_zero:
                continue
            price = store.prices[ticker]
            value = quantity.value_at(price)
            basis = sum(
                (lot.cost_basis.amount for lot in lots.get(ticker, [])),
                start=value.amount * 0,
            )
            rows.append(
                {
                    "ticker": ticker,
                    "sleeve": state.classification.get(ticker, "unclassified"),
                    "quantity": str(quantity.quantity),
                    "price": str(price.amount),
                    "value": str(value.amount),
                    "value_display": str(value),
                    "cost_basis": str(basis),
                    "unrealized": str(value.amount - basis),
                    "lots": len(lots.get(ticker, [])),
                }
            )

        return {
            "account_id": account_id,
            "cash": str(portfolio.cash.amount),
            "cash_display": str(portfolio.cash),
            "positions": rows,
        }

    # ---- proposals ---------------------------------------------

    @app.post("/api/accounts/{account_id}/proposals", response_model=ProposalOut)
    def create_proposal(account_id: str) -> ProposalOut:
        """Generate a proposal and run it through the gate.

        Both in one call, and returned together. A UI that fetched
        trades from one endpoint and violations from another could
        render a trade list with the blocks still loading — and an
        advisor who approves what is on screen has approved something
        the gate rejected.
        """
        state = _require_account(account_id)
        portfolio = state.portfolio()
        lots = state.lots()
        acquisitions = store.household_acquisitions(account_id)
        disposals = store.household_disposals(account_id)
        blackout = _blackout(state, disposals)

        # Lot-aware sells only where lots have a tax consequence. Inside
        # a Roth every lot costs the same to sell — nothing — so the
        # proportional path is the honest one there.
        taxable = not state.account.account_type.is_tax_advantaged
        proposal = generate_proposal(
            portfolio,
            store.prices,
            state.model,
            state.classification,
            on=store.today,
            lots=lots if taxable else None,
            rates=state.rates if taxable else None,
            blackout=blackout,
        )

        # Harvest what the screen finds, in the same proposal, so there
        # is one gate run and one approval.
        if taxable and state.rates is not None and state.substitutes is not None:
            proposal = add_harvest(
                proposal,
                harvest_trades(
                    lots,
                    store.prices,
                    acquisitions,
                    state.substitutes,
                    state.rates,
                    state.classification,
                    on=store.today,
                    account_id=account_id,
                    account_type=state.account.account_type,
                    blackout=blackout,
                    exclude_lot_ids={
                        sel.lot_id for t in proposal.sells for sel in t.lots
                    },
                    exclude_tickers={t.ticker for t in proposal.buys},
                ),
            )

        context = ComplianceContext(
            proposal=proposal,
            portfolio=portfolio,
            prices=store.prices,
            on=store.today,
            account_id=account_id,
            account_type=state.account.account_type,
            lots=lots,
            # Household scope, not account scope — see
            # Store.household_acquisitions.
            acquisitions=acquisitions,
            substitutes=state.substitutes,
            disposals=disposals,
        )
        compliance = state.gate.evaluate(context)

        proposal_id = store.next_proposal_id()
        store.proposals[proposal_id] = StoredProposal(
            proposal_id=proposal_id,
            account_id=account_id,
            proposal=proposal,
            compliance=compliance,
        )

        _audit(
            "rebalancer",
            Action.PROPOSAL_GENERATED,
            proposal_id,
            account_id=account_id,
            model=f"{state.model.model_id} v{state.model.version}",
            trades=str(len(compliance.passed)),
            turnover=str(proposal.turnover.amount),
        )
        for violation in compliance.violations:
            if violation.severity is Severity.BLOCK:
                _audit(
                    "compliance-gate",
                    Action.CONSTRAINT_VIOLATED,
                    proposal_id,
                    constraint=violation.constraint_id,
                    # NOT `subject=` — the audit entry's own subject is
                    # the proposal, and reusing the name would collide
                    # with the positional argument above.
                    ticker=violation.subject,
                    authority=violation.authority,
                )

        return _render(store.proposals[proposal_id])

    @app.get("/api/proposals/{proposal_id}", response_model=ProposalOut)
    def get_proposal(proposal_id: str) -> ProposalOut:
        return _render(_require_proposal(proposal_id))

    @app.post("/api/proposals/{proposal_id}/approve", response_model=ProposalOut)
    def approve(proposal_id: str, request: ApproveRequest) -> ProposalOut:
        stored = _require_proposal(proposal_id)

        if stored.status != "pending":
            raise HTTPException(
                status_code=409,
                detail=(
                    f"{proposal_id} is already {stored.status}. A decision is "
                    "made once; changing it means generating a new proposal, "
                    "so the record of what was approved stays true."
                ),
            )
        if not stored.compliance.passed:
            # Approval executes the trades the gate let through. When it
            # let none through there is nothing to approve. Executing a
            # BLOCKED trade would be an override — a separate, separately
            # recorded decision — and this endpoint never does that.
            raise HTTPException(
                status_code=422,
                detail=(
                    "nothing in this proposal survived the compliance gate, so "
                    "there is nothing to approve. Overriding a block is a "
                    "separate, separately recorded decision — not an approval."
                ),
            )

        store.proposals[proposal_id] = StoredProposal(
            proposal_id=stored.proposal_id,
            account_id=stored.account_id,
            proposal=stored.proposal,
            compliance=stored.compliance,
            status="approved",
            decided_by=request.actor,
            note=request.note,
        )
        _audit(
            request.actor,
            Action.PROPOSAL_APPROVED,
            proposal_id,
            account_id=stored.account_id,
            # What was approved is the GATED list. The record says how
            # many trades the gate removed, so "approved" can never be
            # read as "approved everything that was proposed".
            executed=str(len(stored.compliance.passed)),
            removed=str(len(stored.compliance.blocked)),
            note=request.note,
        )
        return _render(store.proposals[proposal_id])

    # ---- harvest --------------------------------------------------

    @app.get("/api/accounts/{account_id}/harvest", response_model=HarvestScreenOut)
    def harvest_screen(account_id: str) -> HarvestScreenOut:
        """Lots at a loss, which may be taken, and what may not be bought.

        The screen, not the trade. Generating a proposal is what acts
        on it; this is for seeing why a harvest was or was not proposed.
        """
        state = _require_account(account_id)
        disposals = store.household_disposals(account_id)
        blackout = _blackout(state, disposals)

        if (
            state.account.account_type.is_tax_advantaged
            or state.rates is None
            or state.substitutes is None
        ):
            return HarvestScreenOut(
                account_id=account_id,
                opportunities=(),
                blackout=tuple(sorted(blackout)),
            )

        lots = state.lots()
        flat = [lot for ticker in sorted(lots) for lot in lots[ticker]]
        found = find_harvest_opportunities(
            flat,
            store.prices,
            store.household_acquisitions(account_id),
            state.substitutes,
            state.rates,
            on=store.today,
            account_id=account_id,
            account_type=state.account.account_type,
        )
        return HarvestScreenOut(
            account_id=account_id,
            opportunities=tuple(HarvestOut.of(h) for h in found),
            blackout=tuple(sorted(blackout)),
        )

    @app.post("/api/proposals/{proposal_id}/reject", response_model=ProposalOut)
    def reject(proposal_id: str, request: RejectRequest) -> ProposalOut:
        stored = _require_proposal(proposal_id)
        if stored.status != "pending":
            raise HTTPException(
                status_code=409, detail=f"{proposal_id} is already {stored.status}"
            )

        store.proposals[proposal_id] = StoredProposal(
            proposal_id=stored.proposal_id,
            account_id=stored.account_id,
            proposal=stored.proposal,
            compliance=stored.compliance,
            status="rejected",
            decided_by=request.actor,
            note=request.reason,
        )
        _audit(
            request.actor,
            Action.PROPOSAL_REJECTED,
            proposal_id,
            reason=request.reason,
        )
        return _render(store.proposals[proposal_id])

    # ---- audit --------------------------------------------------

    @app.get("/api/audit", response_model=AuditOut)
    def audit_log() -> AuditOut:
        """The log, plus a live verification of its own chain.

        Verification runs on every read rather than on a schedule. A
        tamper check nobody looks at is a tamper check that finds things
        months late.
        """
        breaks = store.audit.verify()
        return AuditOut(
            entries=tuple(
                AuditEntryOut(
                    seq=e.seq,
                    occurred_at=e.occurred_at,
                    actor=e.actor,
                    action=e.action.value,
                    subject=e.subject,
                    payload=dict(e.payload),
                    digest=e.digest,
                )
                for e in store.audit.entries
            ),
            head=store.audit.head,
            is_intact=not breaks,
            breaks=tuple(str(b) for b in breaks),
        )

    # ---- helpers -------------------------------------------------

    def _require_account(account_id: str) -> AccountState:
        try:
            return store.account(account_id)
        except KeyError:
            raise HTTPException(
                status_code=404, detail=f"no account {account_id!r}"
            ) from None

    def _blackout(state: AccountState, disposals: Sequence[Disposal]) -> frozenset[str]:
        """What may not be bought today — see washsale.blackout_tickers."""
        if state.substitutes is None:
            return frozenset()
        return blackout_tickers(disposals, state.substitutes, on=store.today)

    def _require_proposal(proposal_id: str) -> StoredProposal:
        stored = store.proposals.get(proposal_id)
        if stored is None:
            raise HTTPException(status_code=404, detail=f"no proposal {proposal_id!r}")
        return stored

    def _render(stored: StoredProposal) -> ProposalOut:
        return ProposalOut.of(
            stored.proposal_id,
            stored.account_id,
            stored.proposal,
            stored.compliance,
            stored.status,
        )

    return app
