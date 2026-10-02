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

============================================================
ONE WORLD PER VISITOR
============================================================

The demo store is in memory. Served publicly with a single store, every
visitor would see every other visitor's approvals, and the first person
to click "approve" would decide the demo for everyone after them.

So `create_app` accepts either a Store (one fixed world, which is what
the tests want) or a factory that builds one. Given a factory, each
browser gets its own world, identified by a cookie, built on first
visit and dropped when it has been idle longest and the cap is reached.
A visitor can also start over with POST /api/reset. Nothing about the
engine changes; only who is looking at which copy of it.
"""

from __future__ import annotations

import secrets
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

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

SESSION_COOKIE = "__session"
"""Firebase Hosting forwards exactly one cookie to a Cloud Run backend,
and it has to be called __session; every other cookie is stripped on
the way through. Any other name works on the service's own URL and
silently fails behind the hosting rewrite, with every request arriving
as a brand-new visitor."""
MAX_SESSIONS = 200
"""How many visitor worlds to keep in memory before dropping the one
idle longest. Each is a few kilobytes of events; two hundred is plenty
for a portfolio page and bounded enough for a tiny container."""

StoreFactory = Callable[[], Store]


def current_store(request: Request) -> Store:
    """The world this request is looking at, placed by the middleware."""
    world: Store = request.state.store
    return world


World = Annotated[Store, Depends(current_store)]
"""Endpoint parameter type: the visitor's store. Module-level so FastAPI
can resolve the annotation under `from __future__ import annotations`."""


def create_app(store: Store | StoreFactory, *, root_path: str = "") -> FastAPI:
    """Build the app around a store, or around a factory of stores.

    Args:
        store: A Store for one fixed world (tests do this, so each test
            gets a fresh world injected rather than shared through
            import order), or a zero-argument callable that builds a
            Store, in which case every visitor gets their own.
        root_path: Mount the whole app under this prefix, e.g. "/blotter",
            for hosting behind a path-based rewrite. The page uses
            relative URLs so it works at either place.
    """
    fixed: Store | None = store if isinstance(store, Store) else None
    factory: StoreFactory | None = None if fixed is not None else store  # type: ignore[assignment]
    sessions: OrderedDict[str, Store] = OrderedDict()

    app = FastAPI(
        title="Meridian",
        version="0.1.0",
        root_path=root_path,
        description=(
            "A tax-aware rebalancing engine. Every monetary value in "
            "this API is a JSON STRING — see meridian.api.schemas."
        ),
    )

    # ---- which world is this request looking at? ----------------

    def _new_session() -> tuple[str, Store]:
        assert factory is not None
        session_id = secrets.token_urlsafe(16)
        sessions[session_id] = factory()
        while len(sessions) > MAX_SESSIONS:
            sessions.popitem(last=False)
        return session_id, sessions[session_id]

    @app.middleware("http")
    async def _attach_store(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if fixed is not None:
            request.state.store = fixed
            return await call_next(request)

        session_id = request.cookies.get(SESSION_COOKIE)
        is_new = session_id is None or session_id not in sessions
        if is_new:
            session_id, world = _new_session()
        else:
            assert session_id is not None
            sessions.move_to_end(session_id)
            world = sessions[session_id]
        request.state.store = world
        request.state.session_id = session_id

        response = await call_next(request)
        if is_new:
            response.set_cookie(
                SESSION_COOKIE,
                session_id,
                httponly=True,
                samesite="lax",
                max_age=60 * 60 * 24 * 30,
            )
        return response

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

    def _audit(
        store: Store, actor: str, action: Action, subject: str, **payload: str
    ) -> None:
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

    # ---- the visitor's world -----------------------------------

    @app.post("/api/reset")
    def reset(request: Request) -> dict[str, str]:
        """Start this visitor over with a fresh demo world.

        Only meaningful when each visitor has their own; with one fixed
        store there is nothing per-visitor to reset.
        """
        if factory is None:
            raise HTTPException(
                status_code=409, detail="this instance serves one shared world"
            )
        session_id: str = request.state.session_id
        sessions[session_id] = factory()
        return {"status": "reset"}

    # ---- accounts ----------------------------------------------

    @app.get("/api/accounts")
    def list_accounts(store: World) -> list[dict[str, str]]:
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
    def account_drift(account_id: str, store: World) -> DriftOut:
        state = _require_account(store, account_id)
        portfolio = state.portfolio()
        values = group_values(
            market_values(portfolio, store.prices), state.classification
        )
        return DriftOut.of(account_id, compute_drift(values, state.model))

    @app.get("/api/accounts/{account_id}/positions")
    def account_positions(account_id: str, store: World) -> dict[str, object]:
        """Holdings with their lots. Money as strings, as everywhere."""
        state = _require_account(store, account_id)
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
    def create_proposal(account_id: str, store: World) -> ProposalOut:
        """Generate a proposal and run it through the gate.

        Both in one call, and returned together. A UI that fetched
        trades from one endpoint and violations from another could
        render a trade list with the blocks still loading — and an
        advisor who approves what is on screen has approved something
        the gate rejected.
        """
        state = _require_account(store, account_id)
        portfolio = state.portfolio()
        lots = state.lots()
        acquisitions = store.household_acquisitions(account_id)
        disposals = store.household_disposals(account_id)
        blackout = _blackout(store, state, disposals)

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
            store,
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
                    store,
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
    def get_proposal(proposal_id: str, store: World) -> ProposalOut:
        return _render(_require_proposal(store, proposal_id))

    @app.post("/api/proposals/{proposal_id}/approve", response_model=ProposalOut)
    def approve(
        proposal_id: str,
        request: ApproveRequest,
        store: World,
    ) -> ProposalOut:
        stored = _require_proposal(store, proposal_id)

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
            store,
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

    @app.post("/api/proposals/{proposal_id}/reject", response_model=ProposalOut)
    def reject(
        proposal_id: str,
        request: RejectRequest,
        store: World,
    ) -> ProposalOut:
        stored = _require_proposal(store, proposal_id)
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
            store,
            request.actor,
            Action.PROPOSAL_REJECTED,
            proposal_id,
            reason=request.reason,
        )
        return _render(store.proposals[proposal_id])

    # ---- harvest --------------------------------------------------

    @app.get("/api/accounts/{account_id}/harvest", response_model=HarvestScreenOut)
    def harvest_screen(account_id: str, store: World) -> HarvestScreenOut:
        """Lots at a loss, which may be taken, and what may not be bought.

        The screen, not the trade. Generating a proposal is what acts
        on it; this is for seeing why a harvest was or was not proposed.
        """
        state = _require_account(store, account_id)
        disposals = store.household_disposals(account_id)
        blackout = _blackout(store, state, disposals)

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

    # ---- audit --------------------------------------------------

    @app.get("/api/audit", response_model=AuditOut)
    def audit_log(store: World) -> AuditOut:
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

    def _blackout(
        store: Store, state: AccountState, disposals: Sequence[Disposal]
    ) -> frozenset[str]:
        """What may not be bought today — see washsale.blackout_tickers."""
        if state.substitutes is None:
            return frozenset()
        return blackout_tickers(disposals, state.substitutes, on=store.today)

    def _require_account(store: Store, account_id: str) -> AccountState:
        try:
            return store.account(account_id)
        except KeyError:
            raise HTTPException(
                status_code=404, detail=f"no account {account_id!r}"
            ) from None

    def _require_proposal(store: Store, proposal_id: str) -> StoredProposal:
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

    if not root_path:
        return app

    # ---- mounted under a prefix ----------------------------------
    # Hosting rewrites a path such as /blotter/** to this service with
    # the prefix intact, so the app has to answer there. Mounting keeps
    # every route above unchanged; the page's relative URLs resolve
    # against /blotter/ as long as the trailing slash is present, which
    # the redirect guarantees.
    outer = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    prefix = "/" + root_path.strip("/")

    @outer.get(prefix, include_in_schema=False)
    def _to_slash() -> RedirectResponse:
        return RedirectResponse(prefix + "/", status_code=307)

    outer.mount(prefix, app)
    return outer
