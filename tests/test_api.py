"""Tests for the HTTP layer.

The headline test is `test_no_money_anywhere_is_a_json_number`. It walks
every response the API can produce and fails if a monetary value ever
comes back as a JSON number rather than a string.

That is a boundary test rather than a unit test, and it is the one that
stops the discipline eroding one endpoint at a time — which is exactly
how it would erode.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient

from meridian.api.app import create_app
from meridian.api.demo import build_demo_store
from meridian.api.store import Store
from meridian.money import Money


@pytest.fixture
def store() -> Store:
    """A fresh world per test.

    The store is injected rather than global, so tests do not share
    state through import order — the coupling that produces a suite
    which passes alone and fails in CI.
    """
    return build_demo_store()


@pytest.fixture
def client(store: Store) -> Iterator[TestClient]:
    with TestClient(create_app(store)) as c:
        yield c


# ============================================================
# THE BOUNDARY RULE
# ============================================================

MONETARY_KEYS = {
    "value",
    "total",
    "cash",
    "cash_before",
    "cash_after",
    "turnover",
    "consideration",
    "shortfall",
    "cost_basis",
    "unrealized",
    "price",
    "quantity",
    "actual",
    "target",
    "drift",
    "band_width",
    "proceeds",
    "gain",
    "realized_gain",
    "market_value",
    "unrealized_loss",
    "tax_benefit",
}
"""Every field carrying an exact decimal. Shares and weights are here
too: a quantity of 53.353 and a weight of 0.166667 are values the server
computed deliberately, and anything that can survive a round trip as a
string should."""


def _walk(node: Any, path: str = "") -> Iterator[tuple[str, str, Any]]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, f"{path}.{key}")
            if key in MONETARY_KEYS:
                yield (path, key, value)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            yield from _walk(item, f"{path}[{i}]")


def _every_response(client: TestClient) -> Iterator[tuple[str, Any]]:
    yield "/api/accounts", client.get("/api/accounts").json()
    for account in ("taxable-1", "roth-1"):
        yield (
            f"/api/accounts/{account}/drift",
            client.get(f"/api/accounts/{account}/drift").json(),
        )
        yield (
            f"/api/accounts/{account}/positions",
            client.get(f"/api/accounts/{account}/positions").json(),
        )
        yield (
            f"POST /api/accounts/{account}/proposals",
            client.post(f"/api/accounts/{account}/proposals").json(),
        )
        yield (
            f"/api/accounts/{account}/harvest",
            client.get(f"/api/accounts/{account}/harvest").json(),
        )
    yield "/api/audit", client.get("/api/audit").json()


def test_no_money_anywhere_is_a_json_number(client: TestClient) -> None:
    """The rule the whole layer exists to keep.

    JSON has one number type, and every JavaScript engine parses it into
    an IEEE-754 double — the same binary float that cannot represent
    0.10. A monetary value sent as a number is silently rounded the
    moment the browser calls JSON.parse. Nothing errors; the books just
    stop matching.
    """
    offenders = []
    for endpoint, body in _every_response(client):
        for path, key, value in _walk(body):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                offenders.append(f"{endpoint} {path}.{key} = {value!r}")

    assert not offenders, "monetary values returned as JSON numbers:\n" + "\n".join(
        offenders
    )


def test_a_string_amount_survives_a_javascript_round_trip() -> None:
    """Demonstrates the failure this rule prevents, rather than
    asserting it in the abstract."""
    # More significant digits than an IEEE-754 double can hold. A
    # double carries about 15-17; this has 20. Note that a SHORTER
    # amount would survive — Python and JavaScript both print the
    # shortest string that round-trips — which is exactly why the bug
    # is so easy to miss in testing and so damaging at scale.
    exact = Decimal("12345678901234567.89")

    # As a JSON number, the way JavaScript would receive it:
    as_number = json.loads(json.dumps(float(exact)))
    assert Decimal(str(as_number)) != exact  # silently changed

    # As a string:
    as_string = json.loads(json.dumps(str(exact)))
    assert Decimal(as_string) == exact  # intact


def test_amounts_parse_back_to_exact_decimals(client: TestClient) -> None:
    body = client.get("/api/accounts/taxable-1/drift").json()
    for sleeve in body["sleeves"]:
        assert Decimal(sleeve["value"]) == Decimal(sleeve["value"])
        assert isinstance(sleeve["value"], str)


# ============================================================
# DRIFT AND POSITIONS
# ============================================================


def test_drift_reports_the_model_it_measured_against(client: TestClient) -> None:
    body = client.get("/api/accounts/taxable-1/drift").json()
    assert body["model_id"] == "classic-60-40"
    assert body["model_version"] == 1
    assert body["needs_rebalancing"] is True


def test_drift_rows_are_worst_first(client: TestClient) -> None:
    body = client.get("/api/accounts/taxable-1/drift").json()
    drifts = [abs(Decimal(s["drift"])) for s in body["sleeves"]]
    assert drifts == sorted(drifts, reverse=True)


def test_positions_include_lot_counts(client: TestClient) -> None:
    body = client.get("/api/accounts/taxable-1/positions").json()
    tickers = {row["ticker"] for row in body["positions"]}
    assert tickers == {"VTI", "AAPL", "BND", "GLD"}
    assert all(row["lots"] >= 1 for row in body["positions"])


def test_an_unknown_account_is_404(client: TestClient) -> None:
    assert client.get("/api/accounts/nope/drift").status_code == 404


# ============================================================
# PROPOSALS AND THE GATE
# ============================================================


def test_a_proposal_returns_its_compliance_verdict_together(
    client: TestClient,
) -> None:
    """Trades and violations in ONE response.

    A UI that fetched them separately could render a trade list with the
    blocks still loading — and an advisor who approves what is on screen
    has approved something the gate rejected.
    """
    body = client.post("/api/accounts/taxable-1/proposals").json()
    assert "trades" in body
    assert "violations" in body
    assert "is_clear" in body


def test_the_wash_sale_screen_looks_across_the_household(
    client: TestClient,
) -> None:
    """The scope test, and the most valuable assertion in this file.

    taxable-1 sells GLD at a loss. roth-1 bought GLD inside the 61-day
    window. Screened per-account, that is invisible and the sale looks
    clean. Screened across the household, it is a PERMANENT FORFEITURE
    under Rev. Rul. 2008-5 — and the difference is real client money.

    An earlier version of the store scoped acquisitions to one account,
    and this exact demo reported a mild deferral warning instead. The
    engine was right; the layer feeding it was not.
    """
    body = client.post("/api/accounts/taxable-1/proposals").json()

    wash = [v for v in body["violations"] if v["constraint_id"] == "wash-1091"]
    assert wash, "the household wash sale was not detected"

    finding = wash[0]
    assert finding["severity"] == "block"
    assert "roth-1" in finding["message"]
    assert "PERMANENTLY FORFEITED" in finding["message"]
    assert finding["authority"] == "Rev. Rul. 2008-5"


def test_every_violation_carries_its_authority(client: TestClient) -> None:
    """'The system said no' is not an answer anyone can give a client or
    an examiner."""
    body = client.post("/api/accounts/taxable-1/proposals").json()
    for violation in body["violations"]:
        assert "constraint_id" in violation
        assert "authority" in violation


def test_a_proposal_can_be_fetched_again_unchanged(client: TestClient) -> None:
    created = client.post("/api/accounts/taxable-1/proposals").json()
    fetched = client.get(f"/api/proposals/{created['proposal_id']}").json()
    assert fetched == created


# ============================================================
# APPROVAL
# ============================================================


def _clear_proposal(client: TestClient) -> dict[str, Any]:
    """The Roth account has no blocking constraints, so its proposals
    are approvable."""
    body: dict[str, Any] = client.post("/api/accounts/roth-1/proposals").json()
    return body


def test_approval_records_who(client: TestClient) -> None:
    created = _clear_proposal(client)
    approved = client.post(
        f"/api/proposals/{created['proposal_id']}/approve",
        json={"actor": "a.advisor", "note": "reviewed"},
    ).json()
    assert approved["status"] == "approved"

    audit = client.get("/api/audit").json()
    approvals = [e for e in audit["entries"] if e["action"] == "proposal.approved"]
    assert approvals[-1]["actor"] == "a.advisor"


def test_an_approval_without_an_actor_is_refused(client: TestClient) -> None:
    """There is no default of 'system'. An approval that cannot name who
    gave it is not an approval."""
    created = _clear_proposal(client)
    response = client.post(
        f"/api/proposals/{created['proposal_id']}/approve", json={"actor": ""}
    )
    assert response.status_code == 422


def test_a_proposal_with_blocks_is_approved_as_gated(client: TestClient) -> None:
    """The gate removed the blocked trades and re-checked that the
    survivors are fundable. Approving the survivors is executing what
    the gate allowed — not an override — and the record says how many
    trades were removed."""
    created = client.post("/api/accounts/taxable-1/proposals").json()
    assert not created["is_clear"]
    assert created["is_approvable"]
    assert created["blocked"]

    response = client.post(
        f"/api/proposals/{created['proposal_id']}/approve",
        json={"actor": "a.advisor", "note": "as gated"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "approved"

    audit = client.get("/api/audit").json()
    entry = [e for e in audit["entries"] if e["action"] == "proposal.approved"][-1]
    assert entry["payload"]["executed"] == str(len(created["trades"]))
    assert entry["payload"]["removed"] == str(len(created["blocked"]))


def test_a_proposal_with_nothing_surviving_cannot_be_approved(store: Store) -> None:
    """When the gate let nothing through there is nothing to approve.
    Executing a blocked trade would be an override — a separate,
    separately recorded decision."""
    from meridian.compliance import ComplianceGate, MinimumCash

    # An account-level block stops every trade.
    store.accounts["taxable-1"].gate = ComplianceGate(
        (MinimumCash("ips-6", Money("1000000.00"), authority="IPS clause 6"),)
    )
    with TestClient(create_app(store)) as client:
        created = client.post("/api/accounts/taxable-1/proposals").json()
        assert not created["is_approvable"]
        assert created["trades"] == []

        response = client.post(
            f"/api/proposals/{created['proposal_id']}/approve",
            json={"actor": "a.advisor"},
        )
    assert response.status_code == 422
    assert "nothing to approve" in response.json()["detail"]


def test_a_decision_is_made_once(client: TestClient) -> None:
    """Changing it means generating a new proposal, so the record of
    what was approved stays true."""
    created = _clear_proposal(client)
    pid = created["proposal_id"]
    client.post(f"/api/proposals/{pid}/approve", json={"actor": "a.advisor"})

    again = client.post(f"/api/proposals/{pid}/approve", json={"actor": "b.advisor"})
    assert again.status_code == 409


def test_rejection_requires_a_reason(client: TestClient) -> None:
    """A rejection with no reason teaches the next reviewer nothing."""
    created = _clear_proposal(client)
    response = client.post(
        f"/api/proposals/{created['proposal_id']}/reject",
        json={"actor": "a.advisor", "reason": ""},
    )
    assert response.status_code == 422


def test_rejection_is_recorded_with_its_reason(client: TestClient) -> None:
    created = _clear_proposal(client)
    rejected = client.post(
        f"/api/proposals/{created['proposal_id']}/reject",
        json={"actor": "a.advisor", "reason": "waiting on a deposit"},
    ).json()
    assert rejected["status"] == "rejected"

    audit = client.get("/api/audit").json()
    entry = [e for e in audit["entries"] if e["action"] == "proposal.rejected"][-1]
    assert entry["payload"]["reason"] == "waiting on a deposit"


# ============================================================
# AUDIT
# ============================================================


def test_generating_a_proposal_is_audited(client: TestClient) -> None:
    assert client.get("/api/audit").json()["entries"] == []
    client.post("/api/accounts/taxable-1/proposals")

    audit = client.get("/api/audit").json()
    assert any(e["action"] == "proposal.generated" for e in audit["entries"])


def test_every_block_is_audited_by_the_gate_not_the_ui(client: TestClient) -> None:
    """A UI that wrote its own audit entries would be a UI that could
    forget to."""
    body = client.post("/api/accounts/taxable-1/proposals").json()
    blocks = [v for v in body["violations"] if v["severity"] == "block"]

    audit = client.get("/api/audit").json()
    recorded = [e for e in audit["entries"] if e["action"] == "compliance.violated"]
    assert len(recorded) == len(blocks)
    assert all(e["actor"] == "compliance-gate" for e in recorded)


def test_the_chain_verifies_on_every_read(client: TestClient) -> None:
    """A tamper check nobody looks at is a check that finds things
    months late."""
    client.post("/api/accounts/taxable-1/proposals")
    audit = client.get("/api/audit").json()
    assert audit["is_intact"] is True
    assert audit["breaks"] == []
    assert len(audit["head"]) == 64


def test_the_chain_survives_a_full_workflow(client: TestClient) -> None:
    client.post("/api/accounts/taxable-1/proposals")
    created = _clear_proposal(client)
    client.post(
        f"/api/proposals/{created['proposal_id']}/approve",
        json={"actor": "a.advisor", "note": "fine"},
    )
    assert client.get("/api/audit").json()["is_intact"] is True


def test_audit_payloads_are_all_strings(client: TestClient) -> None:
    """A hash computed over a float is a hash over whatever that
    platform rounded to."""
    client.post("/api/accounts/taxable-1/proposals")
    for entry in client.get("/api/audit").json()["entries"]:
        assert all(isinstance(v, str) for v in entry["payload"].values())


# ============================================================
# THE PAGE
# ============================================================


def test_the_interface_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Meridian" in response.text


def test_the_page_never_calls_number_on_money() -> None:
    """The one rule the front end must keep.

    Turning an amount into a JavaScript Number would undo, in the
    browser, the exactness the entire server was built to preserve. The
    single parseFloat in the file is on a WEIGHT for percentage display
    — a ratio, not money — and it is commented as such.
    """
    import re

    from meridian.api.app import STATIC

    page = (STATIC / "index.html").read_text(encoding="utf-8")

    # Strip comments before checking. The first version of this test
    # matched the file's own comment explaining the rule — a test that
    # fails on documentation is a test that gets deleted.
    code = re.sub(r"<!--.*?-->", "", page, flags=re.DOTALL)
    code = re.sub(r"^\s*//.*$", "", code, flags=re.MULTILINE)

    assert "Number(" not in code
    assert code.count("parseFloat") <= 1


# ============================================================
# DOMAIN ERRORS REACH THE CALLER INTACT
# ============================================================


def test_a_ledger_refusal_becomes_a_422_with_its_message(store: Store) -> None:
    """The engine's refusals are correctness, not crashes.

    A ledger that cannot be folded, an allocation that would lose money,
    a model whose targets do not sum to one — each raises with a message
    written to be read. Letting those escape as a 500 with a stack trace
    throws the message away and tells the caller nothing.
    """
    from datetime import date

    from meridian.ledger import Buy
    from meridian.money import Price, Shares

    # A purchase the account cannot fund.
    store.accounts["taxable-1"].events.append(
        Buy(99, date(2026, 9, 5), "BND", Shares("400"), Price("50.00"))
    )

    with TestClient(create_app(store)) as client:
        response = client.get("/api/accounts/taxable-1/drift")

    assert response.status_code == 422
    body = response.json()
    assert body["kind"] == "LedgerError"
    assert "only $15,000.00 is available" in body["detail"]


def test_the_page_reflects_engine_state_rather_than_its_own(
    store: Store,
) -> None:
    """The page has no numbers of its own.

    Reprice a holding in the engine and every figure the API serves
    moves with it — total, weights, drift, and the resulting trades.
    `index.html` is byte-identical across both calls.
    """
    from meridian.money import Price

    with TestClient(create_app(store)) as client:
        before = client.get("/api/accounts/taxable-1/drift").json()

    store.prices["GLD"] = Price("400.00")

    with TestClient(create_app(store)) as client:
        after = client.get("/api/accounts/taxable-1/drift").json()

    assert before["total"] == "100000.00"
    assert after["total"] == "125000.00"

    alt_before = next(s for s in before["sleeves"] if s["sleeve"] == "alt")
    alt_after = next(s for s in after["sleeves"] if s["sleeve"] == "alt")
    assert alt_before["drift_display"] == "+5.0%"
    assert alt_after["drift_display"] == "+22.0%"


# ============================================================
# LOTS ON THE WIRE
# ============================================================


def test_taxable_sells_name_their_lots(client: TestClient) -> None:
    """A sell in a taxable account was sized lot by lot, and the order
    says which — the identification Treas. Reg. 1.1012-1(c) asks for,
    present on the wire rather than reconstructed later."""
    body = client.post("/api/accounts/taxable-1/proposals").json()
    sells = [t for t in body["trades"] + body["blocked"] if t["side"] == "sell"]
    assert sells
    for trade in sells:
        assert trade["lots"], f"{trade['ticker']} sell has no lot identification"
        assert isinstance(trade["realized_gain"], str)
        quantities = sum(Decimal(sel["quantity"]) for sel in trade["lots"])
        assert quantities == Decimal(trade["quantity"])


def test_roth_sells_carry_no_lots_and_no_gain(client: TestClient) -> None:
    """Inside a Roth no lot is cheaper than another, so sells are sized
    proportionally and the gain is null — not zero. An unknown gain is
    not a gain of nothing."""
    body = client.post("/api/accounts/roth-1/proposals").json()
    sells = [t for t in body["trades"] + body["blocked"] if t["side"] == "sell"]
    assert sells
    for trade in sells:
        assert trade["lots"] == []
        assert trade["realized_gain"] is None


# ============================================================
# THE HARVEST SCREEN
# ============================================================


def test_the_harvest_screen_shows_the_blocked_gold_loss(client: TestClient) -> None:
    """The demo's GLD lot is under water, and the Roth bought GLD inside
    the window. The screen shows the opportunity AND why it cannot be
    taken, before anyone generates a proposal."""
    body = client.get("/api/accounts/taxable-1/harvest").json()
    gold = [h for h in body["opportunities"] if h["ticker"] == "GLD"]
    assert gold
    assert gold[0]["is_blocked"]
    assert "PERMANENTLY FORFEITED" in gold[0]["block_reason"]
    assert gold[0]["alternatives"] == ["SLV"]


def test_a_blocked_harvest_is_not_proposed(client: TestClient) -> None:
    """The proposal contains no harvest trades for GLD: the screen said
    it is blocked, and proposing it would be proposing a trade the firm
    already knows is bad."""
    body = client.post("/api/accounts/taxable-1/proposals").json()
    harvests = [t for t in body["trades"] + body["blocked"] if t["harvest"]]
    assert harvests == []


def test_the_roth_has_no_harvest_screen(client: TestClient) -> None:
    body = client.get("/api/accounts/roth-1/harvest").json()
    assert body["opportunities"] == []


# ============================================================
# ONE WORLD PER VISITOR
# ============================================================


def test_each_visitor_gets_their_own_world() -> None:
    """Two browsers, two cookies, two stores. One visitor's approval is
    invisible to the other, and the first to click does not decide the
    demo for everyone after them."""
    app = create_app(build_demo_store)
    with TestClient(app) as alice, TestClient(app) as bob:
        a = alice.post("/api/accounts/roth-1/proposals").json()
        alice.post(
            f"/api/proposals/{a['proposal_id']}/approve", json={"actor": "alice"}
        )
        assert alice.get("/api/audit").json()["entries"]
        assert bob.get("/api/audit").json()["entries"] == []
        assert bob.get(f"/api/proposals/{a['proposal_id']}").status_code == 404


def test_the_session_cookie_is_set_once_and_reused() -> None:
    app = create_app(build_demo_store)
    with TestClient(app) as client:
        first = client.get("/api/accounts")
        assert "meridian_demo" in first.cookies
        client.post("/api/accounts/roth-1/proposals")
        again = client.get("/api/audit").json()
        assert len(again["entries"]) == 1  # same world, state kept


def test_reset_gives_the_visitor_a_fresh_world() -> None:
    app = create_app(build_demo_store)
    with TestClient(app) as client:
        client.post("/api/accounts/roth-1/proposals")
        assert client.get("/api/audit").json()["entries"]
        assert client.post("/api/reset").status_code == 200
        assert client.get("/api/audit").json()["entries"] == []


def test_reset_is_refused_on_a_single_shared_world(client: TestClient) -> None:
    assert client.post("/api/reset").status_code == 409


def test_the_app_can_be_mounted_under_a_prefix() -> None:
    """Hosting rewrites /blotter/** to the service with the prefix
    intact. The page uses relative URLs, so it only needs the trailing
    slash, which the redirect guarantees."""
    app = create_app(build_demo_store, root_path="/blotter")
    with TestClient(app) as client:
        bare = client.get("/blotter", follow_redirects=False)
        assert bare.status_code == 307
        assert bare.headers["location"].endswith("/blotter/")
        assert client.get("/blotter/").status_code == 200
        assert client.get("/blotter/api/accounts").status_code == 200
        assert client.get("/api/accounts").status_code == 404


def test_the_page_uses_relative_urls() -> None:
    from meridian.api.app import STATIC

    page = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "fetch(url(path)" in page
    assert "fetch('/api" not in page and 'fetch("/api' not in page
