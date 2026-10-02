"""Tests for the compliance gate.

Every rejection must name the constraint that produced it and the
authority behind it. "The system said no" is not an answer anyone can
give a client or an examiner.
"""

from __future__ import annotations

from datetime import date

from meridian.compliance import (
    ComplianceContext,
    ComplianceGate,
    ConcentrationLimit,
    Constraint,
    MinimumCash,
    PreclearanceRequired,
    RestrictedSecurity,
    Severity,
    ShortTermGainLimit,
    WashSaleBlock,
)
from meridian.household import AccountType
from meridian.ledger import Buy, Deposit, LedgerEvent, fold
from meridian.model import Model, Sleeve
from meridian.money import Money, Price, Shares, Weight
from meridian.rebalance import RebalancePolicy, Side, generate_proposal
from meridian.taxlot import LotMethod, TaxLot, TaxRates, build_lots
from meridian.washsale import Acquisition, SubstituteMap

D = date(2026, 9, 8)

PRICES = {
    "VTI": Price("140.00"),
    "AAPL": Price("180.00"),
    "BND": Price("50.00"),
    "GLD": Price("150.00"),
}
CLASSIFICATION = {
    "VTI": "equity",
    "AAPL": "equity",
    "BND": "bond",
    "GLD": "alt",
}
MODEL = Model(
    "classic-60-40",
    1,
    {
        "equity": Sleeve(Weight("0.55"), Weight("0.05"), security="VTI"),
        "bond": Sleeve(Weight("0.35"), Weight("0.05"), security="BND"),
        "alt": Sleeve(Weight("0.10"), Weight("0.03"), security="GLD"),
    },
)

DRIFTED: list[LedgerEvent] = [
    Deposit(1, D, Money("100000.00")),
    Buy(2, D, "VTI", Shares("300"), Price("140.00")),  # 42,000
    Buy(3, D, "AAPL", Shares("100"), Price("180.00")),  # 18,000
    Buy(4, D, "BND", Shares("500"), Price("50.00")),  # 25,000
    Buy(5, D, "GLD", Shares("100"), Price("150.00")),  # 15,000
]


def context(
    events: list[LedgerEvent] | None = None,
    *,
    lots: dict[str, list[TaxLot]] | None = None,
    acquisitions: list[Acquisition] | None = None,
    substitutes: SubstituteMap | None = None,
    account_type: AccountType = AccountType.TAXABLE,
    policy: RebalancePolicy | None = None,
    lot_aware: bool = False,
    rates: TaxRates | None = None,
) -> ComplianceContext:
    stream = events or DRIFTED
    portfolio = fold(stream)
    proposal = generate_proposal(
        portfolio,
        PRICES,
        MODEL,
        CLASSIFICATION,
        on=D,
        policy=policy,
        lots=build_lots(stream) if lot_aware else None,
        rates=rates,
    )
    return ComplianceContext(
        proposal=proposal,
        portfolio=portfolio,
        prices=PRICES,
        on=D,
        account_id="taxable-1",
        account_type=account_type,
        lots=lots or {},
        acquisitions=acquisitions or [],
        substitutes=substitutes,
    )


def gate(*constraints: Constraint) -> ComplianceGate:
    return ComplianceGate(constraints=constraints)


# ============================================================
# THE BASELINE
# ============================================================


def test_a_clean_proposal_passes_untouched() -> None:
    result = gate().evaluate(context())
    assert result.is_clear
    assert len(result.passed) == len(context().proposal.trades)


def test_every_violation_names_its_constraint_and_authority() -> None:
    """'The system said no' is not an answer anyone can give a client or
    an examiner."""
    result = gate(
        RestrictedSecurity(
            constraint_id="ips-tobacco",
            tickers=frozenset({"GLD"}),
            reason="client prohibits precious metals",
            authority="IPS clause 4.2",
        )
    ).evaluate(context())

    violation = result.blocks[0]
    assert violation.constraint_id == "ips-tobacco"
    assert violation.authority == "IPS clause 4.2"
    assert "client prohibits" in violation.message


# ============================================================
# RESTRICTED SECURITIES
# ============================================================


def test_a_restricted_security_is_blocked_in_both_directions() -> None:
    result = gate(
        RestrictedSecurity(
            "wall", frozenset({"GLD"}), "material non-public information"
        )
    ).evaluate(context())

    assert not result.is_clear
    assert all(t.ticker != "GLD" for t in result.passed)
    assert any(t.ticker == "GLD" for t in result.blocked)


def test_sell_only_lets_a_client_exit_a_prohibited_position() -> None:
    """Forcing a client to hold something they have prohibited would be
    the opposite of the intent."""
    result = gate(
        RestrictedSecurity(
            "ips-exit",
            frozenset({"GLD"}),
            "client no longer wishes to hold gold",
            sell_only=True,
        )
    ).evaluate(context())

    gld_sells = [t for t in result.passed if t.ticker == "GLD" and t.side is Side.SELL]
    assert gld_sells
    assert result.is_clear


def test_an_unrelated_security_is_untouched() -> None:
    result = gate(
        RestrictedSecurity("wall", frozenset({"TSLA"}), "restricted list")
    ).evaluate(context())
    assert result.is_clear


# ============================================================
# CONCENTRATION
# ============================================================


def test_concentration_is_checked_on_the_post_trade_position() -> None:
    """The only check that means anything. A limit tested against the
    CURRENT position would happily approve the trade that breaches it.
    """
    concentrated: list[LedgerEvent] = [
        Deposit(1, D, Money("100000.00")),
        Buy(2, D, "AAPL", Shares("100"), Price("180.00")),  # 18,000
        Buy(3, D, "BND", Shares("500"), Price("50.00")),  # 25,000
    ]
    result = gate(
        ConcentrationLimit("conc-10", Weight("0.10"), authority="IPS clause 3.1")
    ).evaluate(context(concentrated))

    assert not result.is_clear
    assert any("above the 10.0% limit" in v.message for v in result.blocks)


def test_broad_market_funds_can_be_exempt() -> None:
    """A 60% position in a total market index is not a concentration
    risk in the sense the rule is about."""
    result = gate(
        ConcentrationLimit(
            "conc-10", Weight("0.10"), exempt=frozenset({"VTI", "BND", "GLD", "AAPL"})
        )
    ).evaluate(context())
    assert result.is_clear


# ============================================================
# MINIMUM CASH
# ============================================================


def test_a_cash_minimum_blocks_the_whole_proposal() -> None:
    """An account-level constraint is not about one ticker, so it stops
    everything rather than picking a trade to blame."""
    result = gate(
        MinimumCash("cash-2pct", Money("50000.00"), authority="IPS clause 6")
    ).evaluate(context())

    assert not result.is_clear
    assert result.passed == ()
    assert len(result.blocked) > 0


def test_a_satisfiable_cash_minimum_passes() -> None:
    result = gate(MinimumCash("cash-min", Money("0.00"))).evaluate(context())
    assert result.is_clear


# ============================================================
# SHORT-TERM GAINS — a warning, not a block
# ============================================================


def test_a_large_short_term_gain_warns_rather_than_blocks() -> None:
    """Realising short-term gain to close a genuine breach may well be
    the right call. An engine that silently refused would be
    substituting its judgment for the advisor's."""
    lots = {
        "GLD": [
            TaxLot("gld-1", "GLD", D.replace(month=8), Shares("100"), Money("5000.00"))
        ]
    }
    result = gate(
        ShortTermGainLimit("stcg", Money("100.00"), authority="tax policy")
    ).evaluate(context(lots=lots))

    assert result.warnings
    assert result.is_clear  # a warning does not stop the trade
    assert "SHORT-TERM" in result.warnings[0].message


def test_no_short_term_warning_inside_a_retirement_account() -> None:
    """There is no taxable event inside an IRA. Warning about one would
    be noise that trains people to ignore the warnings."""
    lots = {
        "GLD": [
            TaxLot("gld-1", "GLD", D.replace(month=8), Shares("100"), Money("5000.00"))
        ]
    }
    result = gate(ShortTermGainLimit("stcg", Money("100.00"))).evaluate(
        context(lots=lots, account_type=AccountType.ROTH_IRA)
    )
    assert not result.warnings


def test_a_long_term_lot_produces_no_warning() -> None:
    lots = {
        "GLD": [
            TaxLot("gld-1", "GLD", date(2020, 1, 1), Shares("100"), Money("5000.00"))
        ]
    }
    result = gate(ShortTermGainLimit("stcg", Money("100.00"))).evaluate(
        context(lots=lots)
    )
    assert not result.warnings


# ============================================================
# WASH SALES — severity escalates
# ============================================================

POLICY = SubstituteMap.symmetric(groups=[["GLD", "IAU"]])


def loss_lots() -> dict[str, list[TaxLot]]:
    """A GLD position sitting at a loss: bought at $200, now $150."""
    return {
        "GLD": [
            TaxLot("gld-1", "GLD", date(2024, 1, 1), Shares("100"), Money("20000.00"))
        ]
    }


def test_a_deferred_wash_sale_warns() -> None:
    result = gate(WashSaleBlock("wash")).evaluate(
        context(
            lots=loss_lots(),
            substitutes=POLICY,
            acquisitions=[
                Acquisition(
                    "a", "taxable-2", AccountType.TAXABLE, "GLD", D, Shares("100")
                )
            ],
        )
    )
    assert result.warnings
    assert result.is_clear
    assert "deferred" in result.warnings[0].message


def test_an_ira_wash_sale_blocks() -> None:
    """Severity escalates. A deferred loss comes back eventually; a
    forfeited one does not, and there is no version of that outcome the
    client wanted."""
    result = gate(WashSaleBlock("wash")).evaluate(
        context(
            lots=loss_lots(),
            substitutes=POLICY,
            acquisitions=[
                Acquisition(
                    "a", "roth-1", AccountType.ROTH_IRA, "GLD", D, Shares("100")
                )
            ],
        )
    )
    assert not result.is_clear
    assert "PERMANENTLY FORFEITED" in result.blocks[0].message
    assert result.blocks[0].authority == "Rev. Rul. 2008-5"


def test_a_sale_at_a_gain_is_not_a_wash_sale_concern() -> None:
    """Section 1091 disallows losses. Screening gains would block trades
    for no reason."""
    gain_lots = {
        "GLD": [
            TaxLot("gld-1", "GLD", date(2024, 1, 1), Shares("100"), Money("5000.00"))
        ]
    }
    result = gate(WashSaleBlock("wash")).evaluate(
        context(
            lots=gain_lots,
            substitutes=POLICY,
            acquisitions=[
                Acquisition(
                    "a", "roth-1", AccountType.ROTH_IRA, "GLD", D, Shares("100")
                )
            ],
        )
    )
    assert result.is_clear


def test_no_substitute_map_means_no_wash_screening() -> None:
    """The engine will not guess what is substantially identical."""
    result = gate(WashSaleBlock("wash")).evaluate(context(lots=loss_lots()))
    assert result.is_clear


# ============================================================
# PRECLEARANCE — Rule 204A-1
# ============================================================


def test_preclearance_blocks_until_approval_is_on_file() -> None:
    """Rule 204A-1 requires access persons to obtain approval before
    acquiring an IPO or limited offering, and the firm to keep records
    of those decisions."""
    result = gate(PreclearanceRequired("204A-1", frozenset({"BND"}))).evaluate(
        context()
    )

    assert not result.is_clear
    assert "preclearance" in result.blocks[0].message
    assert result.blocks[0].authority == "Advisers Act Rule 204A-1"


def test_a_granted_preclearance_lets_the_buy_through() -> None:
    result = gate(
        PreclearanceRequired("204A-1", frozenset({"BND"}), granted=frozenset({"BND"}))
    ).evaluate(context())
    assert result.is_clear


def test_preclearance_only_applies_to_acquisitions() -> None:
    """You do not need permission to stop owning something."""
    result = gate(PreclearanceRequired("204A-1", frozenset({"GLD"}))).evaluate(
        context()
    )
    assert result.is_clear


# ============================================================
# BLOCKING A SELL CAN BREAK THE FUNDING
# ============================================================


def test_blocking_a_sell_drops_the_buys_it_was_funding() -> None:
    """The subtlety. Sells fund buys — remove a sell and the buys may no
    longer be payable. A gate that returned an unfundable list would be
    handing the custodian orders that bounce.
    """
    ctx = context()
    assert any(t.side is Side.SELL for t in ctx.proposal.trades)

    # Block every sell.
    sells = {t.ticker for t in ctx.proposal.sells}
    result = gate(RestrictedSecurity("wall", frozenset(sells), "restricted")).evaluate(
        ctx
    )

    surviving_buys = [t for t in result.passed if t.side is Side.BUY]
    available = ctx.portfolio.cash
    spend = sum((t.consideration for t in surviving_buys), Money.zero())
    assert spend <= available


def test_the_dropped_buy_gets_its_own_reason() -> None:
    ctx = context()
    sells = {t.ticker for t in ctx.proposal.sells}
    result = gate(RestrictedSecurity("wall", frozenset(sells), "restricted")).evaluate(
        ctx
    )

    funding = [v for v in result.violations if v.constraint_id == "funding"]
    assert funding
    assert "after blocked sells were removed" in funding[0].message


def test_the_surviving_list_is_always_fundable() -> None:
    """The invariant the gate must never violate, whatever is blocked."""
    for blocked_ticker in ("VTI", "AAPL", "BND", "GLD"):
        ctx = context()
        result = gate(
            RestrictedSecurity("wall", frozenset({blocked_ticker}), "restricted")
        ).evaluate(ctx)

        raised = sum(
            (t.consideration for t in result.passed if t.side is Side.SELL),
            Money.zero(),
        )
        spent = sum(
            (t.consideration for t in result.passed if t.side is Side.BUY),
            Money.zero(),
        )
        assert spent <= ctx.portfolio.cash + raised


# ============================================================
# ORDERING AND COMBINATION
# ============================================================


def test_blocks_are_reported_before_warnings() -> None:
    result = gate(
        ShortTermGainLimit("stcg", Money("1.00")),
        RestrictedSecurity("wall", frozenset({"GLD"}), "restricted"),
    ).evaluate(context(lots=loss_lots()))

    severities = [v.severity for v in result.violations]
    if Severity.BLOCK in severities and Severity.WARN in severities:
        assert severities.index(Severity.BLOCK) < severities.index(Severity.WARN)


def test_violation_order_does_not_depend_on_constraint_registration_order() -> None:
    """A report whose row order depended on which constraint happened to
    be registered first could not be diffed between runs."""
    a = RestrictedSecurity("wall-a", frozenset({"GLD"}), "restricted")
    b = RestrictedSecurity("wall-b", frozenset({"BND"}), "restricted")

    forward = gate(a, b).evaluate(context())
    backward = gate(b, a).evaluate(context())

    assert [v.constraint_id for v in forward.violations] == [
        v.constraint_id for v in backward.violations
    ]


def test_several_constraints_combine() -> None:
    """Only the restricted-list rule objects on its own merits.

    The `funding` violation alongside it is the CASCADE, not a second
    opinion: blocking the GLD sell removed cash a buy was relying on, so
    that buy is dropped too and gets its own reason. Both belong in the
    report — an advisor seeing one trade vanish with no explanation is
    the failure this is designed to prevent.
    """
    result = gate(
        RestrictedSecurity("wall", frozenset({"GLD"}), "restricted"),
        ConcentrationLimit("conc", Weight("0.99")),
        MinimumCash("cash", Money("0.00")),
    ).evaluate(context())

    assert not result.is_clear

    reasons = {v.constraint_id for v in result.blocks}
    assert "wall" in reasons
    assert reasons <= {"wall", "funding"}  # nothing else objected
    assert "conc" not in reasons
    assert "cash" not in reasons


def test_an_empty_gate_passes_everything() -> None:
    """A gate with no constraints is not a broken gate — a client with
    no restrictions is a real client."""
    ctx = context()
    result = gate().evaluate(ctx)
    assert result.passed == ctx.proposal.trades


# ============================================================
# THE GATE JUDGES THE LOTS THE ORDER NAMES
# ============================================================


def test_the_gate_uses_the_named_lots_not_a_fifo_guess() -> None:
    """Two GLD lots: an old cheap one (long-term) and a recent one bought
    just below today's price (short-term, small gain). The alt sleeve is
    overweight and must sell some GLD.

    Under FIFO the sale would come from the long-term lot and no
    short-term warning is due. Under LIFO the order names the recent
    lot, and the gate must warn about THAT lot — not about what FIFO
    would have done.
    """
    history: list[LedgerEvent] = [
        Deposit(1, date(2024, 1, 2), Money("100000.00")),
        Buy(2, date(2024, 1, 2), "VTI", Shares("300"), Price("140.00")),
        Buy(3, date(2024, 1, 2), "AAPL", Shares("100"), Price("180.00")),
        Buy(4, date(2024, 1, 2), "BND", Shares("500"), Price("50.00")),
        Buy(5, date(2024, 1, 2), "GLD", Shares("50"), Price("100.00")),  # long
        Buy(6, date(2026, 8, 1), "GLD", Shares("50"), Price("120.00")),  # short
    ]
    limit = ShortTermGainLimit("stcg", Money("10.00"))

    recent_first = gate(limit).evaluate(
        context(
            history,
            lot_aware=True,
            policy=RebalancePolicy(lot_method=LotMethod.LIFO),
        )
    )
    assert recent_first.warnings
    assert recent_first.warnings[0].subject == "GLD"

    oldest_first = gate(limit).evaluate(
        context(
            history,
            lot_aware=True,
            policy=RebalancePolicy(lot_method=LotMethod.FIFO),
        )
    )
    assert not oldest_first.warnings
