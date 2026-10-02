"""Tests for tax-loss harvesting — the trades, not the screen.

The screen (`washsale.find_harvest_opportunities`) is tested in
test_washsale.py. These tests cover turning a finding into orders: a
whole-lot, lot-identified sell paired with a buy of the firm's
designated alternative, and every reason a harvest is NOT proposed.
"""

from __future__ import annotations

from datetime import date

import pytest

from meridian.harvest import HarvestPolicy, add_harvest, harvest_trades
from meridian.household import AccountType
from meridian.ledger import Buy, Deposit, LedgerEvent, fold
from meridian.model import Model, Sleeve
from meridian.money import Money, Price, Shares, Weight
from meridian.rebalance import RebalancePolicy, Side, Trade, generate_proposal
from meridian.taxlot import TaxLot, TaxRates, build_lots
from meridian.washsale import Acquisition, SubstituteMap

D = date(2026, 9, 8)
HIGH = TaxRates(
    short_term=Weight("0.37"), long_term=Weight("0.20"), niit=Weight("0.038")
)
PRICES = {
    "VTI": Price("140.00"),
    "SCHB": Price("28.00"),
    "GLD": Price("150.00"),
    "SLV": Price("27.00"),
    "BND": Price("50.00"),
}
CLASSIFICATION = {
    "VTI": "equity",
    "SCHB": "equity",
    "GLD": "alt",
    "SLV": "alt",
    "BND": "bond",
}
SUBSTITUTES = SubstituteMap.symmetric(
    groups=[["GLD", "IAU"]],
    alternatives={"GLD": ("SLV",), "VTI": ("SCHB",)},
)


def lots_with_a_gold_loss() -> dict[str, list[TaxLot]]:
    """GLD bought at $200, now $150: a $5,000 loss on a $20,000 lot."""
    return {
        "GLD": [
            TaxLot("gld-1", "GLD", date(2026, 7, 1), Shares("100"), Money("20000.00"))
        ],
        "VTI": [
            TaxLot("vti-1", "VTI", date(2024, 1, 2), Shares("300"), Money("30000.00"))
        ],
    }


def harvest(
    lots: dict[str, list[TaxLot]] | None = None,
    *,
    policy: HarvestPolicy | None = None,
    acquisitions: list[Acquisition] | None = None,
    account_type: AccountType = AccountType.TAXABLE,
    substitutes: SubstituteMap = SUBSTITUTES,
    classification: dict[str, str] | None = None,
    blackout: frozenset[str] = frozenset(),
    exclude_lot_ids: frozenset[str] = frozenset(),
    exclude_tickers: frozenset[str] = frozenset(),
) -> tuple[Trade, ...]:
    return harvest_trades(
        lots or lots_with_a_gold_loss(),
        PRICES,
        acquisitions or [],
        substitutes,
        HIGH,
        classification or CLASSIFICATION,
        on=D,
        account_id="taxable-1",
        account_type=account_type,
        policy=policy,
        blackout=blackout,
        exclude_lot_ids=exclude_lot_ids,
        exclude_tickers=exclude_tickers,
    )


# ============================================================
# THE PAIR
# ============================================================


def test_a_harvest_is_a_whole_lot_sell_and_a_replacement_buy() -> None:
    trades = harvest()
    assert len(trades) == 2
    sell, buy = trades

    assert sell.side is Side.SELL and sell.ticker == "GLD"
    assert sell.quantity == Shares("100")  # the whole lot
    assert sell.lots[0].lot_id == "gld-1"
    assert sell.realized_gain == Money("-5000.00")
    assert sell.harvest

    assert buy.side is Side.BUY and buy.ticker == "SLV"
    assert buy.sleeve == "alt"  # stays in the sleeve it replaces
    assert buy.consideration <= sell.consideration  # funded by the sell
    assert buy.harvest


def test_the_two_legs_share_a_pair_id() -> None:
    """One decision, two orders. The gate uses this to withdraw one leg
    when the other is blocked."""
    sell, buy = harvest()
    assert sell.pair == buy.pair == "gld-1"


def test_the_sell_is_lot_identified() -> None:
    sell = harvest()[0]
    assert [sel.lot_id for sel in sell.lots] == ["gld-1"]


# ============================================================
# WHAT STOPS A HARVEST
# ============================================================


def test_a_loss_below_the_dollar_minimum_is_not_harvested() -> None:
    assert harvest(policy=HarvestPolicy(minimum_loss=Money("6000.00"))) == ()


def test_a_loss_below_the_relative_minimum_is_not_harvested() -> None:
    """$5,000 is 25% of the lot. Demand 30% and nothing happens — a
    small dip on a big position is churn, not a harvest."""
    assert harvest(policy=HarvestPolicy(minimum_loss_fraction=Weight("0.30"))) == ()
    assert harvest(policy=HarvestPolicy(minimum_loss_fraction=Weight("0.20")))


def test_a_blocked_opportunity_is_not_proposed() -> None:
    """The screen says a Roth purchase would forfeit the loss. Proposing
    the sale anyway would be proposing a trade the firm already knows
    is bad."""
    trades = harvest(
        acquisitions=[
            Acquisition("a", "roth-1", AccountType.ROTH_IRA, "GLD", D, Shares("10"))
        ]
    )
    assert trades == ()


def test_the_lot_itself_does_not_block_its_own_harvest() -> None:
    """The purchase that created the lot is inside the window (bought
    two months ago) but it is the lot being sold, not a replacement."""
    recent = {
        "GLD": [
            TaxLot("gld-1", "GLD", date(2026, 8, 20), Shares("100"), Money("20000.00"))
        ]
    }
    own_purchase = Acquisition(
        "taxable-1:GLD-9",
        "taxable-1",
        AccountType.TAXABLE,
        "GLD",
        date(2026, 8, 20),
        Shares("100"),
    )
    trades = harvest(recent, acquisitions=[own_purchase])
    assert trades and trades[0].ticker == "GLD"


def test_no_designated_alternative_means_sell_only() -> None:
    bare = SubstituteMap.symmetric(groups=[["GLD", "IAU"]])
    trades = harvest(substitutes=bare)
    assert len(trades) == 1
    assert trades[0].side is Side.SELL


def test_an_alternative_in_the_blackout_is_not_bought() -> None:
    """SLV was itself sold at a loss within 30 days. Buying it would
    wash THAT loss."""
    trades = harvest(blackout=frozenset({"SLV"}))
    assert [t.side for t in trades] == [Side.SELL]


def test_an_unclassified_alternative_is_not_bought() -> None:
    """A replacement with no sleeve would be sold as untargeted at the
    next review — a harvest that round-trips into cash."""
    without_slv = {k: v for k, v in CLASSIFICATION.items() if k != "SLV"}
    trades = harvest(classification=without_slv)
    assert [t.side for t in trades] == [Side.SELL]


def test_nothing_is_harvested_inside_a_retirement_account() -> None:
    assert harvest(account_type=AccountType.ROTH_IRA) == ()


def test_lots_the_rebalance_already_sells_are_skipped() -> None:
    assert harvest(exclude_lot_ids=frozenset({"gld-1"})) == ()


def test_tickers_the_rebalance_buys_are_not_harvested() -> None:
    """Selling at a loss what the same proposal buys washes itself."""
    assert harvest(exclude_tickers=frozenset({"GLD"})) == ()


def test_policy_validation() -> None:
    with pytest.raises(ValueError, match="minimum_loss"):
        HarvestPolicy(minimum_loss=Money("-1"))
    with pytest.raises(ValueError, match="fraction"):
        HarvestPolicy(minimum_loss_fraction=Weight("2"))


# ============================================================
# COMBINING WITH A REBALANCE
# ============================================================

MODEL = Model(
    "m",
    1,
    {
        "equity": Sleeve(Weight("0.60"), Weight("0.05"), security="VTI"),
        "alt": Sleeve(Weight("0.10"), Weight("0.05"), security="GLD"),
        "bond": Sleeve(Weight("0.30"), Weight("0.05"), security="BND"),
    },
)

# Fully invested, every sleeve inside its band, one lot under water. The
# rebalance has nothing to do; only the harvest does.
HISTORY: list[LedgerEvent] = [
    Deposit(1, date(2024, 1, 2), Money("77000.00")),
    Buy(2, date(2024, 1, 2), "VTI", Shares("300"), Price("140.00")),  # 42,000
    Buy(3, date(2024, 1, 2), "BND", Shares("500"), Price("50.00")),  # 25,000
    Buy(4, date(2026, 7, 1), "GLD", Shares("50"), Price("200.00")),  # 10,000 -> 7,500
]


def test_add_harvest_records_the_trigger_and_the_cash() -> None:
    """An on-band portfolio with a harvestable loss: the rebalance has
    nothing to do, the harvest does, and the proposal says so."""
    portfolio = fold(HISTORY)
    lots = build_lots(HISTORY)
    proposal = generate_proposal(
        portfolio, PRICES, MODEL, CLASSIFICATION, on=D, lots=lots, rates=HIGH
    )
    assert proposal.is_empty

    trades = harvest_trades(
        lots,
        PRICES,
        [],
        SUBSTITUTES,
        HIGH,
        CLASSIFICATION,
        on=D,
        account_id="t",
        account_type=AccountType.TAXABLE,
        exclude_lot_ids={sel.lot_id for t in proposal.sells for sel in t.lots},
        exclude_tickers={t.ticker for t in proposal.buys},
    )
    combined = add_harvest(proposal, trades)

    assert combined.trigger == "harvest"
    assert combined.harvests
    assert not combined.rebalancing_trades
    assert combined.cash_after == proposal.cash_after + sum(
        (t.cash_effect for t in trades), Money.zero()
    )


def test_a_combined_proposal_is_applicable_to_the_ledger() -> None:
    """Harvest sells plus rebalance trades, in one ledger pass. The
    harvest buy is funded by its own sell, so nothing overdraws."""
    portfolio = fold(HISTORY)
    lots = build_lots(HISTORY)
    proposal = generate_proposal(
        portfolio,
        PRICES,
        MODEL,
        CLASSIFICATION,
        on=D,
        policy=RebalancePolicy(cash_trigger=Weight("0.01")),
        lots=lots,
        rates=HIGH,
    )
    assert proposal.is_empty  # no idle cash, nothing breached
    trades = harvest_trades(
        lots,
        PRICES,
        [],
        SUBSTITUTES,
        HIGH,
        CLASSIFICATION,
        on=D,
        account_id="t",
        account_type=AccountType.TAXABLE,
        exclude_lot_ids={sel.lot_id for t in proposal.sells for sel in t.lots},
        exclude_tickers={t.ticker for t in proposal.buys},
    )
    combined = add_harvest(proposal, trades)
    after = fold([*HISTORY, *combined.to_events(100)])  # raises on any violation
    assert after.total_value(PRICES) == portfolio.total_value(PRICES)
    assert "SLV" in after.positions
