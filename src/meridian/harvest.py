"""Tax-loss harvesting — acting on what the screen finds.

============================================================
A FINDING IS NOT A TRADE
============================================================

`washsale.find_harvest_opportunities` answers "which lots are sitting at
a loss, and may they be sold?" This module answers the next question:
WHAT ORDERS DO I PLACE? — and produces trades that go through the same
compliance gate, the same approval, and the same ledger as any
rebalance.

Two orders per harvest:

  SELL the whole lot, identified by lot id. Whole, because a lot held
  at a loss has nothing to recommend keeping part of it, and a partial
  harvest leaves a smaller loss to harvest next month at another
  spread. Identified, so the ledger disposes of exactly this lot.

  BUY the firm's designated ALTERNATIVE with the proceeds — correlated
  enough to keep the client's exposure, different enough that the firm
  will defend it as not substantially identical. Without a designated
  alternative the proceeds stay in cash and the trade's reason says so:
  the client is out of the market for 31 days, and that is a cost the
  advisor should see written down.

============================================================
WHAT STOPS A HARVEST
============================================================

  The screen says it is BLOCKED — a purchase in the window, anywhere in
  the household — so it is not proposed. The gate would block it too;
  proposing it would be proposing a trade the firm already knows is
  bad, and the screen remains the place to see why.

  The loss is below the policy minimum. Harvesting is not free: two
  spreads, two commissions, an alternative that tracks imperfectly. A
  $40 loss is not worth it.

  The rebalance already sells the lot, or buys the ticker. A harvest
  that double-sells a lot produces an order the ledger refuses; one
  that sells what the rebalance buys washes itself.

  The alternative is in the BLACKOUT — sold at a loss itself within 30
  days — or is not classified into a sleeve. An unclassified
  replacement would be sold as untargeted at the next review, which is
  how a harvest quietly round-trips into cash.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from typing import Final

from meridian.household import AccountType
from meridian.money import Money, Price, Shares, Weight
from meridian.rebalance import LotSelection, Proposal, Side, Trade
from meridian.taxlot import TaxLot, TaxRates
from meridian.washsale import (
    Acquisition,
    HarvestOpportunity,
    SubstituteMap,
    find_harvest_opportunities,
)

__all__ = ["HarvestPolicy", "add_harvest", "harvest_trades"]

DEFAULT_MINIMUM_LOSS: Final = Money("500.00")
DEFAULT_MINIMUM_LOSS_FRACTION: Final = Weight("0.05")


@dataclass(frozen=True, slots=True)
class HarvestPolicy:
    """The firm's harvesting knobs."""

    minimum_loss: Money = DEFAULT_MINIMUM_LOSS
    """Smallest unrealised loss worth two orders and a tracking gap."""

    minimum_loss_fraction: Weight = DEFAULT_MINIMUM_LOSS_FRACTION
    """The loss must also be at least this share of the lot's basis.

    A dollar floor alone harvests a 2% dip on a large position: a whole
    $85,000 lot sold for a $3,000 loss, two spreads paid, and the
    replacement tracking imperfectly for a month. Requiring the loss to
    be material RELATIVE to the lot stops that churn. Both tests must
    pass.
    """

    replace: bool = True
    """Buy the designated alternative with the proceeds. False leaves
    the cash uninvested, which some firms prefer in a drawdown."""

    share_increment: Decimal = Decimal("0.001")

    def __post_init__(self) -> None:
        if self.minimum_loss.is_negative:
            raise ValueError("minimum_loss cannot be negative")
        if not 0 <= self.minimum_loss_fraction.value <= 1:
            raise ValueError("minimum_loss_fraction must be between 0 and 1")
        if self.share_increment <= 0:
            raise ValueError("share_increment must be positive")

    def is_material(self, loss: Money, basis: Money) -> bool:
        """Both floors: absolute, and relative to what was paid."""
        if abs(loss) < self.minimum_loss:
            return False
        return abs(loss).amount >= basis.amount * self.minimum_loss_fraction.value


def harvest_trades(
    lots: Mapping[str, Sequence[TaxLot]],
    prices: Mapping[str, Price],
    acquisitions: Sequence[Acquisition],
    substitutes: SubstituteMap,
    rates: TaxRates,
    classification: Mapping[str, str],
    *,
    on: date,
    account_id: str,
    account_type: AccountType,
    policy: HarvestPolicy | None = None,
    blackout: Collection[str] = frozenset(),
    exclude_lot_ids: Collection[str] = frozenset(),
    exclude_tickers: Collection[str] = frozenset(),
) -> tuple[Trade, ...]:
    """Orders that realise every harvestable loss the screen finds.

    Args:
        exclude_lot_ids: Lots a rebalance in the same proposal already
            sells. Selling them twice is an order the ledger refuses.
        exclude_tickers: Tickers a rebalance in the same proposal buys.
            Harvesting a loss in something the proposal also buys would
            wash itself.
        blackout: Tickers sold at a loss within the last 30 days. Never
            chosen as a replacement.

    Returns trades in a deterministic order: largest tax benefit first,
    each sell followed by its replacement buy.
    """
    policy = policy or HarvestPolicy()
    if account_type.is_tax_advantaged:
        return ()

    flat = [lot for ticker in sorted(lots) for lot in lots[ticker]]
    found = find_harvest_opportunities(
        flat,
        prices,
        acquisitions,
        substitutes,
        rates,
        on=on,
        account_id=account_id,
        account_type=account_type,
        minimum_loss=policy.minimum_loss,
    )

    trades: list[Trade] = []
    for opportunity in found:
        if opportunity.is_blocked:
            continue
        lot = opportunity.lot
        if lot.lot_id in exclude_lot_ids or lot.ticker in exclude_tickers:
            continue
        if not policy.is_material(opportunity.unrealized_loss, lot.cost_basis):
            continue
        sleeve = classification.get(lot.ticker)
        if sleeve is None:
            continue

        price = prices[lot.ticker]
        sell = _sell_whole_lot(opportunity, price, sleeve, on)
        trades.append(sell)

        if policy.replace:
            buy = _replacement(
                opportunity,
                sell.consideration,
                prices,
                substitutes,
                classification,
                sleeve,
                policy,
                blackout,
                exclude_tickers,
            )
            if buy is not None:
                trades.append(buy)

    return tuple(trades)


def _sell_whole_lot(
    opportunity: HarvestOpportunity, price: Price, sleeve: str, on: date
) -> Trade:
    lot = opportunity.lot
    proceeds = lot.quantity.value_at(price)
    selection = LotSelection(
        lot_id=lot.lot_id,
        quantity=lot.quantity,
        acquired=lot.acquired,
        proceeds=proceeds,
        cost_basis=lot.cost_basis,
        period=opportunity.period,
    )
    return Trade(
        ticker=lot.ticker,
        side=Side.SELL,
        quantity=lot.quantity,
        price=price,
        sleeve=sleeve,
        reason=(
            f"harvest: lot {lot.lot_id} at a {opportunity.period.value}-term loss "
            f"of {abs(opportunity.unrealized_loss)}, worth {opportunity.tax_benefit} "
            "in tax"
        ),
        lots=(selection,),
        harvest=True,
        pair=lot.lot_id,
    )


def _replacement(
    opportunity: HarvestOpportunity,
    proceeds: Money,
    prices: Mapping[str, Price],
    substitutes: SubstituteMap,
    classification: Mapping[str, str],
    sleeve: str,
    policy: HarvestPolicy,
    blackout: Collection[str],
    exclude_tickers: Collection[str],
) -> Trade | None:
    """The buy that keeps the client invested, or None with the reason
    recorded on the sell's side by its absence being visible."""
    sold = opportunity.lot.ticker
    for candidate in substitutes.alternatives_for(sold):
        if candidate in blackout or candidate in exclude_tickers:
            continue
        if substitutes.are_identical(sold, candidate):
            # A policy error: the firm listed something it also calls
            # identical. Refuse rather than buy it.
            continue
        if candidate not in prices or classification.get(candidate) != sleeve:
            continue
        price = prices[candidate]
        quantity = Shares(proceeds.amount / price.amount).round_to(
            policy.share_increment
        )
        if quantity.is_zero:
            continue
        return Trade(
            ticker=candidate,
            side=Side.BUY,
            quantity=quantity,
            price=price,
            sleeve=sleeve,
            reason=(
                f"harvest: replaces {sold} exposure with the firm's designated "
                f"alternative while the 30-day window runs"
            ),
            harvest=True,
            pair=opportunity.lot.lot_id,
        )
    return None


def add_harvest(proposal: Proposal, trades: Sequence[Trade]) -> Proposal:
    """Attach harvest trades to a rebalance proposal.

    The result is one proposal: one gate run, one approval, one set of
    ledger events, sells first. The trigger records 'harvest' only when
    the rebalance itself had nothing to do, so a review can say whether
    it acted for tracking or for tax.
    """
    if not trades:
        return proposal
    cash_after = proposal.cash_after + sum(
        (t.cash_effect for t in trades), Money.zero()
    )
    trigger = "harvest" if proposal.is_empty else proposal.trigger
    return replace(
        proposal,
        trades=(*proposal.trades, *trades),
        cash_after=cash_after,
        trigger=trigger,
    )
