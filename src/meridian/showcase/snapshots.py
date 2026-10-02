"""Standing on a past date — the history rendered as JSON.

============================================================
EVERY AMOUNT IS A STRING, STILL
============================================================

The same rule as the API, for the same reason: this file is read by a
browser, and a browser turns JSON numbers into IEEE-754 doubles. So
every monetary value, share quantity, weight and return is a string,
and each comes with a `_display` sibling the server already formatted.

The one concession to charting is `*_cents` and `*_bps` fields in the
daily series: whole cents and whole basis points as INTEGERS, which a
double represents exactly up to 2^53. A chart needs numbers; giving it
integers is how it gets them without ever parsing a decimal string.

============================================================
WHAT A SNAPSHOT IS
============================================================

The complete state of the household as it stood at the close of one
day, derived by replaying the event stream up to that day and nothing
after it — `ledger.fold` over a prefix, `taxlot.build_lots` over the
same prefix, the drift report against the model that was live, the
performance from inception to that date, the harvest screen against
the acquisitions known by then, and the audit chain's head as it was.

Nothing is stored and looked up. Every snapshot is recomputed, which is
the property being demonstrated.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal
from pathlib import Path

from meridian.audit import AuditEntry, AuditLog
from meridian.compliance import ComplianceResult, Violation
from meridian.drift import DriftReport, compute_drift, group_values
from meridian.ledger import (
    Buy,
    Deposit,
    Dividend,
    Fee,
    LedgerEvent,
    Portfolio,
    Sell,
    Withdrawal,
    market_values,
)
from meridian.money import Money, Price, Weight
from meridian.performance import (
    ExternalFlow,
    FlowKind,
    PerformanceError,
    Valuation,
    compute_performance,
    money_weighted_return,
)
from meridian.rebalance import LotSelection, Trade, Unplaced
from meridian.showcase.history import AccountHistory, History, Review, event_dates
from meridian.taxlot import Disposal, LotMethod, TaxLot, dispose
from meridian.washsale import (
    HarvestOpportunity,
    WashSale,
    find_harvest_opportunities,
    find_wash_sales,
)

__all__ = ["render", "snapshot_dates", "write"]

JSON = dict[str, object]

MIN_HARVEST = Money("250.00")
MIN_DAYS_FOR_MWR = 30


# ============================================================
# Scalars
# ============================================================


def _m(x: Money) -> str:
    return str(x.amount)


def _w(x: Weight) -> str:
    return str(x.value)


def _pct(x: Decimal, places: int = 2) -> str:
    return Weight(x).as_percent(places)


def _cents(x: Money) -> int:
    return x.quantize().to_cents()


def _bps(x: Weight) -> int:
    return int((x.value * 10_000).to_integral_value())


# ============================================================
# Domain objects
# ============================================================


def _event(e: LedgerEvent) -> JSON:
    base: JSON = {"seq": e.seq, "on": e.on.isoformat(), "kind": type(e).__name__}
    match e:
        case Deposit(amount=amount) | Withdrawal(amount=amount):
            base |= {"amount": _m(amount), "amount_display": str(amount)}
        case Buy(ticker=ticker, quantity=quantity, price=price):
            base |= {
                "ticker": ticker,
                "quantity": str(quantity.quantity),
                "price": str(price.amount),
                "consideration": _m(e.consideration),
                "consideration_display": str(e.consideration),
            }
        case Sell(ticker=ticker, quantity=quantity, price=price, lot_ids=lot_ids):
            base |= {
                "ticker": ticker,
                "quantity": str(quantity.quantity),
                "price": str(price.amount),
                "consideration": _m(e.proceeds),
                "consideration_display": str(e.proceeds),
                "lot_ids": list(lot_ids),
            }
        case Dividend(ticker=ticker, amount=amount):
            base |= {
                "ticker": ticker,
                "amount": _m(amount),
                "amount_display": str(amount),
            }
        case Fee(amount=amount, description=description):
            base |= {
                "amount": _m(amount),
                "amount_display": str(amount),
                "description": description,
            }
    return base


def _lot(lot: TaxLot, price: Price, on: date) -> JSON:
    value = lot.market_value(price)
    unrealized = value - lot.cost_basis
    return {
        "lot_id": lot.lot_id,
        "acquired": lot.acquired.isoformat(),
        "quantity": str(lot.quantity.quantity),
        "cost_basis": _m(lot.cost_basis),
        "cost_basis_display": str(lot.cost_basis),
        "value": _m(value),
        "value_display": str(value),
        "unrealized": _m(unrealized),
        "unrealized_display": str(unrealized),
        "period": lot.period_at(on).value if on >= lot.acquired else "short",
        "covered": lot.covered,
        "wash_sale_adjustment": (
            None if lot.disallowed_loss_added is None else _m(lot.disallowed_loss_added)
        ),
    }


def _selection(sel: LotSelection) -> JSON:
    return {
        "lot_id": sel.lot_id,
        "quantity": str(sel.quantity.quantity),
        "acquired": sel.acquired.isoformat(),
        "proceeds": _m(sel.proceeds),
        "cost_basis": _m(sel.cost_basis),
        "gain": _m(sel.gain),
        "gain_display": str(sel.gain),
        "period": sel.period.value,
    }


def _trade(t: Trade, *, blocked: bool) -> JSON:
    gain = t.realized_gain
    return {
        "ticker": t.ticker,
        "side": t.side.value,
        "sleeve": t.sleeve,
        "quantity": str(t.quantity.quantity),
        "price": str(t.price.amount),
        "consideration": _m(t.consideration),
        "consideration_display": str(t.consideration),
        "reason": t.reason,
        "blocked": blocked,
        "lots": [_selection(s) for s in t.lots],
        "realized_gain": None if gain is None else _m(gain),
        "realized_gain_display": "" if gain is None else str(gain),
    }


def _unplaced(u: Unplaced) -> JSON:
    return {
        "sleeve": u.sleeve,
        "shortfall": _m(u.shortfall),
        "shortfall_display": str(u.shortfall),
        "reason": u.reason,
    }


def _violation(v: Violation) -> JSON:
    return {
        "constraint_id": v.constraint_id,
        "severity": v.severity.value,
        "subject": v.subject,
        "message": v.message,
        "authority": v.authority,
    }


def _drift(report: DriftReport) -> JSON:
    rows = []
    for r in report.sleeves:
        sign = "+" if r.drift.value > 0 else ""
        rows.append(
            {
                "sleeve": r.sleeve,
                "value": _m(r.value),
                "value_display": str(r.value),
                "actual": _w(r.actual),
                "actual_display": r.actual.as_percent(),
                "target": _w(r.target),
                "target_display": r.target.as_percent(),
                "drift": _w(r.drift),
                "drift_display": f"{sign}{r.drift.as_percent()}",
                "band_width": _w(r.band_width),
                "band_display": f"±{r.band_width.as_percent()}",
                "breached": r.breached,
                "in_model": r.in_model,
            }
        )
    return {
        "model_id": report.model_id,
        "model_version": report.model_version,
        "total": _m(report.total),
        "total_display": str(report.total),
        "needs_rebalancing": report.needs_rebalancing,
        "breaches": len(report.breaches),
        "sleeves": rows,
    }


def _compliance(result: ComplianceResult) -> JSON:
    return {
        "is_clear": result.is_clear,
        "passed": [_trade(t, blocked=False) for t in result.passed],
        "blocked": [_trade(t, blocked=True) for t in result.blocked],
        "violations": [_violation(v) for v in result.violations],
    }


def _review(r: Review) -> JSON:
    p = r.proposal
    return {
        "review_id": r.review_id,
        "on": r.on.isoformat(),
        "account_id": r.account_id,
        "status": r.status,
        "note": r.note,
        "model_id": p.model_id,
        "model_version": p.model_version,
        "trigger": p.trigger,
        "target_mode": p.policy.target_mode.value,
        "lot_method": p.policy.lot_method.value,
        "turnover": _m(p.turnover),
        "turnover_display": str(p.turnover),
        "cash_before": _m(p.cash_before),
        "cash_after": _m(p.cash_after),
        "drift_before": _drift(p.drift_before),
        "unplaced": [_unplaced(u) for u in p.unplaced],
        "compliance": _compliance(r.compliance),
        "applied_seqs": list(r.applied_seqs),
    }


def _audit_entry(e: AuditEntry) -> JSON:
    return {
        "seq": e.seq,
        "occurred_at": e.occurred_at.isoformat(),
        "actor": e.actor,
        "action": e.action.value,
        "subject": e.subject,
        "payload": dict(e.payload),
        "previous_hash": e.previous_hash,
        "digest": e.digest,
    }


def _harvest(h: HarvestOpportunity) -> JSON:
    return {
        "lot_id": h.lot.lot_id,
        "ticker": h.lot.ticker,
        "acquired": h.lot.acquired.isoformat(),
        "quantity": str(h.lot.quantity.quantity),
        "market_value": _m(h.market_value),
        "unrealized_loss": _m(h.unrealized_loss),
        "unrealized_loss_display": str(h.unrealized_loss),
        "period": h.period.value,
        "tax_benefit": _m(h.tax_benefit),
        "tax_benefit_display": str(h.tax_benefit),
        "is_blocked": h.is_blocked,
        "block_reason": h.block_reason,
        "blocked_by": [
            {
                "acquisition_id": a.acquisition_id,
                "account_id": a.account_id,
                "account_type": a.account_type.value,
                "ticker": a.ticker,
                "on": a.on.isoformat(),
                "forfeits": a.account_type.forfeits_wash_sale_basis,
            }
            for a in h.blocked_by
        ],
        "alternatives": list(h.alternatives),
    }


def _wash_sale(w: WashSale) -> JSON:
    return {
        "lot_id": w.disposal.lot_id,
        "ticker": w.disposal.ticker,
        "disposed": w.disposal.disposed.isoformat(),
        "loss": _m(w.disposal.gain),
        "disallowed": _m(w.disallowed),
        "deferred": _m(w.deferred),
        "forfeited": _m(w.forfeited),
        "replacements": [
            {
                "acquisition_id": r.acquisition_id,
                "account_id": r.account_id,
                "on": r.on.isoformat(),
                "matched": str(r.matched.quantity),
                "disallowed": _m(r.disallowed),
                "outcome": r.outcome.value,
            }
            for r in w.replacements
        ],
    }


# ============================================================
# Derived series
# ============================================================


def _disposals_through(events: Sequence[LedgerEvent]) -> list[Disposal]:
    """Every realised disposal, replaying sells exactly as build_lots
    does — by the lots the Sell names, or FIFO when it names none."""
    lots: dict[str, list[TaxLot]] = {}
    found: list[Disposal] = []
    for event in events:
        if isinstance(event, Buy):
            lots.setdefault(event.ticker, []).append(
                TaxLot(
                    lot_id=f"{event.ticker}-{event.seq}",
                    ticker=event.ticker,
                    acquired=event.on,
                    quantity=event.quantity,
                    cost_basis=event.consideration,
                )
            )
        elif isinstance(event, Sell):
            held = lots.get(event.ticker, [])
            if event.lot_ids:
                result = dispose(
                    held,
                    event.quantity,
                    event.price,
                    on=event.on,
                    method=LotMethod.SPECIFIC_ID,
                    chosen=event.lot_ids,
                )
            else:
                result = dispose(held, event.quantity, event.price, on=event.on)
            found.extend(result.disposals)
            lots[event.ticker] = list(result.remaining_lots)
    return found


def _realized(disposals: Sequence[Disposal], year: int) -> JSON:
    """Realised gains and losses for one calendar year, split the way
    Form 8949 wants them."""
    short = Money.zero()
    long = Money.zero()
    for d in disposals:
        if d.disposed.year != year:
            continue
        if d.period.value == "short":
            short = short + d.gain
        else:
            long = long + d.gain
    total = short + long
    return {
        "year": year,
        "short_term": _m(short),
        "short_term_display": str(short),
        "long_term": _m(long),
        "long_term_display": str(long),
        "total": _m(total),
        "total_display": str(total),
        "disposals": sum(1 for d in disposals if d.disposed.year == year),
    }


def _flows(events: Sequence[LedgerEvent]) -> list[ExternalFlow]:
    flows: list[ExternalFlow] = []
    for e in events:
        match e:
            case Deposit(on=on, amount=amount):
                flows.append(ExternalFlow(on, amount))
            case Withdrawal(on=on, amount=amount):
                flows.append(ExternalFlow(on, -amount))
            case Fee(on=on, amount=amount):
                flows.append(ExternalFlow(on, -amount, FlowKind.FEE))
            case _:
                pass
    return flows


def _performance(
    state: AccountHistory,
    valuations: Sequence[Valuation],
    through: date,
) -> JSON:
    events = state.events_through(through)
    series = [v for v in valuations if v.on <= through]
    if len(series) < 2:
        return {"available": False, "note": "performance needs two valuations"}

    result = compute_performance(series, _flows(events))

    annualized: JSON | None
    if result.can_annualize:
        gross, net = result.annualized()
        annualized = {
            "gross": str(gross),
            "gross_display": _pct(gross),
            "net": str(net),
            "net_display": _pct(net),
        }
        annualized_note = ""
    else:
        annualized = None
        annualized_note = (
            f"not annualised: {result.days}-day period. GIPS prohibits "
            "annualising returns for periods under one year."
        )

    mwr: str | None = None
    mwr_display = ""
    mwr_note = ""
    if result.days < MIN_DAYS_FOR_MWR:
        mwr_note = f"money-weighted return withheld under {MIN_DAYS_FOR_MWR} days"
    else:
        cash_flows: list[tuple[date, Money]] = []
        for e in events:
            if isinstance(e, Deposit):
                cash_flows.append((e.on, -e.amount))
            elif isinstance(e, Withdrawal):
                cash_flows.append((e.on, e.amount))
        cash_flows.append((through, series[-1].value))
        try:
            rate = money_weighted_return(cash_flows)
        except PerformanceError as exc:
            mwr_note = str(exc)
        else:
            mwr = f"{rate:.6f}"
            mwr_display = f"{rate * 100:.2f}%"

    return {
        "available": True,
        "start": result.start.isoformat(),
        "finish": result.finish.isoformat(),
        "days": result.days,
        "beginning_value": _m(result.beginning_value),
        "ending_value": _m(result.ending_value),
        "net_external_flow": _m(result.net_external_flow),
        "net_external_flow_display": str(result.net_external_flow),
        "fees_paid": _m(result.fees_paid),
        "fees_paid_display": str(result.fees_paid),
        "twr_gross": str(result.twr_gross),
        "twr_gross_display": _pct(result.twr_gross),
        "twr_net": str(result.twr_net),
        "twr_net_display": _pct(result.twr_net),
        "fee_drag_display": _pct(result.fee_drag),
        "annualized": annualized,
        "annualized_note": annualized_note,
        "mwr": mwr,
        "mwr_display": mwr_display,
        "mwr_note": mwr_note,
    }


def _valuations(state: AccountHistory, history: History) -> list[Valuation]:
    """Daily closing values from inception. Taken AFTER the day's flows,
    which is the convention `time_weighted_return` assumes."""
    series: list[Valuation] = []
    for day in history.prices.days:
        if day < history.start:
            continue
        portfolio = state.portfolio_on(day)
        series.append(Valuation(day, portfolio.total_value(history.prices.on(day))))
    return series


# ============================================================
# The snapshot
# ============================================================


def _account_snapshot(
    state: AccountHistory,
    history: History,
    on: date,
    valuations: Sequence[Valuation],
) -> JSON:
    prices = history.prices.on(on)
    events = state.events_through(on)
    portfolio = state.portfolio_on(on)
    lots = state.lots_on(on)
    values = market_values(portfolio, prices)
    drift = compute_drift(group_values(values, history.classification), state.model)
    total = portfolio.total_value(prices)

    positions = []
    for ticker, quantity in sorted(portfolio.positions.items()):
        if quantity.is_zero:
            continue
        value = values[ticker]
        basis = sum((lot.cost_basis for lot in lots.get(ticker, [])), Money.zero())
        unrealized = value - basis
        positions.append(
            {
                "ticker": ticker,
                "sleeve": history.classification.get(ticker, "unclassified"),
                "quantity": str(quantity.quantity),
                "price": str(prices[ticker].amount),
                "value": _m(value),
                "value_display": str(value),
                "cost_basis": _m(basis),
                "cost_basis_display": str(basis),
                "unrealized": _m(unrealized),
                "unrealized_display": str(unrealized),
                "lots": [_lot(lot, prices[ticker], on) for lot in lots.get(ticker, [])],
            }
        )

    disposals = _disposals_through(events)
    taxable = not state.account.account_type.is_tax_advantaged

    harvest: list[JSON] = []
    wash_sales: list[JSON] = []
    if taxable and state.rates is not None:
        acquisitions = history.household_acquisitions_through(on)
        flat = [lot for ticker in sorted(lots) for lot in lots[ticker]]
        harvest = [
            _harvest(h)
            for h in find_harvest_opportunities(
                flat,
                prices,
                acquisitions,
                history.substitutes,
                state.rates,
                on=on,
                account_id=state.account.account_id,
                account_type=state.account.account_type,
                minimum_loss=MIN_HARVEST,
            )
        ]
        wash_sales = [
            _wash_sale(w)
            for w in find_wash_sales(
                disposals, acquisitions, history.substitutes
            ).findings
        ]

    return {
        "account_id": state.account.account_id,
        "account_type": state.account.account_type.value,
        "events_through_seq": events[-1].seq if events else 0,
        "cash": _m(portfolio.cash),
        "cash_display": str(portfolio.cash),
        "total": _m(total),
        "total_display": str(total),
        "positions": positions,
        "drift": _drift(drift),
        "performance": _performance(state, valuations, on),
        "realized": [
            _realized(disposals, year)
            for year in range(history.start.year, on.year + 1)
        ],
        "harvest": harvest,
        "wash_sales": wash_sales,
    }


def snapshot_dates(history: History) -> list[date]:
    """Every Friday close, every day something happened, and today."""
    days: set[date] = {d for d in history.prices.days if d.weekday() == 4}
    days.update(event_dates(history))
    days.add(history.today)
    return sorted(d for d in days if history.start <= d <= history.today)


def _daily(
    history: History, valuations: Mapping[str, Sequence[Valuation]]
) -> list[JSON]:
    taxable = history.accounts["taxable-1"]
    rows: list[JSON] = []
    by_day = {
        account_id: {v.on: v.value for v in series}
        for account_id, series in valuations.items()
    }
    for day in history.prices.days:
        if day < history.start:
            continue
        prices = history.prices.on(day)
        portfolio = taxable.portfolio_on(day)
        drift = compute_drift(
            group_values(market_values(portfolio, prices), history.classification),
            taxable.model,
        )
        row: JSON = {
            "on": day.isoformat(),
            "taxable_total_cents": _cents(by_day["taxable-1"][day]),
            "roth_total_cents": _cents(by_day["roth-1"][day]),
            "household_total_display": str(
                by_day["taxable-1"][day] + by_day["roth-1"][day]
            ),
            "taxable_breached": drift.needs_rebalancing,
        }
        for r in drift.sleeves:
            row[f"taxable_{r.sleeve}_bps"] = _bps(r.actual)
        for ticker, price in prices.items():
            row[f"{ticker}_cents"] = int((price.amount * 100).to_integral_value())
        rows.append(row)
    return rows


def _story(history: History) -> JSON:
    """Pointers into the data, so a page need not search for them."""
    blocked = [r for r in history.reviews if r.status == "rejected"]
    first_block = blocked[0] if blocked else None
    forfeit_block = next(
        (
            r
            for r in blocked
            if any("FORFEITED" in v.message for v in r.compliance.blocks)
        ),
        None,
    )
    loss_sales = [
        r
        for r in history.reviews
        if r.status == "approved"
        and any(
            t.realized_gain is not None and t.realized_gain.is_negative
            for t in r.proposal.sells
        )
    ]
    first_loss_sale_after_block = next(
        (
            r
            for r in loss_sales
            if forfeit_block is not None and r.on > forfeit_block.on
        ),
        None,
    )
    return {
        "first_blocked_review": first_block.review_id if first_block else None,
        "forfeiture_block_review": forfeit_block.review_id if forfeit_block else None,
        "clean_loss_sale_review": (
            first_loss_sale_after_block.review_id
            if first_loss_sale_after_block
            else None
        ),
        "approved_reviews": sum(1 for r in history.reviews if r.status == "approved"),
        "rejected_reviews": len(blocked),
        "no_action_reviews": sum(1 for r in history.reviews if r.status == "no_action"),
    }


def render(history: History) -> JSON:
    valuations = {
        account_id: _valuations(state, history)
        for account_id, state in history.accounts.items()
    }
    audit_entries = list(history.audit.entries)

    snapshots: list[JSON] = []
    for on in snapshot_dates(history):
        as_of = AuditLog(tuple(e for e in audit_entries if e.occurred_at.date() <= on))
        reviews_today = [r.review_id for r in history.reviews if r.on == on]
        accounts = {
            account_id: _account_snapshot(state, history, on, valuations[account_id])
            for account_id, state in history.accounts.items()
        }
        household_total = sum(
            (Money(str(a["total"])) for a in accounts.values()), Money.zero()
        )
        snapshots.append(
            {
                "on": on.isoformat(),
                "weekday": on.strftime("%A"),
                "is_trading_day": history.prices.is_trading_day(on),
                "review_ids": reviews_today,
                "household_total": _m(household_total),
                "household_total_display": str(household_total),
                "audit_entries": len(as_of),
                "audit_head": as_of.head,
                "accounts": accounts,
            }
        )

    return {
        "meta": {
            "engine": "meridian 0.1.0",
            "generated_by": "python -m meridian.showcase",
            "start": history.start.isoformat(),
            "today": history.today.isoformat(),
            "synthetic": True,
            "note": (
                "Prices are seeded random walks, not market data. Client "
                "activity (deposits, dividends, fees, one withdrawal, two "
                "client-directed purchases) is scripted. Every trade, lot "
                "selection, compliance verdict, and audit entry was produced "
                "by the engine from those inputs and is reproducible from the "
                "event stream. Trading days are Mon-Fri; market holidays are "
                "not modelled."
            ),
            "money_rule": (
                "Every amount, quantity, weight and return is a JSON string. "
                "Integer *_cents and *_bps fields exist only for charting."
            ),
            "snapshots": len(snapshots),
            "reviews": len(history.reviews),
            "audit_entries": len(audit_entries),
        },
        "story": _story(history),
        "household": {
            "household_id": history.household.household_id,
            "accounts": [
                {
                    "account_id": a.account_id,
                    "account_type": a.account_type.value,
                    "custodian": a.custodian,
                    "owner": a.owner,
                    "model_id": history.accounts[a.account_id].model.model_id,
                    "model_version": history.accounts[a.account_id].model.version,
                    "lot_aware": not a.account_type.is_tax_advantaged,
                }
                for a in history.household.accounts.values()
            ],
        },
        "models": {
            state.model.model_id: {
                "model_id": state.model.model_id,
                "version": state.model.version,
                "name": state.model.name,
                "sleeves": {
                    key: {
                        "target": _w(s.target),
                        "target_display": s.target.as_percent(),
                        "tolerance": _w(s.tolerance),
                        "band_kind": s.band_kind.value,
                        "lower_display": s.lower_bound.as_percent(),
                        "upper_display": s.upper_bound.as_percent(),
                        "security": s.security,
                    }
                    for key, s in state.model.sleeves.items()
                },
            }
            for state in history.accounts.values()
        },
        "classification": dict(history.classification),
        "substitutes": {
            "identical": {
                k: sorted(v) for k, v in history.substitutes.identical.items()
            },
            "alternatives": {
                k: list(v) for k, v in history.substitutes.alternatives.items()
            },
        },
        "rates": {
            "taxable-1": {
                "short_term": "0.37",
                "long_term": "0.20",
                "niit": "0.038",
                "note": "high-bracket client; supplied, never assumed",
            }
        },
        "events": {
            account_id: [_event(e) for e in state.events]
            for account_id, state in history.accounts.items()
        },
        "reviews": [_review(r) for r in history.reviews],
        "audit": {
            "entries": [_audit_entry(e) for e in audit_entries],
            "head": history.audit.head,
            "is_intact": history.audit.is_intact,
        },
        "prices": {
            ticker: [
                [d.isoformat(), str(series[d].amount)] for d in history.prices.days
            ]
            for ticker, series in history.prices.closes.items()
        },
        "daily": _daily(history, valuations),
        "snapshots": snapshots,
    }


def write(history: History, path: Path) -> int:
    """Write the rendered history. Returns the byte count."""
    document = render(history)
    text = json.dumps(document, separators=(",", ":"), ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def account_portfolio(history: History, account_id: str, on: date) -> Portfolio:
    """Convenience for tests: the ledger's view on a given date."""
    return history.accounts[account_id].portfolio_on(on)
