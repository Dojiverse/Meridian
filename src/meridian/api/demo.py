"""A worked example to run the interface against.

Two accounts belonging to one household, chosen to exercise the parts of
the engine that matter:

  taxable-1  drifted off its model, holds a losing GLD position, and has
             a matching GLD purchase sitting in the Roth — so the wash
             sale screen has something real to catch.

  roth-1     tax-advantaged, so harvesting is meaningless there and the
             short-term-gain warning stays quiet.

Fixed dates and fixed prices. Nothing here reads a clock, so the numbers
on the screen are the same today as they were yesterday — which is what
lets a screenshot be compared against a later run.
"""

from __future__ import annotations

from datetime import date

from meridian.api.store import AccountState, Store
from meridian.compliance import (
    ComplianceGate,
    ConcentrationLimit,
    MinimumCash,
    RestrictedSecurity,
    ShortTermGainLimit,
    WashSaleBlock,
)
from meridian.household import Account, AccountType
from meridian.ledger import Buy, Deposit, LedgerEvent
from meridian.model import BandKind, Model, Sleeve
from meridian.money import Money, Price, Shares, Weight
from meridian.washsale import SubstituteMap

TODAY = date(2026, 9, 8)

PRICES = {
    "VTI": Price("140.00"),
    "AAPL": Price("180.00"),
    "BND": Price("50.00"),
    "GLD": Price("150.00"),
    "VEA": Price("52.00"),
}

CLASSIFICATION = {
    "VTI": "equity",
    "AAPL": "equity",
    "VEA": "equity",
    "BND": "bond",
    "GLD": "alt",
}

CLASSIC = Model(
    model_id="classic-60-40",
    version=1,
    name="Classic balanced",
    sleeves={
        "equity": Sleeve(Weight("0.55"), Weight("0.05"), security="VTI"),
        "bond": Sleeve(Weight("0.35"), Weight("0.05"), security="BND"),
        # A RELATIVE band on the small sleeve. A 5-point absolute band
        # would let a 10% target vanish to zero without ever tripping.
        "alt": Sleeve(
            Weight("0.10"), Weight("0.30"), BandKind.RELATIVE, security="GLD"
        ),
    },
)

CONSERVATIVE = Model(
    model_id="conservative-40-60",
    version=2,
    name="Conservative",
    sleeves={
        "equity": Sleeve(Weight("0.40"), Weight("0.05"), security="VTI"),
        "bond": Sleeve(Weight("0.60"), Weight("0.05"), security="BND"),
    },
)

# The firm's policy on substantial identity. GLD and IAU both hold
# physical gold; this firm has decided to treat them as identical. A
# judgment call, recorded as data.
SUBSTITUTES = SubstituteMap.symmetric(
    groups=[["GLD", "IAU"], ["VTI", "ITOT"]],
    alternatives={"GLD": ("SLV",), "VTI": ("SCHB",)},
)


def build_demo_store() -> Store:
    store = Store(prices=dict(PRICES), today=TODAY)

    # ---- taxable-1 --------------------------------------------
    # 60/25/15 against a 55/35/10 model: bond is ten points light and
    # alt is five points heavy. GLD was bought at $200 and is now $150,
    # so selling it realises a loss — which the Roth purchase below
    # would wash.
    taxable_events: list[LedgerEvent] = [
        Deposit(1, date(2024, 1, 2), Money("120000.00")),
        Buy(2, date(2024, 1, 2), "VTI", Shares("300"), Price("140.00")),
        Buy(3, date(2024, 1, 2), "AAPL", Shares("100"), Price("180.00")),
        Buy(4, date(2024, 1, 2), "BND", Shares("500"), Price("50.00")),
        Buy(5, date(2026, 8, 20), "GLD", Shares("100"), Price("200.00")),
    ]

    store.accounts["taxable-1"] = AccountState(
        account=Account("taxable-1", AccountType.TAXABLE, "Schwab", "client"),
        model=CLASSIC,
        classification=CLASSIFICATION,
        events=taxable_events,
        substitutes=SUBSTITUTES,
        gate=ComplianceGate(
            constraints=(
                RestrictedSecurity(
                    constraint_id="ips-4.2",
                    tickers=frozenset({"AAPL"}),
                    reason="client's former employer; may exit but not add",
                    authority="IPS clause 4.2",
                    sell_only=True,
                ),
                ConcentrationLimit(
                    constraint_id="ips-3.1",
                    limit=Weight("0.40"),
                    authority="IPS clause 3.1",
                    exempt=frozenset({"VTI", "BND"}),
                ),
                MinimumCash(
                    constraint_id="ips-6",
                    minimum=Money("0.00"),
                    authority="IPS clause 6",
                ),
                ShortTermGainLimit(
                    constraint_id="tax-stcg",
                    threshold=Money("250.00"),
                    authority="firm tax policy",
                ),
                WashSaleBlock(constraint_id="wash-1091"),
            )
        ),
    )

    # ---- roth-1 ------------------------------------------------
    # Holds GLD too, bought inside the wash-sale window. That is what
    # turns the taxable GLD sale from a deferral into a permanent
    # forfeiture under Rev. Rul. 2008-5.
    roth_events: list[LedgerEvent] = [
        Deposit(1, date(2024, 3, 1), Money("60000.00")),
        Buy(2, date(2024, 3, 1), "VTI", Shares("200"), Price("140.00")),
        Buy(3, date(2026, 9, 1), "GLD", Shares("50"), Price("152.00")),
    ]

    store.accounts["roth-1"] = AccountState(
        account=Account("roth-1", AccountType.ROTH_IRA, "Schwab", "client"),
        model=CONSERVATIVE,
        classification=CLASSIFICATION,
        events=roth_events,
        substitutes=SUBSTITUTES,
        gate=ComplianceGate(
            constraints=(
                MinimumCash("ips-6", Money("0.00"), authority="IPS clause 6"),
                # Deliberately still registered. It produces nothing
                # inside a Roth — there is no taxable event to warn
                # about — and that silence is the constraint working,
                # not the constraint missing.
                ShortTermGainLimit("tax-stcg", Money("250.00")),
            )
        ),
    )

    return store
