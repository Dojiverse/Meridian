"""Hypothesis strategies — the recipes for the crash-test robot.

A "strategy" tells Hypothesis how to invent a value. Instead of writing
example inputs by hand (and only ever testing the cases you thought of),
you describe the SHAPE of valid input and let Hypothesis hunt for the
one that breaks you.

When it finds a failure it then SHRINKS it: it keeps simplifying the
input while the failure persists, so what lands on your desk is the
smallest case that still breaks — two holdings and three cents, not the
47-holding portfolio it happened to find first.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from hypothesis import strategies as st

from meridian.ledger import (
    Buy,
    Deposit,
    Dividend,
    Fee,
    LedgerEvent,
    Sell,
    Withdrawal,
)
from meridian.money import Money, Price, Shares, Weight

# Ticker-ish keys. Kept to a small alphabet on purpose: short, similar
# keys make collisions and tie-breaks FAR more likely, which is exactly
# where the interesting bugs live. Random 20-character strings would
# almost never tie.
keys = st.text(alphabet="ABCDEFG", min_size=1, max_size=3)


@st.composite
def money(
    draw: st.DrawFn,
    min_cents: int = -1_000_000_000,
    max_cents: int = 1_000_000_000,
) -> Money:
    """Any amount from -$10M to $10M, always a whole number of cents.

    Built from an integer count of cents rather than a decimal string,
    so the generated values are exactly the shape real money takes.
    """
    return Money.from_cents(draw(st.integers(min_value=min_cents, max_value=max_cents)))


@st.composite
def positive_money(draw: st.DrawFn) -> Money:
    return Money.from_cents(draw(st.integers(min_value=1, max_value=1_000_000_000)))


@st.composite
def weights(draw: st.DrawFn, min_size: int = 1, max_size: int = 8) -> dict[str, Weight]:
    """A set of weights summing to EXACTLY 1.

    Generating these is subtly hard. Drawing random decimals and dividing
    by their sum reintroduces the very rounding error we are testing for,
    so instead:

        1. Draw a set of keys.
        2. Cut a stick of 1,000,000 units at random points.
        3. Each piece divided by 1,000,000 is an exact decimal, and the
           pieces sum to exactly 1 because the cuts partition the stick.

    Cutting a stick cannot lose length, which is precisely the property
    the code under test is supposed to have.
    """
    units = 1_000_000

    names = draw(st.lists(keys, min_size=min_size, max_size=max_size, unique=True))
    n = len(names)

    # n-1 cut points, sorted; the gaps between them are the pieces.
    cuts = sorted(draw(st.lists(st.integers(0, units), min_size=n - 1, max_size=n - 1)))
    bounds = [0, *cuts, units]
    pieces = [bounds[i + 1] - bounds[i] for i in range(n)]

    return {
        name: Weight(Decimal(piece) / units)
        for name, piece in zip(names, pieces, strict=True)
    }


@st.composite
def event_streams(
    draw: st.DrawFn, min_events: int = 0, max_events: int = 25
) -> list[LedgerEvent]:
    """A VALID sequence of ledger events.

    The hard part is that validity is stateful: you cannot sell shares
    you have not bought, or spend cash you do not have. A strategy that
    drew events independently would generate mostly-invalid streams, and
    the tests would spend their time confirming that the guards fire
    rather than exercising the arithmetic.

    So this draws the stream the way it actually happens — one event at
    a time, tracking cash and positions as it goes, and only ever
    offering choices that are legal in the state reached so far.

    Prices are whole cents and quantities are whole shares on purpose.
    Fractional-share rounding is tested in test_money.py; mixing it in
    here would blur which layer a failure came from.
    """
    tickers = ["VTI", "BND", "GLD", "AAPL"]
    n = draw(st.integers(min_value=min_events, max_value=max_events))

    events: list[LedgerEvent] = []
    cash = Money.zero()
    positions: dict[str, Shares] = {}
    on = date(2026, 1, 1)

    for seq in range(1, n + 1):
        held = [t for t, q in positions.items() if q.quantity > 0]

        # Only offer what the current state permits.
        options = ["deposit", "fee"]
        if cash.amount > 0:
            options += ["withdrawal", "buy"]
        if held:
            options += ["sell", "dividend"]

        match draw(st.sampled_from(options)):
            case "deposit":
                amount = Money.from_cents(draw(st.integers(1, 100_000_00)))
                events.append(Deposit(seq, on, amount))
                cash = cash + amount

            case "withdrawal":
                amount = Money.from_cents(draw(st.integers(1, cash.to_cents())))
                events.append(Withdrawal(seq, on, amount))
                cash = cash - amount

            case "fee":
                # Deliberately not capped by cash — fees may overdraw.
                amount = Money.from_cents(draw(st.integers(1, 10_000)))
                events.append(Fee(seq, on, amount))
                cash = cash - amount

            case "buy":
                price = Price(Decimal(draw(st.integers(1, 100_000))) / 100)
                affordable = int(cash.amount / price.amount)
                if affordable < 1:
                    # Cannot afford a single share at this price. Post a
                    # deposit instead of discarding the draw, so the
                    # stream keeps its length and Hypothesis keeps its
                    # ability to shrink cleanly.
                    amount = Money.from_cents(draw(st.integers(1, 100_000_00)))
                    events.append(Deposit(seq, on, amount))
                    cash = cash + amount
                    continue
                quantity = Shares(draw(st.integers(1, affordable)))
                ticker = draw(st.sampled_from(tickers))
                events.append(Buy(seq, on, ticker, quantity, price))
                cash = cash - quantity.value_at(price)
                positions[ticker] = positions.get(ticker, Shares.zero()) + quantity

            case "sell":
                ticker = draw(st.sampled_from(held))
                available = int(positions[ticker].quantity)
                quantity = Shares(draw(st.integers(1, available)))
                price = Price(Decimal(draw(st.integers(1, 100_000))) / 100)
                events.append(Sell(seq, on, ticker, quantity, price))
                positions[ticker] = positions[ticker] - quantity
                cash = cash + quantity.value_at(price)

            case "dividend":
                ticker = draw(st.sampled_from(held))
                amount = Money.from_cents(draw(st.integers(1, 50_000)))
                events.append(Dividend(seq, on, ticker, amount))
                cash = cash + amount

    return events


@st.composite
def shares(draw: st.DrawFn) -> Shares:
    """Share quantities, including fractional ones down to 1e-8."""
    raw = draw(st.integers(min_value=-10_000_000_000, max_value=10_000_000_000))
    return Shares(Decimal(raw) * Decimal("0.00000001"))


@st.composite
def prices(draw: st.DrawFn) -> Price:
    """Prices from $0.0001 to $10,000, to four decimal places."""
    raw = draw(st.integers(min_value=1, max_value=100_000_000))
    return Price(Decimal(raw) * Decimal("0.0001"))
