"""Money and the other quantities this system measures.

============================================================
WHY THESE ARE CLASSES AND NOT JUST NUMBERS
============================================================

Two reasons, and both are load-bearing.

1. EXACTNESS. Binary floating point cannot represent 0.10, the same way
   decimal cannot represent 1/3. `0.1 + 0.2 != 0.3` in Python. One
   transaction off by a hair is invisible; a million of them is a set of
   books that does not balance. So every value here wraps `Decimal`,
   which is exact, and refuses to be built from a float at all.

2. UNITS. A dollar amount and a share count are both "numbers," and
   nothing stops you adding them together — which is meaningless and
   silent. Making them separate types means the mistake is caught twice:
   by mypy before the code runs, and by a TypeError if it somehow does.

   Think of each as a labelled jar. The label is part of the value. You
   can pour one money jar into another. You cannot pour shares into
   dollars, and the jar itself refuses rather than you remembering to.

The type algebra below is deliberate:

    Shares x Price  -> Money      (what a position is worth)
    Money  / Price  -> Shares     (how many shares that buys)
    Money  x Weight -> Money      (a slice of a portfolio)
    Money  / Money  -> Weight     (what fraction of the total this is)

Every one of those is an operation this engine actually performs, and
the units work out. Anything not in that list is a bug, and the type
checker will say so.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Final

# ============================================================
# Rounding policy
# ============================================================
# ROUND_HALF_UP is what a person expects: 0.005 becomes 0.01.
#
# Python's own default is ROUND_HALF_EVEN ("banker's rounding"), which
# rounds 0.005 to 0.00 and 0.015 to 0.02 — alternating, so that a large
# sample does not drift upward. That is the right choice for statistics
# and the wrong one for a client statement, where a number that rounds
# "down" for no visible reason generates a phone call.
#
# The point is less which one we picked and more that it is written down
# in one place as a named constant, rather than being whatever the
# library happened to default to.
MONEY_ROUNDING: Final = ROUND_HALF_UP

# The smallest unit money is expressed in: one cent.
CENT: Final = Decimal("0.01")

# Quantities are held to eight decimal places. Custodians commonly
# support three to five for fractional shares; we store more precision
# than we display and round only at the point an order is placed.
SHARE_PRECISION: Final = Decimal("0.00000001")


def _to_decimal(value: Decimal | int | str, what: str) -> Decimal:
    """Convert an input to Decimal, refusing floats.

    This is the gate. `Decimal(0.1)` succeeds in Python and produces
    0.1000000000000000055511151231257827021181583404541015625 — the
    float's error, faithfully preserved. Accepting a float here would
    quietly undo the entire reason this module exists, so a float is
    rejected at the door rather than converted.

    Strings, ints, and Decimals are all exact and all allowed.
    """
    if isinstance(value, float):
        raise TypeError(
            f"{what} cannot be built from a float ({value!r}). "
            f'Use a string instead: {what}("{value}"). '
            "Floats carry binary rounding error that would corrupt "
            "every calculation downstream."
        )
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, str):
        try:
            return Decimal(value)
        except Exception as exc:
            raise ValueError(f"{what} could not parse {value!r}") from exc
    raise TypeError(f"{what} cannot be built from {type(value).__name__}")


# ============================================================
# Weight — a ratio. 0.55 means 55%.
# ============================================================


@dataclass(frozen=True, slots=True, order=True)
class Weight:
    """A proportion of a whole, stored as a decimal rather than a percent.

    Store 0.55, render "55.0%". Mixing the two representations in one
    codebase produces off-by-100 errors that read as correct in review,
    so the rule is that a Weight is always the decimal form and the
    percent only ever exists as display text.
    """

    value: Decimal

    def __init__(self, value: Decimal | int | str) -> None:
        # frozen dataclasses forbid normal attribute assignment, so the
        # converted value has to go in through object.__setattr__.
        object.__setattr__(self, "value", _to_decimal(value, "Weight"))

    def __add__(self, other: Weight) -> Weight:
        if not isinstance(other, Weight):
            raise TypeError(f"cannot add {type(other).__name__} to Weight")
        return Weight(self.value + other.value)

    def __sub__(self, other: Weight) -> Weight:
        if not isinstance(other, Weight):
            raise TypeError(f"cannot subtract {type(other).__name__} from Weight")
        return Weight(self.value - other.value)

    def __neg__(self) -> Weight:
        return Weight(-self.value)

    def __abs__(self) -> Weight:
        return Weight(abs(self.value))

    def as_percent(self, places: int = 1) -> str:
        """Render for humans: Weight('0.55') -> '55.0%'."""
        scaled = (self.value * 100).quantize(
            Decimal(1).scaleb(-places), rounding=MONEY_ROUNDING
        )
        return f"{scaled}%"

    def __repr__(self) -> str:
        return f"Weight({str(self.value)!r})"


# ============================================================
# Money — an amount of currency.
# ============================================================


@dataclass(frozen=True, slots=True, order=True)
class Money:
    """An exact amount of money.

    Frozen (immutable) on purpose. An amount that can be modified in
    place can be modified by accident from somewhere you are not
    looking; every operation here returns a new Money instead.

    Note this class deliberately does NOT define __mul__ for
    Money * Money. Multiplying two amounts of money produces
    dollars-squared, which is not a thing. Only the operations in the
    type algebra at the top of this file exist.
    """

    amount: Decimal

    def __init__(self, amount: Decimal | int | str) -> None:
        object.__setattr__(self, "amount", _to_decimal(amount, "Money"))

    # ---- construction helpers ----

    @classmethod
    def zero(cls) -> Money:
        return cls("0")

    @classmethod
    def from_cents(cls, cents: int) -> Money:
        """Build from an integer number of cents. 1234 -> $12.34.

        The allocator works in whole cents because integers cannot carry
        a fraction that needs rounding away, which is what makes the
        conservation guarantee possible. This is how it converts back.
        """
        if not isinstance(cents, int):
            raise TypeError("from_cents requires an int")
        return cls(Decimal(cents) * CENT)

    # ---- arithmetic ----

    def __add__(self, other: Money) -> Money:
        # The annotation above is the first lock: mypy rejects
        # `Money + Shares` without running anything. This isinstance
        # check is the second lock, for values arriving at runtime from
        # JSON or a database where the type checker cannot see them.
        if not isinstance(other, Money):
            raise TypeError(
                f"cannot add {type(other).__name__} to Money — "
                "only Money may be added to Money"
            )
        return Money(self.amount + other.amount)

    def __sub__(self, other: Money) -> Money:
        if not isinstance(other, Money):
            raise TypeError(
                f"cannot subtract {type(other).__name__} from Money — "
                "only Money may be subtracted from Money"
            )
        return Money(self.amount - other.amount)

    def __mul__(self, weight: Weight) -> Money:
        """Money x Weight -> Money. A slice of a total."""
        if not isinstance(weight, Weight):
            raise TypeError(
                f"cannot multiply Money by {type(weight).__name__} — "
                "only by a Weight (use Money.divide_by_price for shares)"
            )
        return Money(self.amount * weight.value)

    def __neg__(self) -> Money:
        return Money(-self.amount)

    def __abs__(self) -> Money:
        return Money(abs(self.amount))

    def ratio_to(self, total: Money) -> Weight:
        """Money / Money -> Weight. What fraction of the total this is.

        A zero total yields a zero weight rather than raising. An empty
        portfolio has no allocation error — every sleeve is legitimately
        0% of nothing — and making the caller guard for it at every call
        site would be noise.
        """
        if not isinstance(total, Money):
            raise TypeError(f"cannot take a ratio of Money to {type(total).__name__}")
        if total.amount == 0:
            return Weight("0")
        return Weight(self.amount / total.amount)

    # ---- rounding and conversion ----

    def quantize(self) -> Money:
        """Round to whole cents using the documented policy."""
        return Money(self.amount.quantize(CENT, rounding=MONEY_ROUNDING))

    def to_cents(self) -> int:
        """Exact integer cents. Raises if the amount is not a whole cent.

        Deliberately strict. A silent round here is exactly the kind of
        invisible leak this system exists to prevent, so a caller with a
        fractional cent has to say what it wants done with it by calling
        quantize() first.
        """
        scaled = self.amount / CENT
        if scaled != scaled.to_integral_value():
            raise ValueError(
                f"{self!r} is not a whole number of cents. "
                "Call .quantize() first and decide the rounding explicitly."
            )
        return int(scaled)

    # ---- predicates ----

    @property
    def is_zero(self) -> bool:
        return self.amount == 0

    @property
    def is_negative(self) -> bool:
        return self.amount < 0

    # ---- display ----

    def __str__(self) -> str:
        """Human-facing: -$1,234.56"""
        q = self.quantize().amount
        sign = "-" if q < 0 else ""
        return f"{sign}${abs(q):,.2f}"

    def __repr__(self) -> str:
        return f"Money({str(self.amount)!r})"


# ============================================================
# Price — currency per share.
# ============================================================


@dataclass(frozen=True, slots=True, order=True)
class Price:
    """The price of one share. Dollars per share, not dollars."""

    amount: Decimal

    def __init__(self, amount: Decimal | int | str) -> None:
        object.__setattr__(self, "amount", _to_decimal(amount, "Price"))

    def __repr__(self) -> str:
        return f"Price({str(self.amount)!r})"

    def __str__(self) -> str:
        return f"${self.amount:,.4f}"


# ============================================================
# Shares — a quantity of a security.
# ============================================================


@dataclass(frozen=True, slots=True, order=True)
class Shares:
    """A quantity of a security. Fractional quantities are real.

    Most custodians now support fractional shares to three or more
    decimal places, so this cannot be an integer type.
    """

    quantity: Decimal

    def __init__(self, quantity: Decimal | int | str) -> None:
        object.__setattr__(self, "quantity", _to_decimal(quantity, "Shares"))

    @classmethod
    def zero(cls) -> Shares:
        return cls("0")

    def __add__(self, other: Shares) -> Shares:
        if not isinstance(other, Shares):
            raise TypeError(f"cannot add {type(other).__name__} to Shares")
        return Shares(self.quantity + other.quantity)

    def __sub__(self, other: Shares) -> Shares:
        if not isinstance(other, Shares):
            raise TypeError(f"cannot subtract {type(other).__name__} from Shares")
        return Shares(self.quantity - other.quantity)

    def __neg__(self) -> Shares:
        return Shares(-self.quantity)

    def __abs__(self) -> Shares:
        return Shares(abs(self.quantity))

    def value_at(self, price: Price) -> Money:
        """Shares x Price -> Money. What the position is worth."""
        if not isinstance(price, Price):
            raise TypeError(
                f"cannot value Shares at {type(price).__name__} — needs a Price"
            )
        return Money(self.quantity * price.amount)

    def round_to(self, increment: Decimal) -> Shares:
        """Round DOWN to the custodian's tradeable increment.

        Always down, never nearest. Rounding a buy up would order more
        than the cash on hand can fund; rounding a sell up would sell
        shares that are not held. Both are rejected at the custodian,
        and both produce a client statement that does not tie. The
        leftover cash simply stays as cash.
        """
        if increment <= 0:
            raise ValueError("increment must be positive")
        whole_steps = abs(self.quantity) / increment
        steps = whole_steps.to_integral_value(rounding="ROUND_DOWN")
        magnitude = steps * increment
        return Shares(-magnitude if self.quantity < 0 else magnitude)

    @property
    def is_zero(self) -> bool:
        return self.quantity == 0

    def __repr__(self) -> str:
        return f"Shares({str(self.quantity)!r})"

    def __str__(self) -> str:
        return f"{self.quantity.normalize():f} shares"
