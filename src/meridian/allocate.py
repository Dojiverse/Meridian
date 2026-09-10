"""Dividing money without losing any.

============================================================
THE PROBLEM
============================================================

Allocate $10,000.00 equally across six sleeves. Each gets
$1,666.6666..., which rounds to $1,666.67. Six of those is $10,000.02
— two cents that do not exist. Round down instead and you get
$9,999.96, orphaning four cents.

Neither result reconciles. At scale, neither is auditable.

============================================================
THE SOLUTION: LARGEST REMAINDER METHOD
============================================================

Borrowed from apportionment theory, where it is known as Hamilton's
method and is used to divide legislative seats among states. The shape
of the problem is identical: divide an indivisible quantity by
proportions that do not divide evenly.

    1. Convert the total to whole cents (integers cannot carry a
       fraction that needs rounding away — this is the trick).
    2. Give every claim the whole number of cents it definitely earns
       (floor).
    3. Count what is left over.
    4. Hand out the leftovers, one cent at a time, to whoever was
       shortchanged most by the flooring.

The result sums to the total BY CONSTRUCTION. We never add up the parts
afterwards and hope; there is no arrangement of inputs for which the sum
comes out wrong.

============================================================
THE DETAIL INSIDE THE DETAIL: TIE-BREAKS
============================================================

Six equal sleeves means six identical fractional parts. Four cents have
to go to four of six equal claims, so THE TIE-BREAK DECIDES THE ANSWER.

If that tie-break is dictionary iteration order, or an unstable sort,
the same inputs produce different output on different runs. Nothing
crashes. No error is raised. The books just quietly stop matching last
quarter's.

That is not a rounding bug, it is an AUDITABILITY bug: you can no
longer reproduce a past allocation, which is the one thing this system
exists to be able to do.

So the tie-break here is a documented total order — largest fraction
first, then key ascending — and it is part of the contract, not an
implementation detail.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from decimal import Decimal
from typing import Final

from meridian.money import Money, Weight

__all__ = ["AllocationError", "allocate", "equal_weights", "normalize_weights"]

# Weights are held to this many decimal places when they have to be
# derived rather than stated. Six places is one hundredth of a basis
# point — far finer than any model is specified to, and coarse enough
# that the arithmetic stays exact.
WEIGHT_PRECISION: Final = Decimal("0.000001")


class AllocationError(ValueError):
    """Raised when an allocation is asked for that cannot conserve value."""


def allocate(total: Money, weights: Mapping[str, Weight]) -> dict[str, Money]:
    """Divide `total` across `weights` so the parts sum to exactly `total`.

    Args:
        total: The amount to divide. May be negative (a debit).
        weights: Target proportions by key. Must sum to exactly 1.

    Returns:
        A dict with the same keys as `weights`, whose values sum to
        exactly `total`.

    Raises:
        AllocationError: if the weights cannot produce a conserving
            allocation — they do not sum to 1, one is negative, or the
            mapping is empty while there is money to place.

    The guarantee, which the property tests in tests/test_allocate.py
    assert over thousands of generated inputs:

        sum(allocate(total, weights).values()) == total
    """
    _validate(total, weights)

    if not weights:
        # No claims and no money is a legitimate no-op. (_validate has
        # already rejected the case where there IS money to place.)
        return {}

    # ---- Negative totals ------------------------------------------
    # Flooring is asymmetric around zero: floor(-2.5) is -3, not -2. Run
    # the algorithm on the magnitude and flip the signs back at the end,
    # so a debit of $100 divides exactly like a credit of $100 and
    # conservation holds identically in both directions.
    sign = -1 if total.is_negative else 1
    magnitude = abs(total)

    # Work in whole cents. quantize() first so a total carrying sub-cent
    # precision (the result of an earlier multiplication, say) is
    # resolved explicitly here rather than blowing up inside to_cents().
    total_cents = magnitude.quantize().to_cents()

    shares = _apportion(total_cents, {k: w.value for k, w in weights.items()})

    return {key: Money.from_cents(sign * cents) for key, cents in shares.items()}


def _apportion(total_units: int, weights: Mapping[str, Decimal]) -> dict[str, int]:
    """Split `total_units` into whole units that sum to exactly it.

    The largest remainder method, in integer units. This is the single
    audited implementation of "divide something indivisible" — allocate()
    calls it with cents, normalize_weights() calls it with millionths of
    a weight, and the quarterly fee split will call it with cents again.
    One implementation to get right, one to test, one to defend.

    Args:
        total_units: A non-negative count of indivisible units.
        weights: Proportions by key, summing to exactly 1.
    """
    # ---- Step 1: give everyone what they definitely earn ----------
    floors: dict[str, int] = {}
    fractions: dict[str, Decimal] = {}

    for key, weight in weights.items():
        exact = Decimal(total_units) * weight
        whole = int(exact.to_integral_value(rounding="ROUND_FLOOR"))
        floors[key] = whole
        fractions[key] = exact - whole

    # ---- Step 2: count what is left over --------------------------
    remainder = total_units - sum(floors.values())

    # ---- Step 3: hand the leftovers out, one unit at a time -------
    # The sort key is the whole contract. Largest fraction first
    # (negated, because sort is ascending), then key ascending as the
    # documented tie-break. Python's sort is stable, but relying on
    # stability would mean relying on the order the caller happened to
    # build the dict in — so the key is made a total order instead, and
    # the result no longer depends on input ordering at all.
    order = sorted(fractions, key=lambda k: (-fractions[k], k))

    for key in order[:remainder]:
        floors[key] += 1

    return floors


def normalize_weights(raw: Mapping[str, Decimal | int | str]) -> dict[str, Weight]:
    """Turn arbitrary positive proportions into exact weights summing to 1.

    Needed because most real target sets cannot be written down exactly.
    Six equal sleeves are 1/6 each, and 1/6 has no exact decimal form —
    six copies of 0.166667 sum to 1.000002, which allocate() rightly
    refuses.

    So the weights themselves get the same treatment the money does:
    expressed in millionths, apportioned by largest remainder, summing
    to exactly one. The same function, one level up.

        normalize_weights({"a": 1, "b": 1, "c": 1})
        -> a: 0.333334, b: 0.333333, c: 0.333333

    Note the extra millionth lands on "a" by the documented tie-break,
    not by chance.
    """
    if not raw:
        return {}

    values = {key: _to_weight_decimal(key, v) for key, v in raw.items()}

    for key, value in values.items():
        if value < 0:
            raise AllocationError(f"proportion for {key!r} is negative ({value})")

    subtotal = sum(values.values(), Decimal(0))
    if subtotal <= 0:
        raise AllocationError("proportions must include at least one positive value")

    # One over the precision: 0.000001 -> 1_000_000 units.
    units = int(1 / WEIGHT_PRECISION)
    proportions = {key: value / subtotal for key, value in values.items()}

    apportioned = _apportion(units, proportions)
    return {
        key: Weight(Decimal(count) * WEIGHT_PRECISION)
        for key, count in apportioned.items()
    }


def equal_weights(keys: Iterable[str]) -> dict[str, Weight]:
    """Weights that split evenly across `keys` and sum to exactly 1.

        equal_weights(["VTI", "BND"]) -> both 0.5
        equal_weights(["a", "b", "c"]) -> 0.333334, 0.333333, 0.333333

    The uneven last digit is not a bug. It is where the indivisible
    remainder went, and it went somewhere documented.
    """
    return normalize_weights({key: 1 for key in keys})


def _to_weight_decimal(key: str, value: Decimal | int | str) -> Decimal:
    """Accept the same input types as Weight, and reject floats the same way."""
    if isinstance(value, float):
        raise TypeError(
            f"proportion for {key!r} cannot be a float ({value!r}); "
            "use a string or Decimal"
        )
    if isinstance(value, Decimal):
        return value
    return Decimal(value)


def _validate(total: Money, weights: Mapping[str, Weight]) -> None:
    """Reject inputs for which no conserving allocation exists.

    Every check here prevents a silent value leak rather than a crash,
    which is why they are errors and not warnings.
    """
    if not isinstance(total, Money):
        raise TypeError(f"total must be Money, got {type(total).__name__}")

    if not weights:
        # Nothing to allocate to. Fine if there is nothing to allocate;
        # otherwise the money would simply vanish.
        if not total.is_zero:
            raise AllocationError(
                f"cannot allocate {total} across zero weights — "
                "the money would have nowhere to go"
            )
        return

    for key, weight in weights.items():
        if not isinstance(weight, Weight):
            raise TypeError(
                f"weight for {key!r} must be a Weight, got {type(weight).__name__}"
            )
        if weight.value < 0:
            raise AllocationError(
                f"weight for {key!r} is negative ({weight.value}); "
                "negative targets are not meaningful"
            )

    # Exactly 1, not approximately. Weights that sum to 0.9999 would
    # silently orphan a cent on every allocation, and the whole point of
    # this module is that such a thing cannot happen. Callers building
    # weights from division should route them through this same
    # allocator rather than hand-rounding them.
    weight_total = sum((w.value for w in weights.values()), Decimal(0))
    if weight_total != 1:
        raise AllocationError(
            f"weights must sum to exactly 1, got {weight_total}. "
            "If these were computed by division, allocate the rounding "
            "explicitly rather than letting it drift."
        )
