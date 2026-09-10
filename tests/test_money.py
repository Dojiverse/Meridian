"""Tests for the money primitives.

Two themes throughout:

  1. EXACTNESS — Decimal arithmetic, and floats refused at the door.
  2. UNITS — dollars, shares, and prices are different things, and the
     type refuses to pretend otherwise.
"""

from __future__ import annotations

import re
from decimal import Decimal

import pytest
from hypothesis import given

from meridian.money import Money, Price, Shares, Weight
from tests.strategies import money, prices, shares

# ============================================================
# EXACTNESS
# ============================================================


def test_the_problem_this_module_exists_to_solve() -> None:
    assert 0.1 + 0.2 != 0.3
    assert Money("0.10") + Money("0.20") == Money("0.30")


def test_floats_are_refused() -> None:
    """A float would import its own rounding error into exact arithmetic."""
    for cls in (Money, Shares, Price, Weight):
        with pytest.raises(TypeError, match="float"):
            cls(0.1)  # type: ignore[arg-type]


def test_the_error_message_says_what_to_do_instead() -> None:
    with pytest.raises(TypeError, match=re.escape('Money("0.1")')):
        Money(0.1)  # type: ignore[arg-type]


def test_strings_ints_and_decimals_are_accepted() -> None:
    assert Money("10.50").amount == Decimal("10.50")
    assert Money(10).amount == Decimal(10)
    assert Money(Decimal("10.50")).amount == Decimal("10.50")


@given(a=money(), b=money())
def test_addition_is_exact(a: Money, b: Money) -> None:
    assert (a + b).amount == a.amount + b.amount


@given(a=money(), b=money())
def test_add_then_subtract_returns_the_original(a: Money, b: Money) -> None:
    """Round-tripping cannot drift, which is not true of floats."""
    assert (a + b) - b == a


# ============================================================
# UNITS
# ============================================================


def test_money_will_not_absorb_a_bare_number() -> None:
    with pytest.raises(TypeError, match="only Money"):
        Money("10.00") + 5  # type: ignore[operator]


def test_money_will_not_absorb_shares() -> None:
    """The mistake this whole design exists to make impossible.

    mypy rejects this line without running it; the isinstance check
    catches the same error when the value arrives at runtime from JSON
    or a database, where the type checker cannot see it. Two locks.
    """
    with pytest.raises(TypeError, match="cannot add Shares to Money"):
        Money("10.00") + Shares("5")  # type: ignore[operator]


def test_shares_will_not_absorb_money() -> None:
    with pytest.raises(TypeError, match="cannot add Money to Shares"):
        Shares("5") + Money("10.00")  # type: ignore[operator]


def test_money_times_money_is_not_a_thing() -> None:
    """Dollars-squared is not a unit anything in finance is measured in."""
    with pytest.raises(TypeError, match="only by a Weight"):
        Money("10.00") * Money("2.00")  # type: ignore[operator]


def test_the_type_algebra_holds() -> None:
    """The four operations that ARE meaningful."""
    # Shares x Price -> Money
    assert Shares("10").value_at(Price("25.50")) == Money("255.00")

    # Money x Weight -> Money
    assert Money("1000.00") * Weight("0.55") == Money("550.00")

    # Money / Money -> Weight
    assert Money("550.00").ratio_to(Money("1000.00")) == Weight("0.55")


def test_ratio_to_an_empty_portfolio_is_zero_not_an_error() -> None:
    """Every sleeve is legitimately 0% of nothing."""
    assert Money("0.00").ratio_to(Money.zero()) == Weight("0")


# ============================================================
# ROUNDING AND CONVERSION
# ============================================================


def test_rounding_is_half_up_not_bankers() -> None:
    """Python's default would round 0.005 DOWN to 0.00. A client would
    call about that, so the policy is stated explicitly instead."""
    assert Money("0.005").quantize() == Money("0.01")
    assert Money("0.015").quantize() == Money("0.02")
    assert Money("-0.005").quantize() == Money("-0.01")


def test_to_cents_refuses_a_fractional_cent() -> None:
    """A silent round here is exactly the invisible leak to prevent."""
    with pytest.raises(ValueError, match="not a whole number of cents"):
        Money("10.005").to_cents()


def test_to_cents_round_trips() -> None:
    assert Money.from_cents(1234) == Money("12.34")
    assert Money("12.34").to_cents() == 1234
    assert Money.from_cents(-1234) == Money("-12.34")


@given(m=money())
def test_cents_round_trip_for_any_amount(m: Money) -> None:
    assert Money.from_cents(m.to_cents()) == m


# ============================================================
# SHARE ROUNDING — always toward zero
# ============================================================


def test_shares_round_down_never_up() -> None:
    """Rounding a buy up orders more than the cash can fund; rounding a
    sell up sells shares that are not held. Both bounce."""
    assert Shares("53.3535").round_to(Decimal("0.001")) == Shares("53.353")
    assert Shares("53.3999").round_to(Decimal("0.001")) == Shares("53.399")


def test_negative_share_quantities_round_toward_zero_too() -> None:
    assert Shares("-53.3999").round_to(Decimal("0.001")) == Shares("-53.399")


def test_rounding_to_whole_shares() -> None:
    assert Shares("53.99").round_to(Decimal("1")) == Shares("53")


@given(s=shares())
def test_rounding_never_increases_magnitude(s: Shares) -> None:
    """The safety property: rounding can only ever leave you with less."""
    rounded = s.round_to(Decimal("0.001"))
    assert abs(rounded.quantity) <= abs(s.quantity)


@given(s=shares(), p=prices())
def test_valuation_is_exact(s: Shares, p: Price) -> None:
    assert s.value_at(p).amount == s.quantity * p.amount


# ============================================================
# DISPLAY
# ============================================================


def test_money_formats_for_humans() -> None:
    assert str(Money("1234567.891")) == "$1,234,567.89"
    assert str(Money("-1234.5")) == "-$1,234.50"
    assert str(Money.zero()) == "$0.00"


def test_weight_renders_as_a_percent() -> None:
    assert Weight("0.55").as_percent() == "55.0%"
    assert Weight("0.333334").as_percent(2) == "33.33%"


def test_repr_round_trips_through_eval() -> None:
    """A repr you can paste back into a test is a repr worth having."""
    m = Money("1234.56")
    assert eval(repr(m)) == m
