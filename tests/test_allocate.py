"""The conservation proof.

This is the centrepiece of the project. Everything else in Meridian is
built on the guarantee asserted here: money divided is money conserved.

The property tests are not examples. Each one states a rule that must
hold for EVERY valid input, and Hypothesis spends hundreds of generated
cases per run trying to find the one that breaks it.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from meridian.allocate import (
    AllocationError,
    allocate,
    equal_weights,
    normalize_weights,
)
from meridian.money import Money, Weight
from tests.strategies import money, weights

# ============================================================
# THE INVARIANT
# ============================================================


@given(total=money(), w=weights())
@settings(max_examples=500)
def test_value_is_conserved(total: Money, w: dict[str, Weight]) -> None:
    """However the money is divided, the parts sum to exactly the total.

    Not within a tolerance. Exactly. This single assertion is the reason
    the largest remainder method is used instead of rounding each share
    independently — and it is the first thing to show anyone who asks
    what this project demonstrates.
    """
    parts = allocate(total, w)
    assert sum((m for m in parts.values()), Money.zero()) == total


@given(total=money(), w=weights())
def test_every_key_is_allocated(total: Money, w: dict[str, Weight]) -> None:
    """No sleeve silently disappears, even one weighted zero."""
    assert allocate(total, w).keys() == w.keys()


@given(total=money(), w=weights())
def test_result_is_whole_cents(total: Money, w: dict[str, Weight]) -> None:
    """Nothing sub-cent survives — every part is an orderable amount."""
    for part in allocate(total, w).values():
        assert part.to_cents() == int(part.amount / Decimal("0.01"))


@given(total=money(), w=weights())
def test_signs_match_the_total(total: Money, w: dict[str, Weight]) -> None:
    """A credit divides into credits; a debit divides into debits.

    Guards the negative-total path, where flooring is asymmetric around
    zero and a careless implementation produces a part with the wrong
    sign that still happens to sum correctly.
    """
    for part in allocate(total, w).values():
        if total.is_negative:
            assert not part.amount > 0
        else:
            assert not part.amount < 0


# ============================================================
# DETERMINISM — the tie-break contract
# ============================================================


@given(total=money(), w=weights())
def test_repeated_calls_agree(total: Money, w: dict[str, Weight]) -> None:
    """The same inputs always produce the same output.

    A player piano, not a jazz pianist. If this fails, a past allocation
    can no longer be reproduced, which is an auditability failure rather
    than an arithmetic one.
    """
    assert allocate(total, w) == allocate(total, w)


@given(total=money(), w=weights(min_size=2), data=st.data())
def test_insertion_order_does_not_matter(
    total: Money, w: dict[str, Weight], data: st.DataObject
) -> None:
    """Shuffling the input dict must not change a single cent.

    The sharpest test of the tie-break. If ties were resolved by
    dictionary iteration order, the same portfolio built in a different
    order would allocate differently — silently, with no error, and
    without matching last quarter.
    """
    shuffled_keys = data.draw(st.permutations(list(w.keys())))
    reordered = {k: w[k] for k in shuffled_keys}
    assert allocate(total, reordered) == allocate(total, w)


# ============================================================
# THE WORKED EXAMPLE
# ============================================================


def test_six_equal_sleeves() -> None:
    """$10,000 across six equal sleeves — the case from the plan.

    $10,000 / 6 is $1,666.6666..., so flooring leaves four cents to
    place. Every fractional part is IDENTICAL, so the tie-break alone
    decides who gets them: keys ascending, so the first four
    alphabetically.
    """
    sleeves = ["VTI", "VEA", "IJR", "BND", "GLD", "AGG"]
    result = allocate(Money("10000.00"), equal_weights(sleeves))

    assert result["AGG"] == Money("1666.67")
    assert result["BND"] == Money("1666.67")
    assert result["GLD"] == Money("1666.67")
    assert result["IJR"] == Money("1666.67")
    assert result["VEA"] == Money("1666.66")
    assert result["VTI"] == Money("1666.66")

    assert sum((m for m in result.values()), Money.zero()) == Money("10000.00")


def test_the_naive_approach_would_have_failed() -> None:
    """Proof the problem is real, not theoretical.

    Rounding each share independently — the obvious implementation —
    creates two cents that do not exist.
    """
    naive = [(Decimal("10000.00") / 6).quantize(Decimal("0.01")) for _ in range(6)]
    assert sum(naive) == Decimal("10000.02")  # two cents from nowhere

    correct = allocate(Money("10000.00"), equal_weights([str(i) for i in range(6)]))
    assert sum((m for m in correct.values()), Money.zero()) == Money("10000.00")


def test_a_realistic_model() -> None:
    """55/35/10 across $100,000 — the drill, now exact."""
    model = {
        "equity": Weight("0.55"),
        "bond": Weight("0.35"),
        "alt": Weight("0.10"),
    }
    result = allocate(Money("100000.00"), model)

    assert result["equity"] == Money("55000.00")
    assert result["bond"] == Money("35000.00")
    assert result["alt"] == Money("10000.00")


# ============================================================
# EDGE CASES
# ============================================================


def test_zero_total() -> None:
    result = allocate(Money.zero(), equal_weights(["a", "b", "c"]))
    assert all(m.is_zero for m in result.values())


def test_one_cent_across_three_sleeves() -> None:
    """The smallest interesting case: one indivisible unit, three claims."""
    result = allocate(Money("0.01"), equal_weights(["a", "b", "c"]))
    assert sum((m for m in result.values()), Money.zero()) == Money("0.01")
    assert sorted(m.amount for m in result.values()) == [
        Decimal("0"),
        Decimal("0"),
        Decimal("0.01"),
    ]


def test_negative_total_conserves() -> None:
    """A $100 debit divides exactly like a $100 credit."""
    result = allocate(Money("-100.00"), equal_weights(["a", "b", "c"]))
    assert sum((m for m in result.values()), Money.zero()) == Money("-100.00")


def test_zero_weight_gets_nothing() -> None:
    model = {"a": Weight("1.0"), "b": Weight("0.0")}
    result = allocate(Money("500.00"), model)
    assert result["a"] == Money("500.00")
    assert result["b"] == Money.zero()


def test_empty_weights_with_zero_money_is_fine() -> None:
    assert allocate(Money.zero(), {}) == {}


def test_empty_weights_with_money_is_refused() -> None:
    """Money with nowhere to go must be an error, never a silent drop."""
    with pytest.raises(AllocationError, match="nowhere to go"):
        allocate(Money("100.00"), {})


def test_weights_must_sum_to_one() -> None:
    with pytest.raises(AllocationError, match="sum to exactly 1"):
        allocate(Money("100.00"), {"a": Weight("0.5"), "b": Weight("0.4")})


def test_negative_weight_is_refused() -> None:
    with pytest.raises(AllocationError, match="negative"):
        allocate(Money("100.00"), {"a": Weight("1.5"), "b": Weight("-0.5")})


# ============================================================
# WEIGHT NORMALIZATION
# ============================================================


@given(
    raw=st.dictionaries(
        st.text(alphabet="ABCDEF", min_size=1, max_size=2),
        st.integers(min_value=0, max_value=1000),
        min_size=1,
        max_size=8,
    )
)
def test_normalized_weights_sum_to_exactly_one(raw: dict[str, int]) -> None:
    """The same conservation guarantee, one level up.

    Weights are apportioned by the identical function the money uses,
    so they sum to exactly 1 for any input — which is what lets
    allocate() insist on exactness rather than accepting a tolerance.
    """
    if sum(raw.values()) == 0:
        # All-zero proportions have no meaningful normalization — there
        # is no answer that sums to 1. Assert the refusal rather than
        # skipping, so this branch is genuinely covered.
        with pytest.raises(AllocationError, match="at least one positive"):
            normalize_weights(raw)
        return

    normalized = normalize_weights(raw)
    assert sum((w.value for w in normalized.values()), Decimal(0)) == Decimal(1)


@given(n=st.integers(min_value=1, max_value=30))
def test_equal_weights_sum_to_exactly_one(n: int) -> None:
    """Including n=3, n=6, n=7 — none of which divide evenly."""
    w = equal_weights([f"k{i:02d}" for i in range(n)])
    assert sum((x.value for x in w.values()), Decimal(0)) == Decimal(1)


def test_equal_weights_puts_the_remainder_somewhere_documented() -> None:
    """Three-way split: the extra millionth goes to the first key."""
    w = equal_weights(["a", "b", "c"])
    assert w["a"] == Weight("0.333334")
    assert w["b"] == Weight("0.333333")
    assert w["c"] == Weight("0.333333")


# ============================================================
# THE END-TO-END GUARANTEE
# ============================================================


@given(
    total=money(),
    raw=st.dictionaries(
        st.text(alphabet="ABCDE", min_size=1, max_size=2),
        st.integers(min_value=1, max_value=100),
        min_size=1,
        max_size=6,
    ),
)
@settings(max_examples=300)
def test_normalize_then_allocate_conserves(total: Money, raw: dict[str, int]) -> None:
    """Any proportions, any total, still exact.

    The realistic path: an advisor types "60/30/10" or "1 part each" and
    the money lands on it without a cent going missing.
    """
    parts = allocate(total, normalize_weights(raw))
    assert sum((m for m in parts.values()), Money.zero()) == total
