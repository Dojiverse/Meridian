"""Tests for models and drift — the drill, now with teeth."""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given

from meridian.allocate import equal_weights, normalize_weights
from meridian.drift import compute_drift, group_values
from meridian.model import BandKind, Model, ModelError, Sleeve
from meridian.money import Money, Weight
from tests.strategies import money, weights

# The drill's model, as a real object.
CLASSIC = Model(
    model_id="classic-60-40",
    version=1,
    name="Classic balanced",
    sleeves={
        "equity": Sleeve(Weight("0.55"), Weight("0.05")),
        "bond": Sleeve(Weight("0.35"), Weight("0.05")),
        "alt": Sleeve(Weight("0.10"), Weight("0.03")),
    },
)


# ============================================================
# THE DRILL, REPRODUCED
# ============================================================


def test_the_original_drill() -> None:
    """The exact numbers from drill1.js, now computed exactly.

    VTI 42,000 + AAPL 18,000 = 60,000 equity (60%, target 55%, +5pts)
    BND 25,000                          bond (25%, target 35%, -10pts)
    GLD 15,000                          alt  (15%, target 10%, +5pts)
    """
    holdings = {
        "VTI": Money("42000.00"),
        "AAPL": Money("18000.00"),
        "BND": Money("25000.00"),
        "GLD": Money("15000.00"),
    }
    classification = {
        "VTI": "equity",
        "AAPL": "equity",
        "BND": "bond",
        "GLD": "alt",
    }

    by_sleeve = group_values(holdings, classification)
    assert by_sleeve == {
        "equity": Money("60000.00"),
        "bond": Money("25000.00"),
        "alt": Money("15000.00"),
    }

    report = compute_drift(by_sleeve, CLASSIC)

    assert report.total == Money("100000.00")
    assert report.by_sleeve("equity").drift == Weight("0.05")  # type: ignore[union-attr]
    assert report.by_sleeve("bond").drift == Weight("-0.10")  # type: ignore[union-attr]
    assert report.by_sleeve("alt").drift == Weight("0.05")  # type: ignore[union-attr]


def test_worst_drift_comes_first() -> None:
    """Ordering is part of the contract — it is the row someone acts on."""
    report = compute_drift(
        {
            "equity": Money("60000.00"),
            "bond": Money("25000.00"),
            "alt": Money("15000.00"),
        },
        CLASSIC,
    )
    assert [s.sleeve for s in report.sleeves] == ["bond", "alt", "equity"]
    #                                              -10pts  +5    +5
    # alt before equity because the drifts tie at 5 points and "alt"
    # sorts before "equity". Deterministic, not incidental.


def test_breaches_are_flagged_against_their_own_band() -> None:
    report = compute_drift(
        {
            "equity": Money("60000.00"),  # +5pts, band 5 -> not breached
            "bond": Money("25000.00"),  # -10pts, band 5 -> BREACH
            "alt": Money("15000.00"),  # +5pts, band 3 -> BREACH
        },
        CLASSIC,
    )
    assert {s.sleeve for s in report.breaches} == {"bond", "alt"}
    assert report.needs_rebalancing


def test_a_portfolio_on_target_needs_nothing() -> None:
    report = compute_drift(
        {
            "equity": Money("55000.00"),
            "bond": Money("35000.00"),
            "alt": Money("10000.00"),
        },
        CLASSIC,
    )
    assert not report.needs_rebalancing
    assert all(s.drift == Weight("0") for s in report.sleeves)


# ============================================================
# ABSOLUTE VS RELATIVE BANDS
# ============================================================


def test_absolute_and_relative_bands_differ() -> None:
    """The distinction that decides whether a sleeve ever rebalances."""
    absolute = Sleeve(Weight("0.55"), Weight("0.05"), BandKind.ABSOLUTE)
    relative = Sleeve(Weight("0.55"), Weight("0.05"), BandKind.RELATIVE)

    assert absolute.band_width == Weight("0.05")  # 5 points
    assert relative.band_width == Weight("0.0275")  # 5% of 55% = 2.75 points

    assert absolute.lower_bound == Weight("0.50")
    assert absolute.upper_bound == Weight("0.60")
    assert relative.lower_bound == Weight("0.5225")
    assert relative.upper_bound == Weight("0.5775")


def test_an_absolute_band_never_fires_on_a_tiny_sleeve() -> None:
    """A 5-point band on a 2% sleeve permits it to vanish entirely.

    This is the failure mode that motivates supporting both kinds: the
    sleeve most in need of attention is the one the band ignores.
    """
    tiny = Sleeve(Weight("0.02"), Weight("0.05"), BandKind.ABSOLUTE)
    assert tiny.lower_bound == Weight("0")  # gone, and still in band

    tiny_relative = Sleeve(Weight("0.02"), Weight("0.05"), BandKind.RELATIVE)
    assert tiny_relative.band_width == Weight("0.0010")  # 0.1 points


def test_lower_bound_never_goes_negative() -> None:
    """A sleeve cannot hold a negative share of the portfolio."""
    s = Sleeve(Weight("0.02"), Weight("0.10"), BandKind.ABSOLUTE)
    assert s.lower_bound == Weight("0")


# ============================================================
# THE UNION OF KEYS
# ============================================================


def test_a_targeted_sleeve_held_at_zero_still_appears() -> None:
    """The line most in need of attention must not vanish because the
    portfolio happens not to hold it."""
    report = compute_drift(
        {"equity": Money("55000.00"), "bond": Money("45000.00")},
        CLASSIC,
    )
    alt = report.by_sleeve("alt")
    assert alt is not None
    assert alt.value == Money.zero()
    assert alt.actual == Weight("0")
    assert alt.drift == Weight("-0.10")
    assert alt.breached
    assert alt.in_model


def test_a_holding_the_model_never_mentioned_still_appears() -> None:
    """A legacy position or unclassified transfer must be visible.

    It gets a zero band: any amount of something untargeted is a
    breach, because there is no tolerance for holding it at all.
    """
    report = compute_drift(
        {
            "equity": Money("55000.00"),
            "bond": Money("35000.00"),
            "alt": Money("5000.00"),
            "crypto": Money("5000.00"),  # never targeted
        },
        CLASSIC,
    )
    crypto = report.by_sleeve("crypto")
    assert crypto is not None
    assert crypto.target == Weight("0")
    assert crypto.band_width == Weight("0")
    assert crypto.breached
    assert not crypto.in_model


def test_unclassified_holdings_are_bucketed_not_dropped() -> None:
    """Dropping one would make money disappear from the total, which is
    the one thing this system is built not to do."""
    grouped = group_values(
        {"VTI": Money("100.00"), "MYSTERY": Money("50.00")},
        {"VTI": "equity"},
    )
    assert grouped == {"equity": Money("100.00"), "unclassified": Money("50.00")}
    assert sum(grouped.values(), Money.zero()) == Money("150.00")


# ============================================================
# PROPERTIES
# ============================================================


# The arithmetic floor for a ratio. `actual = value / total` is a
# division, which Decimal rounds at 28 significant digits, so quotients
# summed across up to eight sleeves can land ~1e-27 off. This epsilon
# sits far above that floor and still far below any financially
# meaningful quantity: 1e-20 of a percentage point.
#
# Hypothesis found this. The original assertion demanded exact equality
# and it shrank the counterexample to eight sleeves with one at 100%.
RATIO_EPSILON = Decimal("1e-20")


@given(w=weights())
def test_the_money_in_the_report_is_exactly_the_total(
    w: dict[str, Weight],
) -> None:
    """The exact one. Every cent held appears in exactly one row.

    This is the assertion that must never be relaxed — it is about
    money, not about percentages, and money is closed under addition.
    """
    values = {k: Money.from_cents(1000 + i * 137) for i, k in enumerate(w)}
    model = Model("m", 1, {k: Sleeve(v, Weight("0.05")) for k, v in w.items()})
    report = compute_drift(values, model)

    assert sum((s.value for s in report.sleeves), Money.zero()) == report.total


@given(w=weights())
def test_actual_weights_sum_to_one(w: dict[str, Weight]) -> None:
    """The reported percentages account for the whole portfolio.

    To the resolution of division, not exactly — see RATIO_EPSILON.
    """
    values = {k: Money.from_cents(1000 + i * 137) for i, k in enumerate(w)}
    model = Model("m", 1, {k: Sleeve(v, Weight("0.05")) for k, v in w.items()})
    report = compute_drift(values, model)

    total_actual = sum((s.actual.value for s in report.sleeves), Decimal(0))
    assert abs(total_actual - 1) < RATIO_EPSILON


@given(w=weights())
def test_drifts_sum_to_zero(w: dict[str, Weight]) -> None:
    """Overweights and underweights offset.

    A portfolio cannot be net overweight against a model whose targets
    sum to one — every point of excess in one sleeve is a point missing
    from another. If this ever failed by a visible margin, the totals
    and the targets would have come apart.
    """
    values = {k: Money.from_cents(1000 + i * 137) for i, k in enumerate(w)}
    model = Model("m", 1, {k: Sleeve(v, Weight("0.05")) for k, v in w.items()})
    report = compute_drift(values, model)

    assert abs(sum((s.drift.value for s in report.sleeves), Decimal(0))) < RATIO_EPSILON


@given(total=money(), w=weights())
def test_a_portfolio_built_from_the_model_has_no_drift(
    total: Money, w: dict[str, Weight]
) -> None:
    """The round trip between allocate and drift.

    Allocate money according to a model, then measure that money against
    the same model: the drift must be within a cent's worth. Not exactly
    zero — the allocator rounds to whole cents, so a $0.01 portfolio
    across three sleeves genuinely cannot sit on target.
    """
    from meridian.allocate import allocate

    if total.is_negative or total.is_zero:
        return

    values = allocate(total, w)
    model = Model("m", 1, {k: Sleeve(v, Weight("0.05")) for k, v in w.items()})
    report = compute_drift(values, model)

    # One cent misallocated across the whole portfolio is the largest
    # error whole-cent rounding can produce.
    one_cent_of_total = Decimal("0.01") / total.amount
    for row in report.sleeves:
        assert abs(row.drift.value) <= one_cent_of_total


@given(w=weights())
def test_report_ordering_is_deterministic(w: dict[str, Weight]) -> None:
    values = {k: Money.from_cents(1000 + i * 137) for i, k in enumerate(w)}
    model = Model("m", 1, {k: Sleeve(v, Weight("0.05")) for k, v in w.items()})
    first = compute_drift(values, model)
    second = compute_drift(dict(reversed(list(values.items()))), model)
    assert [s.sleeve for s in first.sleeves] == [s.sleeve for s in second.sleeves]


# ============================================================
# MODEL VALIDATION
# ============================================================


def test_targets_must_sum_to_exactly_one() -> None:
    with pytest.raises(ModelError, match="not exactly 1"):
        Model("bad", 1, {"a": Sleeve(Weight("0.5"), Weight("0.05"))})


def test_a_model_needs_at_least_one_sleeve() -> None:
    with pytest.raises(ModelError, match="no sleeves"):
        Model("empty", 1, {})


def test_version_must_be_positive() -> None:
    with pytest.raises(ModelError, match="version must be"):
        Model("m", 0, {"a": Sleeve(Weight("1"), Weight("0.05"))})


def test_negative_targets_are_refused() -> None:
    with pytest.raises(ModelError, match="cannot be negative"):
        Sleeve(Weight("-0.1"), Weight("0.05"))


def test_models_are_immutable() -> None:
    """Yesterday's proposals must still explain themselves against the
    targets that were live when they were made."""
    with pytest.raises(AttributeError):
        CLASSIC.version = 2  # type: ignore[misc]


def test_a_model_can_be_built_from_uneven_proportions() -> None:
    """Six equal sleeves cannot be written down exactly, so they are
    routed through the same allocator the money uses."""
    w = equal_weights(["a", "b", "c", "d", "e", "f"])
    model = Model("six", 1, {k: Sleeve(v, Weight("0.02")) for k, v in w.items()})
    assert sum((s.target.value for s in model.sleeves.values()), Decimal(0)) == 1


def test_a_model_from_plain_percentages() -> None:
    """The realistic path: someone types 60 / 30 / 10."""
    w = normalize_weights({"equity": 60, "bond": 30, "alt": 10})
    model = Model("simple", 1, {k: Sleeve(v, Weight("0.05")) for k, v in w.items()})
    assert model.target_of("equity") == Weight("0.600000")


# ============================================================
# DISPLAY
# ============================================================


def test_a_report_reads_like_something_an_advisor_would_look_at() -> None:
    report = compute_drift(
        {
            "equity": Money("60000.00"),
            "bond": Money("25000.00"),
            "alt": Money("15000.00"),
        },
        CLASSIC,
    )
    text = str(report)
    assert "classic-60-40 v1" in text
    assert "2 breach(es)" in text
    assert "BREACH" in text
