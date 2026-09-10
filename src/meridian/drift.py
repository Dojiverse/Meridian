"""Drift — how far a portfolio has wandered from its recipe.

This is the drill, promoted to a domain model. The shape is the same:

    1. total the portfolio
    2. group the holdings by asset class
    3. compare each class against its target

What has been added is everything the drill did not have to worry
about: exact arithmetic, tolerance bands, sleeves the model never
mentioned, sleeves the portfolio does not hold, and a deterministic
order for the answer.

============================================================
THE UNION OF KEYS
============================================================

Two failure modes, both silent, both caught the same way.

  * A sleeve the model TARGETS but the portfolio does not HOLD would
    vanish from a report that only iterated over holdings — the one
    line most in need of attention, missing.

  * A holding the portfolio HAS but the model never mentioned would
    vanish from a report that only iterated over the model — a legacy
    position or an unclassified transfer, invisible.

So the report covers the union of both key sets. Nothing disappears
because of which dictionary it happened to be in.

============================================================
WHERE EXACTNESS STOPS
============================================================

Worth being precise about, because it is the one place in this codebase
where "exact" does not mean exact.

MONEY is exact. Addition and subtraction of Decimals are closed
operations — the answer is always representable, always right, and
`sum(values) == total` holds to the cent forever.

RATIOS are not. `actual = value / total` is a DIVISION, and division is
not closed over decimals: 1/3 has no finite decimal form, so Python
rounds it at 28 significant digits. Summing several such quotients can
therefore land a hair off 1 — around 1e-28, which is roughly a
billion-billion-billionth of a percentage point.

That is not a bug to fix, it is arithmetic to acknowledge. The
alternative — forcing the percentages to sum to exactly 1 by nudging
one of them — would mean REPORTING A NUMBER THAT IS NOT THE ANSWER, to
buy a property nobody needs. Money is what must reconcile. A percentage
is a lens on money, and it is allowed to have a finite resolution.

So: the money assertions in these tests use exact equality; the
percentage assertions use an explicit epsilon, and say why.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from meridian.model import Model, Sleeve
from meridian.money import Money, Weight

__all__ = ["DriftReport", "SleeveDrift", "compute_drift", "group_values"]


def group_values(
    values: Mapping[str, Money],
    classification: Mapping[str, str],
    *,
    unclassified: str = "unclassified",
) -> dict[str, Money]:
    """Sum per-ticker values into per-sleeve values.

    The grouping reduce from the drill, with exact money and one
    addition: a ticker with no classification is not dropped. It lands
    in an `unclassified` bucket, which will then show as an untargeted
    overweight in the drift report.

    Silently dropping it would make money disappear from the total,
    which is the one thing this system is built not to do.
    """
    grouped: dict[str, Money] = {}
    for ticker, value in values.items():
        sleeve = classification.get(ticker, unclassified)
        grouped[sleeve] = grouped.get(sleeve, Money.zero()) + value
    return grouped


@dataclass(frozen=True, slots=True)
class SleeveDrift:
    """One line of the drift report."""

    sleeve: str
    value: Money
    actual: Weight
    target: Weight
    drift: Weight
    band_width: Weight
    breached: bool
    in_model: bool

    @property
    def is_overweight(self) -> bool:
        return self.drift.value > 0

    @property
    def is_underweight(self) -> bool:
        return self.drift.value < 0

    def __str__(self) -> str:
        flag = " BREACH" if self.breached else ""
        sign = "+" if self.drift.value > 0 else ""
        return (
            f"{self.sleeve}: {self.actual.as_percent()} vs "
            f"{self.target.as_percent()} target "
            f"({sign}{self.drift.as_percent()}){flag}"
        )


@dataclass(frozen=True, slots=True)
class DriftReport:
    """Every sleeve, worst drift first."""

    total: Money
    model_id: str
    model_version: int
    sleeves: tuple[SleeveDrift, ...]

    @property
    def breaches(self) -> tuple[SleeveDrift, ...]:
        return tuple(s for s in self.sleeves if s.breached)

    @property
    def needs_rebalancing(self) -> bool:
        """True when any single sleeve has left its band.

        One breach is enough. Rebalancing is a whole-portfolio act —
        bringing one sleeve back necessarily moves the others — so there
        is no such thing as rebalancing only the sleeve that tripped.
        """
        return bool(self.breaches)

    def by_sleeve(self, key: str) -> SleeveDrift | None:
        return next((s for s in self.sleeves if s.sleeve == key), None)

    def __str__(self) -> str:
        header = (
            f"{self.model_id} v{self.model_version} — {self.total} — "
            f"{len(self.breaches)} breach(es)"
        )
        return "\n".join([header, *(f"  {s}" for s in self.sleeves)])


def compute_drift(values: Mapping[str, Money], model: Model) -> DriftReport:
    """Compare a portfolio against its model.

    Args:
        values: Market value per sleeve. Include cash as a sleeve if the
            model targets it; the total is the sum of what is passed.
        model: The versioned target allocation.

    Returns:
        A report covering the union of the value keys and the model
        keys, ordered by absolute drift descending, then sleeve name.

    Ordering is part of the contract, not a convenience. Worst drift
    first is the row a portfolio manager acts on, and the alphabetical
    tie-break means two sleeves that have drifted identically always
    appear in the same order — so the report is reproducible and two
    runs can be diffed.
    """
    total = sum(values.values(), Money.zero())

    keys = set(values) | set(model.sleeves)

    rows = [_row_for(key, values, model, total) for key in keys]

    # Largest absolute drift first, then sleeve name ascending. Same
    # total-order discipline as the allocator's tie-break, for the same
    # reason: an order that depends on dict iteration is an order that
    # changes between runs.
    rows.sort(key=lambda r: (-abs(r.drift.value), r.sleeve))

    return DriftReport(
        total=total,
        model_id=model.model_id,
        model_version=model.version,
        sleeves=tuple(rows),
    )


def _row_for(
    key: str,
    values: Mapping[str, Money],
    model: Model,
    total: Money,
) -> SleeveDrift:
    value = values.get(key, Money.zero())
    sleeve: Sleeve | None = model.sleeve_of(key)

    # ratio_to returns zero for an empty portfolio rather than dividing
    # by zero. Every sleeve is legitimately 0% of nothing.
    actual = value.ratio_to(total)
    target = sleeve.target if sleeve else Weight("0")
    drift = Weight(actual.value - target.value)

    if sleeve is not None:
        band_width = sleeve.band_width
    else:
        # A holding the model never mentioned has no tolerance for
        # existing. Any amount of it is a breach, which is what makes an
        # unclassified or legacy position show up rather than hide.
        band_width = Weight("0")

    breached = abs(drift.value) > band_width.value

    return SleeveDrift(
        sleeve=key,
        value=value,
        actual=actual,
        target=target,
        drift=drift,
        band_width=band_width,
        breached=breached,
        in_model=sleeve is not None,
    )
