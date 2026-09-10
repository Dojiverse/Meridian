"""Target models — the recipe a portfolio is supposed to follow.

A model says "55% equity, 35% bonds, 10% alternatives" and, for each
sleeve, how far it may wander before anyone does anything about it.

============================================================
MODELS ARE VERSIONED, NOT EDITED
============================================================

When the investment committee changes the targets, yesterday's
recommendations still have to explain themselves against the targets
that were live when they were made. So a Model carries an id and a
version and is frozen; changing the targets produces a NEW version that
proposals can point at.

Same instinct as the ledger. History is not editable.

============================================================
ABSOLUTE VS RELATIVE BANDS
============================================================

This distinction is not academic, and getting it wrong produces a
portfolio that either never rebalances or never stops.

    A 5-POINT ABSOLUTE band on a 55% target permits 50% - 60%.
    A 5% RELATIVE band on the same target permits 52.25% - 57.75%.

Now apply each to a 2% alternatives sleeve:

    absolute: 0% - 7%    — the sleeve can vanish entirely without
                            tripping. The band never fires.
    relative: 1.9% - 2.1% — fires constantly on noise.

Neither is universally right, which is why real models specify one per
sleeve rather than one for the whole portfolio. Both are supported here
and the choice is stored with the sleeve, not assumed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from meridian.money import Weight

__all__ = ["BandKind", "Model", "ModelError", "Sleeve"]


class ModelError(ValueError):
    """Raised when a model could not be followed as written."""


class BandKind(Enum):
    """How a tolerance band is measured."""

    ABSOLUTE = "absolute"
    """Percentage points either side of target. 55% +/- 5pts -> 50-60%."""

    RELATIVE = "relative"
    """A fraction of the target itself. 55% +/- 5% -> 52.25-57.75%."""


@dataclass(frozen=True, slots=True)
class Sleeve:
    """One line of the recipe: a target and how far it may drift."""

    target: Weight
    tolerance: Weight
    band_kind: BandKind = BandKind.ABSOLUTE

    security: str | None = None
    """What to buy when this sleeve needs money and holds nothing yet.

    A sleeve that already holds something can be topped up in proportion
    to what is there. An EMPTY sleeve gives the engine no way to infer
    which security to buy — so either the model names one here, or the
    trade cannot be generated and is reported as unplaced with a reason.

    Guessing would be worse than refusing. An engine that silently
    picked a security nobody chose is an engine making an investment
    decision it has no authority to make.
    """

    def __post_init__(self) -> None:
        if self.target.value < 0:
            raise ModelError(f"target weight cannot be negative: {self.target}")
        if self.tolerance.value < 0:
            raise ModelError(f"tolerance cannot be negative: {self.tolerance}")

    @property
    def band_width(self) -> Weight:
        """How many percentage points of drift this sleeve tolerates.

        Both band kinds are converted to points here so that everything
        downstream compares like with like. A relative band is a
        fraction OF THE TARGET, so a 5% relative band on a 55% target is
        0.05 x 0.55 = 2.75 points.
        """
        if self.band_kind is BandKind.ABSOLUTE:
            return self.tolerance
        return Weight(self.tolerance.value * self.target.value)

    @property
    def lower_bound(self) -> Weight:
        """The floor of the no-trade region. Never below zero — a sleeve
        cannot hold a negative share of the portfolio."""
        floor = self.target.value - self.band_width.value
        return Weight(max(floor, Decimal(0)))

    @property
    def upper_bound(self) -> Weight:
        return Weight(self.target.value + self.band_width.value)


@dataclass(frozen=True, slots=True)
class Model:
    """A complete, versioned target allocation.

    Targets must sum to exactly 1. A model summing to 0.9999 would
    orphan value on every allocation built from it, and this is the
    cheapest place to catch that — at write time, once, rather than at
    trade time on every account that uses the model.

    Build one with `meridian.allocate.normalize_weights` if the targets
    come from proportions that do not divide evenly.
    """

    model_id: str
    version: int
    sleeves: Mapping[str, Sleeve]
    name: str = ""

    def __post_init__(self) -> None:
        if not self.sleeves:
            raise ModelError(f"model {self.model_id!r} has no sleeves")
        if self.version < 1:
            raise ModelError("model version must be 1 or greater")

        total = sum((s.target.value for s in self.sleeves.values()), Decimal(0))
        if total != 1:
            raise ModelError(
                f"model {self.model_id!r} v{self.version}: targets sum to "
                f"{total}, not exactly 1. Route the raw proportions through "
                "normalize_weights() rather than rounding them by hand."
            )

    @property
    def keys(self) -> frozenset[str]:
        return frozenset(self.sleeves)

    def target_of(self, key: str) -> Weight:
        """The target for a sleeve, or zero for one the model omits.

        Zero rather than an error: a portfolio can legitimately hold
        something the model never mentioned — a legacy position, a
        transfer in, a security that changed classification. That is a
        drift finding to report, not a crash.
        """
        sleeve = self.sleeves.get(key)
        return sleeve.target if sleeve else Weight("0")

    def sleeve_of(self, key: str) -> Sleeve | None:
        return self.sleeves.get(key)

    def __repr__(self) -> str:
        return f"Model({self.model_id!r}, v{self.version}, {len(self.sleeves)} sleeves)"
