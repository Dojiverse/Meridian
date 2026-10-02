"""Synthetic daily prices — seeded, shaped, and labelled as invented.

A geometric random walk per ticker, in regimes: each regime names an
annual drift and volatility over a date range, so the history can be
given a shape (a spring correction, a gold spike, a collapse) without
anyone typing in prices by hand.

Two properties matter and both are deliberate:

  DETERMINISTIC. One seed per ticker, derived from the ticker's name,
  so the same code always produces the same paths and adding a ticker
  does not disturb the others. A showcase whose numbers changed between
  runs could not be compared against a screenshot taken yesterday.

  QUANTISED AT THE DOOR. The walk is computed in floating point — it is
  a random process, not a claim about money — and each close is rounded
  to a cent and wrapped in `Price` before the engine ever sees it. From
  that point on everything is exact.

Trading days are Monday to Friday. Market holidays are not modelled,
which is the same simplification `taxlot.settlement_date` makes, and
the output says so.
"""

from __future__ import annotations

import math
import random
import zlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from meridian.money import Price

__all__ = [
    "PriceBook",
    "Regime",
    "build_prices",
    "trading_days",
]

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True, slots=True)
class Regime:
    """How a price behaves between two dates, inclusive of `start`."""

    start: date
    annual_drift: float
    annual_volatility: float


@dataclass(frozen=True, slots=True)
class PriceBook:
    """Every ticker's close on every trading day in the range."""

    days: tuple[date, ...]
    closes: Mapping[str, Mapping[date, Price]]

    def on(self, day: date) -> dict[str, Price]:
        """Prices as of `day` — the most recent close on or before it.

        A Saturday has no close of its own; it has Friday's. Valuing an
        account on a non-trading day uses the last known price, which
        is what a custodian statement does too.
        """
        last = self.last_trading_day(day)
        return {ticker: series[last] for ticker, series in self.closes.items()}

    def last_trading_day(self, day: date) -> date:
        if day < self.days[0]:
            raise ValueError(f"{day} precedes the first priced day {self.days[0]}")
        candidate = day
        while candidate not in self.closes[next(iter(self.closes))]:
            candidate -= timedelta(days=1)
        return candidate

    def is_trading_day(self, day: date) -> bool:
        return day in self.closes[next(iter(self.closes))]


def trading_days(start: date, end: date) -> Iterator[date]:
    """Monday to Friday between `start` and `end`, inclusive."""
    day = start
    while day <= end:
        if day.weekday() < 5:
            yield day
        day += timedelta(days=1)


def _seed_for(ticker: str) -> int:
    # CRC32 rather than hash(): Python salts str hashes per process,
    # which would make the "deterministic" walk different every run.
    return zlib.crc32(ticker.encode("utf-8"))


def _walk(
    ticker: str,
    start_price: float,
    regimes: Sequence[Regime],
    days: Sequence[date],
) -> dict[date, Price]:
    rng = random.Random(_seed_for(ticker))
    ordered = sorted(regimes, key=lambda r: r.start)

    closes: dict[date, Price] = {}
    level = start_price
    for day in days:
        regime = ordered[0]
        for candidate in ordered:
            if candidate.start <= day:
                regime = candidate
        mu = regime.annual_drift / TRADING_DAYS_PER_YEAR
        sigma = regime.annual_volatility / math.sqrt(TRADING_DAYS_PER_YEAR)
        # Drift-corrected so the stated annual drift is the expected
        # growth of the price, not of its logarithm.
        step = (mu - 0.5 * sigma * sigma) + sigma * rng.gauss(0.0, 1.0)
        level *= math.exp(step)
        closes[day] = Price(Decimal(f"{level:.2f}"))
    return closes


def _scaled(source: Mapping[date, Price], factor: Decimal) -> dict[date, Price]:
    """A second security tracking the first — substantially identical
    in the most literal sense, which is what the substitute map will
    say about it."""
    return {
        day: Price((price.amount * factor).quantize(Decimal("0.01")))
        for day, price in source.items()
    }


def build_prices(start: date, end: date) -> PriceBook:
    """The showcase's price history.

    The shapes, in plain words:

      VTI   climbs, drops a quarter in spring 2025, then rallies hard
            through the year and cools off in 2026. The rally is what
            pushes equity over its band and makes the rebalancer sell
            winners — where lot choice matters.
      AAPL  the client's former-employer stock. Volatile, strong 2025,
            weak 2026.
      BND   a bond fund. Slow, steady, dull — as it should be.
      GLD   flat, then a spike through autumn 2025, then a collapse
            through spring 2026. A client who buys at the top is left
            holding a loss — the thing the wash-sale screen exists for.
      IAU   GLD at a fifth of the price. The firm's policy treats the
            two as substantially identical.
      SLV   silver: the firm's designated alternative to gold when a
            gold loss is harvested. Its own walk, loosely similar in
            shape — correlated, not identical.
      SCHB  a broad-market fund at a fifth of VTI's price, the
            designated alternative to VTI.
    """
    days = tuple(trading_days(start, end))

    vti = _walk(
        "VTI",
        100.0,
        [
            Regime(start, 0.12, 0.14),
            Regime(date(2025, 3, 1), -0.45, 0.22),
            Regime(date(2025, 5, 1), 0.55, 0.13),
            Regime(date(2026, 1, 1), 0.06, 0.15),
        ],
        days,
    )
    aapl = _walk(
        "AAPL",
        150.0,
        [
            Regime(start, 0.10, 0.24),
            Regime(date(2025, 6, 1), 0.60, 0.26),
            Regime(date(2026, 2, 1), -0.15, 0.28),
        ],
        days,
    )
    bnd = _walk("BND", 50.0, [Regime(start, 0.03, 0.05)], days)
    gld = _walk(
        "GLD",
        160.0,
        [
            Regime(start, 0.08, 0.12),
            Regime(date(2025, 5, 1), 0.60, 0.15),
            Regime(date(2025, 11, 14), -1.20, 0.10),
            Regime(date(2026, 1, 20), -0.45, 0.14),
            Regime(date(2026, 4, 15), 0.02, 0.14),
        ],
        days,
    )
    iau = _scaled(gld, Decimal("0.2"))
    slv = _walk(
        "SLV",
        28.0,
        [
            Regime(start, 0.05, 0.20),
            Regime(date(2025, 5, 1), 0.45, 0.24),
            Regime(date(2025, 11, 14), -0.80, 0.18),
            Regime(date(2026, 1, 20), -0.30, 0.22),
            Regime(date(2026, 4, 15), 0.04, 0.20),
        ],
        days,
    )
    schb = _scaled(vti, Decimal("0.2"))

    return PriceBook(
        days=days,
        closes={
            "VTI": vti,
            "AAPL": aapl,
            "BND": bnd,
            "GLD": gld,
            "IAU": iau,
            "SLV": slv,
            "SCHB": schb,
        },
    )
