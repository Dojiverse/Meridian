# Meridian

A tax-aware rebalancing, tax-lot, and performance engine for a registered
investment adviser.

## The invariants this system guarantees

Not a feature list. These are the properties the code is built to make
impossible to violate:

1. **Value is conserved.** However a total is divided, the parts sum to exactly
   the total. Not within a tolerance — exactly.
2. **Money is never a float.** Every monetary value is exact decimal arithmetic,
   from the database through to the JSON on the wire.
3. **History is append-only.** Positions and cash are derived by replaying an
   immutable event stream. Nothing is edited; corrections are new entries.
4. **The same inputs always produce the same output.** Every decision the engine
   makes is reproducible on demand, years later.

## Status

**Phase 01 — money primitives.** See `Plan.md` in the companion repo for the
full build sequence.

## Setup

Requires Python 3.12+.

```bash
python -m venv .venv
source .venv/Scripts/activate     # Windows (Git Bash)
# source .venv/bin/activate       # macOS / Linux

pip install -e ".[dev]"
```

## Running

```bash
pytest              # tests, including the Hypothesis property suite
mypy                # strict type checking
ruff check .        # linting
ruff format .       # formatting
```

## Layout

```
src/meridian/       the calculation core — pure functions, no I/O
tests/              example tests and Hypothesis property tests
```

The core holds no database connections, reads no clock, and makes no network
calls. Everything it needs arrives as an argument. That is what makes the
property tests possible, and what makes every result reproducible.
