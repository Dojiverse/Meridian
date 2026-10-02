# Meridian

A tax-aware rebalancing, tax-lot, and performance engine for a registered
investment adviser.

It answers the questions an adviser's trading desk asks every month: how
far has this account drifted from its model, which trades bring it back,
which **tax lots** should those trades sell, would any of them trigger a
**wash sale** anywhere in the client's household, is there a **loss worth
harvesting**, does the client's Investment Policy Statement **allow** the
result, and can every one of those decisions be **reproduced exactly**
when an examiner asks three years later.

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

## What it does

| Module | Answers |
|---|---|
| `money`, `allocate` | Exact decimal money, and a division that cannot lose a cent (largest remainder method, deterministic tie-break) |
| `ledger` | Positions as a fold over immutable events; any past date is a replay |
| `model`, `drift` | Versioned target models with absolute or relative bands; drift over the union of held and targeted sleeves |
| `rebalance` | The trades that bring an account back inside its bands, sized to the band edge, with sells chosen **lot by lot** for the least tax |
| `taxlot` | FIFO, LIFO, HIFO, min-tax and specific identification; the one-day holding-period boundary; covered vs non-covered |
| `washsale` | IRC §1091 across the whole household, including the Rev. Rul. 2008-5 trap where a replacement in an IRA forfeits the loss outright, and the 30-day blackout on buying back a harvested loss |
| `harvest` | Turns a harvestable loss into a paired sell and replacement buy that go through the same gate as any other trade |
| `compliance` | The IPS as executable constraints, each violation naming its authority; blocked sells drop the buys they were funding |
| `performance` | Time-weighted and money-weighted returns, gross and net together, Brinson-Fachler attribution, the GIPS refusal to annualise under a year |
| `audit` | A hash-chained, append-only log with as-of replay — and an honest account of what the chain cannot protect without an external anchor |
| `api` | FastAPI plus a single-file advisor blotter. Every amount crosses the wire as a string |
| `showcase` | Two years of a synthetic household, with the engine making every decision, rendered as as-of snapshots |

Regulatory behaviour is checked against IRC §1223 and §1091, Treas. Reg.
§1.1012-1(c), Rev. Rul. 2008-5 and 56-602, Advisers Act Rules 204-2 and
204A-1, SEC Rule 17a-4(f), the Marketing Rule, and GIPS.

## Status

All engine phases are complete: 396 tests including Hypothesis property
suites, `mypy --strict` clean, `ruff` clean.

**[WALKTHROUGH.md](WALKTHROUGH.md)** is the guided tour, phase by phase,
including the bugs the property tests and the generated history found
along the way. Read it first.

Not built: the AI layer (rationale narration, citation-backed extraction),
market-holiday calendars, custodian data feeds, persistence beyond the
in-memory store.

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
pytest                       # tests, including the Hypothesis property suite
mypy                         # strict type checking
ruff check .                 # linting
ruff format .                # formatting

python -m meridian.api       # the blotter, at http://127.0.0.1:8000
python -m meridian.showcase  # two years of history -> showcase/history.json
```

## Layout

```
src/meridian/            the calculation core — pure functions, no I/O
src/meridian/api/        the HTTP layer and the one-file blotter
src/meridian/showcase/   the two-year history generator
tests/                   example tests and Hypothesis property tests
```

The core holds no database connections, reads no clock, and makes no network
calls. Everything it needs arrives as an argument. That is what makes the
property tests possible, and what makes every result reproducible.

## The showcase history

`python -m meridian.showcase` builds a two-account household — a taxable
account and a Roth — and runs it through two years of deposits, dividends,
fees, a withdrawal, and two client-directed gold purchases, with the engine
reviewing both accounts every month: drift, lot-aware rebalancing,
harvesting, the household wash-sale screen, the compliance gate, approval
or rejection, every decision hash-chained.

Nothing in it is hand-written history. Prices are seeded random walks and
the output says so. Every trade was proposed by the engine and either
approved as gated or rejected with the gate's reason. The result is 170
as-of snapshots in one JSON file where every amount is a string, so a page
can stand on any day in the two years and show exactly what the account
held and what the engine said.
