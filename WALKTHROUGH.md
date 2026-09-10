# Walkthrough

A guided tour of the code, in the order it makes sense to read it.
Written for going over the project afterwards rather than as reference
documentation.

---

## Phase 01 — Money primitives and the conservation proof

**Status: complete.** 45 tests, mypy strict clean, ruff clean.

### Read the files in this order

| # | File | What to look for |
|---|---|---|
| 1 | `src/meridian/money.py` | Why money is a class and not a number |
| 2 | `src/meridian/allocate.py` | The largest remainder method, and the tie-break |
| 3 | `tests/strategies.py` | How the crash-test robot is told what to invent |
| 4 | `tests/test_allocate.py` | The conservation proof |
| 5 | `tests/test_money.py` | The type algebra, edge by edge |

---

### 1. `money.py` — four labelled jars

Four types, not one: `Money`, `Shares`, `Price`, `Weight`.

They exist as separate types because *dollars and share counts are both
"numbers,"* and nothing in Python stops you adding them — which is
meaningless and silent. Splitting them means the mistake gets caught
twice:

```python
Money("10.00") + Shares("5")
```

- **mypy** rejects that line without running anything
- **`TypeError`** catches it at runtime, for values arriving from JSON or
  a database where mypy cannot see them

Two locks. The one to notice is `py.typed` — an empty marker file that
tells other projects "this package has real type annotations." Without
it shipped, every consumer silently loses lock one. I only found that by
testing the claim rather than trusting it.

**The type algebra** is deliberate and closed:

```
Shares x Price  -> Money      what a position is worth
Money  x Weight -> Money      a slice of a portfolio
Money  / Money  -> Weight     what fraction of the total this is
```

`Money * Money` raises on purpose. Dollars-squared is not a unit.

**Things worth pausing on:**

- `_to_decimal` refuses floats outright. `Decimal(0.1)` *succeeds* in
  Python and gives you `0.1000000000000000055511151231257827...` — the
  float's error, faithfully preserved. Rejecting at the door is the only
  safe move.
- `MONEY_ROUNDING = ROUND_HALF_UP` — Python defaults to banker's
  rounding, which sends `0.005` **down** to `0.00`. Right for
  statistics, wrong for a client statement.
- `to_cents()` refuses a fractional cent instead of rounding it away. A
  silent round there is exactly the invisible leak the project exists to
  prevent, so the caller has to decide.
- `Shares.round_to()` always rounds **toward zero**, never nearest.
  Rounding a buy up orders more than the cash can fund; rounding a sell
  up sells shares that are not held. Both bounce at the custodian.

---

### 2. `allocate.py` — dividing money without losing any

The problem, concretely: $10,000 across six equal sleeves is
$1,666.6666… each. Round each one and you get $10,000.02 — **two cents
that do not exist.**

`test_the_naive_approach_would_have_failed` asserts exactly that, so the
problem is demonstrated rather than described.

**The fix — largest remainder method (Hamilton's method):**

1. Convert to whole **cents** (integers cannot carry a fraction that
   needs rounding away — this is the whole trick)
2. Give every claim the cents it definitely earns (floor)
3. Count what is left over
4. Hand the leftovers out one at a time, to whoever the flooring
   shortchanged most

The result sums to the total **by construction**. The code never adds
the parts up afterwards and checks. There is no input for which it comes
out wrong.

**The part that matters most — the tie-break.**

Six equal sleeves means six *identical* fractional parts. Four cents
must go to four of six equal claims, so the tie-break alone decides the
answer. Look at this line:

```python
order = sorted(fractions, key=lambda k: (-fractions[k], k))
```

Largest fraction first, then key ascending. If that were dictionary
iteration order or an unstable sort, the same portfolio would allocate
differently on different runs. Nothing crashes, no error appears — the
books just stop matching last quarter's. That is an **auditability**
bug, not an arithmetic one, and `test_insertion_order_does_not_matter`
is the test that guards it.

**`_apportion` is the shared core.** `allocate()` calls it with cents;
`normalize_weights()` calls it with millionths of a weight; the
quarterly fee split will call it with cents again. One implementation of
"divide something indivisible" — one to get right, one to test, one to
defend.

**`normalize_weights` exists because 1/6 has no exact decimal form.**
Six copies of `0.166667` sum to `1.000002`, which `allocate` rightly
refuses. So the weights get the same treatment the money does: the
algorithm applied one level up.

---

### 3. `tests/strategies.py` — teaching the robot

A *strategy* tells Hypothesis how to invent a value. The interesting one
is `weights()`, which must produce weights summing to **exactly** 1.

Drawing random decimals and dividing by their sum would reintroduce the
very rounding error being tested for. So instead: **cut a stick.**

```
|--------|-----|------------|---|     1,000,000 units
   piece   piece    piece    piece
```

Draw n−1 cut points, take the gaps. Cutting a stick cannot lose length —
which is precisely the property the code under test is supposed to have.

Also note `keys` uses a **three-letter alphabet**. Short, similar keys
make collisions and ties far more likely, and ties are where the
interesting bugs live. Random 20-character strings would almost never
collide.

---

### 4. `tests/test_allocate.py` — the proof

The one to show anyone who asks what this project demonstrates:

```python
@given(total=money(), w=weights())
@settings(max_examples=500)
def test_value_is_conserved(total, w):
    parts = allocate(total, w)
    assert sum(parts.values(), Money.zero()) == total
```

Five hundred generated portfolios per run — negative totals, one-cent
totals, eight sleeves, a single sleeve at 100%, sleeves weighted zero.
The assertion is exact equality, not a tolerance.

The other properties each guard a specific failure:

| Test | Catches |
|---|---|
| `test_signs_match_the_total` | Flooring is asymmetric around zero, so a negative total can produce a wrong-signed part that still sums correctly |
| `test_repeated_calls_agree` | Non-determinism — a player piano, not a jazz pianist |
| `test_insertion_order_does_not_matter` | The tie-break depending on dict order |
| `test_result_is_whole_cents` | Sub-cent amounts surviving into something orderable |

---

### 5. What Phase 01 does *not* do yet

No ledger, no positions, no drift, no trades. Just: **money that cannot
be added to shares, and division that cannot lose a cent.**

Everything above it stands on this, which is why it came first.

---

## Phase 02 — Positions, models, and drift

**Status: complete.** 88 tests total, mypy strict clean, ruff clean.

### Read the files in this order

| # | File | What to look for |
|---|---|---|
| 1 | `src/meridian/ledger.py` | Why nothing is ever edited |
| 2 | `src/meridian/model.py` | Absolute vs relative bands |
| 3 | `src/meridian/drift.py` | The drill, and where exactness stops |
| 4 | `tests/strategies.py` → `event_streams` | Generating *valid* histories |
| 5 | `tests/test_ledger.py` | Cashing up the till |
| 6 | `tests/test_drift.py` | The original drill, asserted |

---

### 1. `ledger.py` — the bank statement

Six event types (`Deposit`, `Withdrawal`, `Buy`, `Sell`, `Dividend`,
`Fee`), all frozen. **The portfolio is never stored** — `fold()` replays
the whole list to compute it, which is exactly the reduce from the
drill, with a portfolio as the accumulator instead of a number.

`replay_to(events, seq)` is the rewind: the portfolio as it stood after
any given event. That is what makes a past report reproducible rather
than merely archived.

**Design calls worth noticing:**

- **Direction lives in the event type, never in the sign.** A negative
  `Deposit` would be a `Withdrawal` that skips the sufficient-funds
  check, so negative amounts are refused outright.
- **No margin, no shorts.** Buying past available cash and selling
  shares not held both raise. Those reconcile fine in software and
  bounce at the custodian — the worst combination.
- **Fees may overdraw.** A custodian charges the account whether the
  cash is there or not. A system that cannot record that cannot
  represent a state the account can really be in.
- **Valuing without a price raises** instead of assuming zero.
  Understating an account silently is the worst way to be wrong.
- The `match` statement over the event union is **checked for
  exhaustiveness by mypy** — add a seventh event type and it points at
  the line that needs updating.

---

### 2. `model.py` — the recipe, versioned

The band distinction is the thing to understand:

|  | 55% target | 2% target |
|---|---|---|
| **5-point absolute** | 50 – 60% | 0 – 7% — *never fires* |
| **5% relative** | 52.25 – 57.75% | 1.9 – 2.1% |

An absolute band on a small sleeve lets it vanish entirely without
tripping. A relative band on the same sleeve fires on noise. Neither is
universally right, so the choice is stored **per sleeve**.

Models are frozen and carry `(model_id, version)`. When the investment
committee changes targets, yesterday's proposals still explain
themselves against the targets that were live when they were made.

---

### 3. `drift.py` — the drill with teeth

`group_values` is your grouping reduce. `compute_drift` is `getDrift`.

Three things the drill did not do:

- **The union of keys.** A targeted sleeve held at zero still appears
  (the row most in need of attention). A holding the model never
  mentioned also appears, with a *zero* band — any amount of something
  untargeted is a breach, which is how a legacy position surfaces
  instead of hiding.
- **Deterministic ordering.** Worst drift first, then sleeve name. Same
  total-order discipline as the allocator's tie-break, so two runs are
  diffable.
- **An honest boundary on exactness** — see below.

---

### 4. Where exactness stops

**Hypothesis caught this**, and it is the most interesting thing in
Phase 02.

I asserted that drifts sum to exactly zero. It shrank a counterexample
down to eight sleeves with one at 100%, and the sum came out
`-1E-29`.

The cause is real. **Money is exact** because addition and subtraction
of Decimals are closed — the answer is always representable.
**Ratios are not**, because `value / total` is a *division*, and
division is not closed over decimals (1/3 has no finite decimal form).
Python rounds at 28 significant digits, so summed quotients can land a
hair off 1.

The tempting fix — nudging one percentage so they sum to exactly 1 —
would mean **reporting a number that is not the answer**, to buy a
property nobody needs.

So the boundary is drawn explicitly instead:

- assertions about **money** use exact equality, and must never be
  relaxed
- assertions about **percentages** use a stated epsilon (`1e-20`, which
  is a billion-billionth of a percentage point) and say why

`test_the_money_in_the_report_is_exactly_the_total` is the exact one.

---

### 5. `event_streams` — generating valid histories

The hardest strategy in the project so far, because **validity is
stateful**: you cannot sell shares you never bought or spend cash you do
not have. Drawing events independently would generate mostly-invalid
streams and the tests would spend their time confirming that guards
fire.

So it draws the stream the way it actually happens — one event at a
time, tracking cash and positions, only ever offering choices that are
legal in the state reached so far.

---

### It works end to end

```
cash left over: $0.00

classic-60-40 v1 — $100,000.00 — 2 breach(es)
  bond: 25.0% vs 35.0% target (-10.0%) BREACH
  alt: 15.0% vs 10.0% target (+5.0%) BREACH
  equity: 60.0% vs 55.0% target (+5.0%)
```

Same numbers the drill produced — now derived from an event history,
computed exactly, and with the bands deciding which rows an advisor
actually has to act on.

---

## Phase 03 — Trade generation

**Status: complete.** 112 tests, mypy strict clean, ruff clean.

One new module: `src/meridian/rebalance.py`. It answers the question
drift only poses — *what trades do I place?* — and returns a `Proposal`.

### The two headline properties

```python
def test_every_proposal_is_applicable(events):
    proposal = propose(events)
    fold([*events, *proposal.to_events(1000)])  # raises on any violation
```

Every generated proposal can be fed straight into the ledger without
overdrawing cash or shorting a position. That is the connective tissue
between Phase 02 and Phase 03: if the rebalancer could produce an order
the *ledger* refuses, it could produce one the *custodian* refuses, and
the first anyone would know is a rejected trade.

The second: **a proposal conserves value.** Selling $5,000 produces
$5,000 of cash. Total unchanged, exactly.

---

### Three bugs this phase found

Worth reading in order — each was caught by a different mechanism.

#### 1. The property tests weren't reaching the dangerous branch

`test_every_proposal_is_applicable` passed instantly. Too instantly. A
quick probe of what the generator was actually producing:

```
sells   23/300  (7%)
buys   298/300  (99%)
```

The generator capped each holding at a quarter of remaining cash, so
every portfolio came out cash-heavy and nearly every rebalance was
buy-only. **The sell path — the one that can overshoot into a short
position — was running in 7% of cases.**

Fixed by drawing a deployment fraction (50–100%) so portfolios are
genuinely invested. Sells went to 67%.

The lesson generalises: *a property test that never reaches the
dangerous branch proves nothing about it.* Measure coverage of the
branch, not just the pass.

#### 2. Band-edge rebalancing was arithmetically incoherent

The first version moved only *breached* sleeves, reasoning that a sleeve
inside its band should be left alone. The demo exposed it: bond needed
$10,000, only $2,000 was placed, and $3,000 came back as "unplaced."

The arithmetic forbids that design. Move bond to 30% and alt to 13%
while equity holds at 60%, and the weights sum to **103%**. No such
portfolio exists.

> **A band decides *whether* to rebalance, not *what* to trade.** Once
> one trips, the whole portfolio participates — because the money has to
> come from somewhere.

`_band_edge_goals` now moves breached sleeves to their near edge and
lets the unbreached sleeves absorb the residual in proportion to their
targets.

#### 3. Aiming at the exact band edge leaves the breach open

After fixing #2, bond landed at **29.99995%** — five cents short of its
30% edge, because share quantities round *down*. Still technically in
breach. And the next review would compute a five-cent delta, decline to
trade it (below `min_trade`), and flag the same sleeve forever.

Aiming precisely at a boundary means rounding decides which side you
land on. So `RebalancePolicy.band_entry` defaults to 10% of band width —
land a little way *inside*, and the rounding has somewhere to go.

`test_aiming_at_the_exact_band_edge_leaves_the_breach_open` reproduces
the trap deliberately with `band_entry=0`, so the reason for the default
is in the test suite rather than in someone's memory.

---

### `Unplaced` means "something you could act on"

After all three fixes, an 11-cent shortfall was still being reported.
Technically honest, practically useless — nobody places an 11-cent
order, and **a report that cries wolf on rounding dust is a report
advisors stop reading**, so they miss the row that mattered.

Every shortfall now routes through `_record_unplaced`, which drops
anything below `min_trade`. Same threshold for "worth trading" and
"worth telling someone about."

---

### Four decisions, each with a cost

1. **How far to trade** — `TO_BAND_EDGE` (less turnover, less tax) vs
   `TO_TARGET` (least tracking error). The demo below shows band-edge
   moving $11k where to-target moves $20k for the same portfolio.
2. **Cash first.** Not a special case — targets are measured against
   total *investable* value, so idle cash shows up as a shortfall across
   the sleeves and gets spent before anything is sold.
3. **Which security within a sleeve** — proportional to what is already
   held, split through the same `allocate()` the money uses everywhere.
   (Phase 04 replaces this for sells, where tax lots make some shares
   much more expensive to sell than others.)
4. **Rounding direction** — always toward zero. A buy rounded up can't
   be funded; a sell rounded up sells shares that aren't held.

---

### It works end to end

```
BEFORE
  bond: 25.0% vs 35.0% target (-10.0%) BREACH
  alt: 15.0% vs 10.0% target (+5.0%) BREACH
  equity: 60.0% vs 55.0% target (+5.0%)

--- to_band_edge --- turnover $10,999.74
  SELL 15.333 GLD    SELL 5.333 AAPL
  SELL 16 VTI        BUY 109.997 BND

AFTER — 0 breaches
  bond: 30.5%   alt: 12.7%   equity: 56.8%
  value $100,000.00 -> $100,000.00
```

Note `to_target` moves $20k to land exactly on 55/35/10, where
`to_band_edge` moves $11k to land inside every band. Both are correct;
the policy chooses which cost to pay.

---

## Phase 04 — Tax lots

**Status: complete.** 148 tests, mypy strict clean, ruff clean.

One new module: `src/meridian/taxlot.py`. **Researched against the
actual regulations before writing** — four things would have been
subtly wrong from memory.

### Why a position is not a number

You bought Apple three times: 2019 at $50, 2022 at $170, last month at
$180. You hold 300 shares. Now you sell 100.

**Which 100?** Nothing about "you own 300 shares" can answer that, and
the answer changes the tax bill enormously. So a position is never
stored as a quantity — it is the sum of open **lots**.

Cartons of milk in a fridge, each with its own date and price sticker.
Selling "some milk" is not a thing; you take specific cartons.

```
method              basis         gain        tax   lots used
fifo            $5,000.00   $12,500.00  $2,975.00   2019-buy(long)
lifo           $18,000.00     -$500.00   -$204.00   2026-buy(short)
hifo           $18,000.00     -$500.00   -$204.00   2026-buy(short)
min_tax        $18,000.00     -$500.00   -$204.00   2026-buy(short)

Same sale, same day. FIFO vs min-tax: $3,179.00 difference.
```

---

### The four rules that needed checking

Each of these I would have implemented wrongly from memory.

#### 1. The holding period boundary — IRC §1223 / Pub. 544

Counting starts the day **after** acquisition, and long-term requires
**more than** one year.

```
bought 2025-03-01, sold 2026-03-01  ->  exactly one year  ->  SHORT
bought 2025-03-01, sold 2026-03-02  ->  LONG
```

The intuitive implementation (`sold - acquired >= 365 days`) gets this
backwards at the boundary. On a $10,000 gain for a high-bracket client
that one day is the difference between **$4,080 and $2,380**.

Leap days too: 29 February has no anniversary in a non-leap year. The
convention is 1 March, so a leap-day purchase doesn't silently qualify a
day early.

#### 2. Specific ID has a deadline — Treas. Reg. §1.1012-1(c)(8)

Choosing which lots to sell is not merely a preference. The
identification must reach the broker **by settlement date**, with
written confirmation. Miss it and the default applies — FIFO for stock.

So `DisposalResult.identification_deadline` is populated for
`SPECIFIC_ID` and `None` for everything else. An engine that lets an
advisor "pick lots" without surfacing that deadline is describing a
choice the taxpayer may not have made in time.

#### 3. Covered vs non-covered — the 1099-B boundary

Brokers report adjusted basis only for **covered** securities: stock
from 2011-01-01, mutual funds and DRIP from 2012-01-01, debt and options
from 2014-01-01. Before those dates the taxpayer reconstructs the basis
and the custodian will not confirm it.

The flag travels with the lot and into the disposal, because treating a
reconstructed basis as authoritative asserts something the system cannot
support.

#### 4. Average cost is *not* a general option

It is available only for regulated investment company (mutual fund)
shares and DRIP shares — never for ordinary stock. So it is deliberately
**not implemented**: offering it as a general method would invite using
it where it is not allowed. That absence is a rule, not an omission.

---

### `_apportion` gets its third caller

`TaxLot.split()` divides cost basis through the **same** `allocate()`
that divides cash and normalises weights:

```python
parts = allocate(
    self.cost_basis,
    normalize_weights({"sold": quantity, "retained": retained}),
)
```

This is the payoff for building one audited "divide something
indivisible." Splitting basis by hand would be a second place for a cent
to go missing — and **a cent of basis lost is a cent of phantom gain
that someone eventually pays tax on.** Same failure as losing cash, just
deferred and much harder to spot.

`test_split_conserves_basis_for_any_lot` runs 400 generated lots against
it.

---

### `MIN_TAX` is derived, not hardcoded

The conventional wisdom is an ordering: short-term losses, then
long-term losses, then long-term gains, then short-term gains.

Rather than encode that list, the method ranks lots by **the actual tax
each share would cost**:

```python
def tax_per_share(lot):
    gain = price - lot.basis_per_share()
    return gain * rates.rate_for(lot.period_at(on))
```

The conventional ordering falls out as a consequence. It also adapts
automatically to a client whose brackets make the usual ordering wrong —
which a hardcoded list cannot.

`MIN_TAX` requires the client's rates and **raises without them**.
Guessing a bracket produces an authoritative-looking number that isn't.

---

### Every sort key ends in `lot_id`

Two lots bought the same day at the same price are interchangeable for
tax purposes — but they must still be *selected* in a stable order, or
the same instruction produces different lot records on different runs
and a past disposal cannot be reproduced.

Same total-order discipline as the allocator's tie-break and the drift
report's ordering. Third time this pattern has earned its place.

---

## Phase 05 — Wash sales and harvesting

**Status: complete.** 188 tests, mypy strict clean, ruff clean.

Two new modules: `household.py` and `washsale.py`. **Researched against
IRC §1091, §1223(3), and Rev. Rul. 2008-5 before writing.**

### The whole phase in one table

```
Sold 100 VOO at a $1,000 loss on 2026-06-15.

Scenario                                 disallowed   deferred   FORFEITED
nothing repurchased                           $0.00      $0.00       $0.00
rebought 100 in taxable 3 days later     -$1,000.00 -$1,000.00       $0.00
rebought 40 in taxable (partial)           -$400.00   -$400.00       $0.00
rebought 100 in the ROTH IRA             -$1,000.00      $0.00  -$1,000.00
spouse rebought 100                      -$1,000.00 -$1,000.00       $0.00
2 shares of DRIP landed in the window       -$20.00    -$20.00       $0.00
50 taxable + 50 IRA                      -$1,000.00   -$500.00    -$500.00
```

Every row is a different rule. The fourth is the one that destroys
client money.

---

### 1. The IRA trap — Rev. Rul. 2008-5

Normally a wash sale **defers** a loss: it is disallowed now, added to
the replacement's basis, and comes back when the replacement is sold.
Annoying, not expensive.

But §1091(d) **does not increase an IRA's basis** — and basis inside an
IRA is irrelevant to the taxpayer anyway. So a replacement bought in an
IRA or Roth means the loss is **permanently forfeited.** Not deferred.
Gone.

An engine that reports every wash sale as a deferral is quietly telling
a client they will get a deduction back that they never will. That is
why `WashSaleReport` exposes `total_forfeited` separately from
`total_deferred`, and why the harvest screen shouts about it:

```
would be washed by the 2026-06-01 purchase in roth-1 — and because
that is a retirement account, the loss would be PERMANENTLY
FORFEITED (Rev. Rul. 2008-5), not deferred
```

---

### 2. The household, not the account

Almost every rule here follows the **taxpayer**. A sale in a taxable
account is washed by a purchase in the spouse's account, in the client's
IRA, or in a corporation they control.

Software organised around accounts cannot see any of that. It reports a
clean harvest, the accountant finds the wash sale in April, and the
number the advisor promised was never real.

`Household.type_of()` **raises** on an unknown account rather than
defaulting to taxable — because taxable is the one answer under which a
wash sale looks merely deferred rather than forfeited. The dangerous
guess is the plausible one.

---

### 3. What the engine refuses to decide

"Substantially identical" has **never been defined** by Congress or the
IRS. It is a facts-and-circumstances test, and two S&P 500 ETFs from
different issuers is genuinely contested — the IRS has not ruled and
practitioners disagree.

So `SubstituteMap` is **curated policy data**, supplied by the firm and
auditable. A firm that decides VOO and IVV are identical and a firm that
decides they are not get the same engine, different answers, and each
can point at the policy that produced its answer.

An algorithm that quietly resolved a contested legal question would be
asserting an authority it does not have — and could not explain itself
when asked why.

The map is also **validated as symmetric**. A one-way map would catch
selling A and buying B but not the reverse, a bug that only ever
surfaces as an inconsistency in someone's tax return.

---

### 4. Holding period tacks backwards

§1223(3): the replacement's holding period **begins on the same day** as
the shares sold. Selling and rebuying does not reset your clock.

```
replacement basis  $9,000.00 -> $10,000.00
acquired date      2026-06-20 -> 2025-01-10   (tacks back)
sold later at $90: realized -$1,000.00 (long-term)
```

The original $1,000 loss reappears exactly, and the sale is long-term
even though the replacement was bought weeks ago. This cuts both ways —
it can hand you treatment you had not earned on the calendar.

`apply_basis_adjustment` **refuses a positive** disallowed amount. The
loss arrives negative, so it is *subtracted* to raise the basis. Getting
that sign backwards halves the basis instead of raising it, which looks
entirely plausible on a screen.

---

### 5. `_apportion` gets its fourth caller

When a loss is split across several replacement purchases, each part
becomes a basis adjustment on a *different lot*. They have to sum to the
disallowed total exactly — a cent lost here is a cent of phantom gain
years later, in a different tax year, in a different report.

Same `allocate()` as cash, weights, and cost basis. Fourth time.

---

### 6. Two subtleties worth noticing

**A replacement share can only absorb one loss.** Without consumption
bookkeeping, a single small repurchase would appear to disallow several
different losses in full. Disposals are processed in date order and
replacement quantity is consumed as it is matched.

**Gains are never washed.** §1091 disallows *losses*. A gain is taxable
now whatever you buy afterwards — and an engine that screened gains
would block trades for no reason.

---

## Phase 06 — Performance and reporting

**Status: complete.** 230 tests, mypy strict clean, ruff clean.

One new module: `src/meridian/performance.py`.

### There are two right answers

A client puts in $1,000 in January and $50,000 in November, right before
a 10% fall.

```
time-weighted    -1.00%   <- was the chef any good?
money-weighted  -44.18%   <- how was the meal I actually ate?
```

Both correct. They answer different questions.

**Time-weighted** cuts the period at every client deposit and withdrawal
and links the pieces geometrically, so the manager is neither blamed nor
credited for *when* money arrived — something they do not control. GIPS
requires it for most composites.

**Money-weighted** is the internal rate of return, and it does reflect
the client's timing. It is what the client feels.

Using the wrong one is a compliance problem, not just an inaccuracy.

---

### Geometric linking is not addition

```
+50% then -50% = -25.00%   (not 0%)
```

Adding period returns is the single most common performance bug, and it
**always flatters the manager** in volatile periods. `test_returns_link_
geometrically_not_additively` is a two-line guard against it.

---

### Gross and net cannot be separated

Marketing Rule 206(4)-1 requires net performance wherever gross is
shown, with equal prominence.

So there is no `return_of(...)` a caller could use for gross and forget
for net. `PerformanceResult` carries **both**, computed in the same call
from the same inputs:

```
2026-01-01 to 2027-01-01 (annualized): 11.00% gross / 10.00% net
fee drag: 1.00%
```

The difference between them is entirely in how fees are treated — an
expense of the strategy for net, added back as an external flow for
gross. Making them one object turns a rule someone must remember into a
shape the code enforces.

---

### The GIPS refusal

```python
annualize(Decimal("0.04"), days=90)
# PerformanceError: refusing to annualise a 89-day period...
```

GIPS prohibits annualising a return for a period of less than one year.
A 4% quarter is a 4% quarter; extrapolating it to 17% invents
performance for nine months that have not happened.

**It raises rather than warns.** A warning in a log is not a control —
it is a thing nobody reads until after the advertisement went out.

---

### Brinson-Fachler, and the free correctness check

```
excess 1.40% = allocation 0.30% + selection 0.95% + interaction 0.15%
  alt:    allocation -0.00%, selection  0.20%, total  0.20%
  bond:   allocation  0.18%, selection -0.35%, total -0.13%
  equity: allocation  0.13%, selection  1.10%, total  1.33%

identity check: 1.40% == 1.40%  ->  True
```

```
Allocation_i  = (wp_i - wb_i) * (Rb_i - Rb)
Selection_i   = wb_i * (Rp_i - Rb_i)
Interaction_i = (wp_i - wb_i) * (Rp_i - Rb_i)
```

**Note the `- Rb` in the allocation term.** That is the *Fachler*
refinement and it is not cosmetic. Brinson-Hood-Beebower uses the raw
sector return, which credits you for overweighting any sector with a
positive return — even one that badly trailed the benchmark.
Subtracting the total benchmark return means you are credited only for
overweighting a sector that **actually beat** the benchmark, which is
what an advisor means when they say the allocation call worked.

The three effects **sum exactly to the excess return**. That identity is
a free property test, and it is the single most valuable one in the
module: getting one of the three formulas wrong is the likeliest bug in
any attribution engine, and otherwise invisible because every number
still looks plausible. `test_the_identity_holds_for_any_inputs` runs 500
generated portfolio/benchmark pairs against it.

---

### XIRR: Newton-Raphson with a bracket it cannot escape

Newton-Raphson converges fast but wanders off on sign-alternating flows.
So it is bracketed by bisection and falls back to it — **a solver that
silently returns a wrong root is worse than one that says it could not
find one.**

The function also refuses flows that all point the same way: an IRR
needs money to go in *and* come out.

---

### The float boundary, drawn one step further

Same distinction as `drift.py`, now with three tiers:

| | Type | Why |
|---|---|---|
| Money | `Decimal` | Exact. Always. |
| Returns (TWR, Dietz) | `Decimal` | Ratios of money; ~1e-28 of division rounding |
| IRR, risk statistics | `float` | Found by iteration, or needs a square root |

An IRR has no exact decimal form and a standard deviation needs
`sqrt`. Wrapping either in `Decimal` would **imply a precision that does
not exist**. `Decimal` for anything on a statement, `float` for anything
on a risk report.

Note that `risk_statistics` returns `None` for benchmark-relative
figures when no benchmark is supplied, rather than zero. **A missing
measurement is not a measurement of zero.**

---

## Phase 07 — Compliance gate and audit

**Status: complete.** 283 tests, mypy strict clean, ruff clean.

Two new modules: `compliance.py` and `audit.py`. Researched against
Rule 17a-4(f)'s 2022 amendments, Rule 204-2, and Rule 204A-1.

---

### 1. The gate — every rejection names its authority

```
2 passed, 2 blocked, 0 warning(s)
  BLOCK GLD: selling GLD at a loss would be washed by the 2026-09-08
    purchase in roth-1, and because that is a retirement account the
    loss would be PERMANENTLY FORFEITED, not deferred [Rev. Rul. 2008-5]
  BLOCK BND: buying BND needs $5,499.85 but only $3,199.94 remains
    after blocked sells were removed
```

**"The system said no" is not an answer** anyone can give a client or an
examiner. So every `Violation` carries a `constraint_id` and an
`authority` — an IPS clause, a rule citation, a firm policy id.

The IPS is executable data, not a PDF. Six constraints ship:
`RestrictedSecurity`, `ConcentrationLimit`, `MinimumCash`,
`ShortTermGainLimit`, `WashSaleBlock`, `PreclearanceRequired`.

**Three design calls worth noticing:**

- **BLOCK vs WARN.** A restricted-list security is a hard stop.
  Realising $400 of short-term gain to close a genuine breach may be the
  right call — an engine that silently refused would be substituting its
  judgment for the advisor's. Both are recorded either way; severity
  only decides what happens next.
- **Severity escalates on wash sales.** A *deferred* loss warns. A loss
  that would be **permanently forfeited** (Rev. Rul. 2008-5) blocks —
  there is no version of that outcome the client wanted.
- **`sell_only` on the restricted list.** The client may not add to a
  prohibited position but may exit one. Forcing someone to hold what
  they have prohibited would be the opposite of the intent.

#### The second block above is the interesting one

That `BND` rejection was not a rule objecting. Blocking the GLD sell
removed cash a buy was relying on, so the buy was dropped too and got
its **own reason**.

Sells fund buys. **A gate that returned an unfundable trade list would
be handing the custodian orders that bounce.** `_ensure_fundable`
re-checks the survivors and drops buys largest-first, and
`test_the_surviving_list_is_always_fundable` asserts the invariant for
every possible block.

---

### 2. The audit log — and the hole Hypothesis found

Every entry contains the hash of the one before it, so editing any past
entry breaks the link the *next* one recorded:

```
someone edits the compliance violation out of entry 2:
  entry 3: previous_hash does not match the entry before it
           — an earlier entry was modified
```

Built to the shape of **Rule 17a-4(f)'s 2022 audit-trail alternative**:
capture modifications, the date and time, the identity of the actor, and
enough to recreate the original. That rule governs broker-dealers, but
204-2(g) deems substantially equivalent records compliant — so building
to the stricter standard satisfies both.

#### The finding

I wrote a property test asserting that tampering with **any** entry is
caught. Hypothesis shrank a counterexample to a **one-entry log**.

**The chain does not protect its own head.** Each entry is protected by
having its hash recorded in the *next* one — so the most recent entry
has nothing pointing at it. Edit it, and the log still verifies against
itself.

```
...and edits only the LAST entry instead:
  self-verification says intact:  True    <- the chain's blind spot
  checked against the anchor:     False
```

This cannot be fixed from inside the log; self-consistency cannot detect
a change to the thing there is nothing to compare against. The fix is an
**anchor** — publish `head` somewhere the same person cannot rewrite,
and pass it back to `verify(expected_head=...)`.

I had described this limitation in prose and never made the code do
anything about it. Now `verify()` takes an optional anchor,
`is_intact_against()` is the complete check, and two property tests
cover both cases separately.

> A security property you believe you have and do not is worse than one
> you know you lack.

Truncation has the same shape: dropping the most recent entries leaves a
perfectly valid chain, and only the anchor reveals it.

#### Other things the chain catches

Actor changes, backdating, deleted middle entries (sequence numbers
*and* the chain both give it away), and payload edits. Corrections are
**appended, never applied** — six hundred years of accounting practice,
and the reason is that on a whiteboard a correction and a cover-up look
identical.

Payload values are **strings**, deliberately. Money serialised as a JSON
number becomes a float the moment anything parses it, and a hash over a
float is a hash over whatever that platform rounded to.

---

### 3. As-of replay — the demo that lands

```
'What did you recommend, and what did the account hold?'
  audit log at 14:30:01: 3 entries, intact=True
  portfolio after event 3: cash $40,000.00, VTI 300 shares
  portfolio after event 5: cash $0.00, VTI 300 shares
  regenerated proposal identical: True
```

`AuditLog.as_of()` plus the ledger's `replay_to()` turns *"reproduce the
recommendation you made in June"* into a query rather than an
archaeology project — and the regenerated proposal is byte-for-byte
identical, which is the payoff for every determinism decision made since
Phase 01.

---

## Phase 08 — Advisor interface

**Status: complete.** 307 tests, mypy strict clean, ruff clean.

```bash
python -m meridian.api      # then open http://127.0.0.1:8000
```

Five new files under `src/meridian/api/`: `schemas.py`, `app.py`,
`store.py`, `demo.py`, and a single-file `static/index.html`.

---

### The rule this whole layer exists to keep

**Every monetary value crosses the wire as a JSON string.**

```json
{"total": "100000.00", "drift": "-0.10", "quantity": "201.5"}
```

JSON has one number type, and every JavaScript engine parses it into an
IEEE-754 double — the same binary float that cannot represent `0.10`. A
monetary value sent as a *number* is silently rounded the moment the
browser calls `JSON.parse`. Nothing errors. The books just stop
matching.

`test_a_string_amount_survives_a_javascript_round_trip` demonstrates it
rather than asserting it, and the comment on that test is worth reading:
a *shorter* amount survives the round trip fine, because Python and
JavaScript both print the shortest string that round-trips. **That is
exactly why the bug is easy to miss in testing and damaging at scale.**

The enforcement is `test_no_money_anywhere_is_a_json_number`, which
walks **every response the API can produce** and fails if any monetary
field comes back as a number. A boundary test, not a unit test — and the
one that stops the discipline eroding one endpoint at a time.

The types carry the rule too:

```python
MoneyStr = Annotated[Money, PlainSerializer(lambda m: str(m.amount), return_type=str)]
```

A new endpoint cannot accidentally emit a raw number, because reaching
for `float` in a schema would be a visible decision rather than an
oversight.

---

### The bug the demo exposed

The first version of the store gathered wash-sale candidates **per
account**. The demo then reported this:

```
[WARN] GLD: ... washed by the 2026-08-20 purchase in taxable-1;
       the loss would be deferred    [IRC 1091]
```

A mild warning. After scoping to the household:

```
[BLOCK] GLD: selling GLD at a loss would be washed by the 2026-09-01
        purchase in roth-1, and because that is a retirement account
        the loss would be PERMANENTLY FORFEITED, not deferred
        [Rev. Rul. 2008-5]
```

**The engine was right; the layer feeding it was not.** §1091 follows
the taxpayer, so a screen that only sees one account reports a clean
harvest that is not clean — and misses the severity difference entirely,
which is real client money.

`test_the_wash_sale_screen_looks_across_the_household` locks it in. It
is the most valuable assertion in the API tests, because the failure it
guards produces *plausible output* rather than an error.

---

### Two more design calls

**Trades and violations come back in ONE response.** A UI that fetched
them from separate endpoints could render a trade list with the blocks
still loading — and an advisor who approves what is on screen has
approved something the gate rejected.

**Approval is refused while blocks stand** (HTTP 422). Overriding a
block is a separate, separately recorded decision — not an approval. And
a decision is made once: re-approving returns 409, because changing it
would mean the record of what was approved is no longer true.

Approving requires an actor with no default of `"system"`, and rejecting
requires a reason. An approval that cannot name who gave it is not an
approval; a rejection with no reason teaches the next reviewer nothing.

---

### Why the UI is one HTML file and not React

The build plan says React + Vite + TypeScript, and for a production
advisor desktop that is right. **It is the wrong call here**, and the
reasoning is in the file's own header:

- this project exists to show the **engine** is correct — the interface
  is a window onto it, and a component tree plus a build config would
  add a second toolchain without demonstrating anything new
- this code is meant to be **read**, and one commented file follows far
  more easily than fifteen across a `src/` tree

Swapping it for React later is a change to the presentation layer
against exactly the same API.

The one rule the page must keep is the same one the server keeps:
**never call `Number()` on money.** Amounts are rendered from the
`*_display` fields the server already formatted, never parsed. The
single `parseFloat` in the file is on a **weight** — a ratio, not money
— and it is commented as such.

`test_the_page_never_calls_number_on_money` enforces it by stripping
comments first. (The first version of that test failed on the file's own
documentation explaining the rule — a test that fails on documentation
is a test that gets deleted.)

---

### It runs

```
GET /             -> 200     the blotter
GET /api/accounts -> 200
GET /docs         -> 200     generated OpenAPI
```

The demo world is two accounts in one household: `taxable-1` drifted off
its model and holding a losing GLD position, and `roth-1` holding a GLD
purchase inside the wash-sale window. Fixed dates, fixed prices —
nothing reads a clock, so a screenshot taken today matches a run
tomorrow.

---

## What is left

Phase 09 — the AI layer (§14 of the plan): rationale narration behind
the numeric fidelity gate, citation-backed extraction, model calls
recorded as ledger events. **Needs an Anthropic API key.**
