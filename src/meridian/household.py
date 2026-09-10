"""Households and accounts.

============================================================
WHY THE HOUSEHOLD IS THE UNIT
============================================================

Almost every rule that matters here follows the TAXPAYER, not the
account. A loss harvested in a taxable brokerage account is disallowed
by a repurchase in the client's IRA, in their spouse's account, or in a
corporation they control — because the IRS attributes all of those to
the same person.

Software organised around accounts cannot see any of that. It reports a
clean harvest, the client's accountant finds the wash sale in April, and
the number the advisor promised was never real.

So the household is the unit of analysis, and an account is a member of
one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum

__all__ = ["Account", "AccountType", "Household"]


class AccountType(Enum):
    """Registration type. Determines tax treatment, and therefore which
    strategies are even meaningful."""

    TAXABLE = "taxable"
    """Ordinary brokerage. The only place harvesting does anything."""

    TRADITIONAL_IRA = "traditional_ira"
    ROTH_IRA = "roth_ira"
    RETIREMENT_PLAN = "retirement_plan"
    """401(k), 403(b), and similar employer plans."""

    TRUST = "trust"
    """Taxable, but a separate taxpayer where the trust is irrevocable —
    a distinction this module does not yet model. Treated as taxable
    for now, and flagged as an open question rather than assumed away."""

    @property
    def is_tax_advantaged(self) -> bool:
        """Whether gains and losses inside the account are invisible to
        the client's current-year tax return.

        Harvesting inside one of these accomplishes nothing: there is no
        loss to deduct because there was never a gain to tax.
        """
        return self in {
            AccountType.TRADITIONAL_IRA,
            AccountType.ROTH_IRA,
            AccountType.RETIREMENT_PLAN,
        }

    @property
    def forfeits_wash_sale_basis(self) -> bool:
        """Whether a wash-sale replacement bought here DESTROYS the loss
        rather than deferring it.

        Rev. Rul. 2008-5. Normally a disallowed loss is added to the
        replacement's basis and comes back when that replacement is
        sold — deferred, not lost. But when the replacement is bought
        inside an IRA, section 1091(d) does not increase the IRA's
        basis, and basis inside an IRA is irrelevant to the taxpayer
        anyway.

        The loss is simply gone. Permanently.

        This is the single most valuable thing a wash-sale engine can
        catch, because the alternative is not a smaller benefit — it is
        a client who lost real money to a bookkeeping accident nobody
        noticed.
        """
        return self in {
            AccountType.TRADITIONAL_IRA,
            AccountType.ROTH_IRA,
            AccountType.RETIREMENT_PLAN,
        }


@dataclass(frozen=True, slots=True)
class Account:
    account_id: str
    account_type: AccountType
    custodian: str = ""
    owner: str = ""
    """Whose account it is. Spouses are separate owners but the same
    household, because section 1091 attributes a spouse's purchases to
    the taxpayer."""

    def __repr__(self) -> str:
        return f"Account({self.account_id!r}, {self.account_type.value})"


@dataclass(frozen=True, slots=True)
class Household:
    """A group of accounts belonging to one taxpayer unit."""

    household_id: str
    accounts: Mapping[str, Account]

    @classmethod
    def of(cls, household_id: str, accounts: Iterable[Account]) -> Household:
        return cls(household_id, {a.account_id: a for a in accounts})

    def type_of(self, account_id: str) -> AccountType:
        account = self.accounts.get(account_id)
        if account is None:
            # Guessing "taxable" would be the dangerous default: it is
            # the one answer under which a wash sale looks merely
            # deferred rather than forfeited.
            raise KeyError(
                f"account {account_id!r} is not in household "
                f"{self.household_id!r} — wash-sale scope cannot be "
                "determined without knowing the registration type"
            )
        return account.account_type

    @property
    def taxable_accounts(self) -> tuple[Account, ...]:
        return tuple(
            a for a in self.accounts.values() if not a.account_type.is_tax_advantaged
        )
