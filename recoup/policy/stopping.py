"""Stopping rules.

A recovery workflow that cannot stop is a harassment machine. These rules are
the bound in "bounded recovery workflow": every case terminates, for a reason
that is recorded.

Two kinds of stop live here:

  * state stops   - something about the case makes further work illegitimate
                    or pointless (hard decline, opt-out, attempts exhausted)
  * economic stop - the next action costs more than it is expected to return
"""
from __future__ import annotations

from datetime import datetime

from recoup.domain import (
    ActionType,
    Attempt,
    Case,
    StopReason,
)

MAX_TOTAL_ACTIONS = 8
MAX_MESSAGES = 5

# Expected recovery must clear this multiple of cost before we act. A hurdle
# above 1.0 keeps us away from marginal actions whose upside is noise.
HURDLE_MULTIPLE = 3.0

# Modelling assumption, stated openly: each additional contact carries a
# brand/annoyance cost beyond its channel price, growing with contact count.
# Set to 0 to disable and see how much more the agent would spend.
NUISANCE_BASE_PAISE = 300


def nuisance_cost_paise(prior_contacts: int) -> int:
    """Escalating cost of bothering someone who is already being bothered."""
    return NUISANCE_BASE_PAISE * (prior_contacts ** 2)


def is_paused(case: Case, now: datetime) -> bool:
    """A recorded promise-to-pay buys the customer quiet until the date lands."""
    return case.promise_to_pay_at is not None and now < case.promise_to_pay_at


def check(
    case: Case,
    history: list[Attempt],
    now: datetime,
    horizon: datetime,
    recovered: bool = False,
) -> StopReason | None:
    """State-based stops. Returns the reason, or None to continue."""
    if recovered:
        return StopReason.RECOVERED
    if case.dispute_open:
        return StopReason.DISPUTE_OPEN
    if case.opted_out:
        return StopReason.CUSTOMER_OPT_OUT
    if case.is_hard_decline and not _has_contactable_path(case):
        return StopReason.HARD_DECLINE
    if now >= horizon:
        return StopReason.HORIZON_REACHED
    if len(history) >= MAX_TOTAL_ACTIONS:
        return StopReason.MAX_ATTEMPTS
    messages = sum(1 for a in history if a.action.type is ActionType.SEND_MESSAGE)
    if messages >= MAX_MESSAGES:
        return StopReason.MAX_ATTEMPTS
    return None


def _has_contactable_path(case: Case) -> bool:
    """A dead instrument still leaves a human worth asking for a new one.

    Hard decline kills the *retry* path, not the case - unless the customer
    has also opted out, in which case there is nothing left to do.
    """
    return not case.opted_out


def is_economic(
    amount_paise: int,
    p_success: float,
    action_cost_paise: int,
    prior_contacts: int = 0,
    hurdle: float = HURDLE_MULTIPLE,
) -> bool:
    """Is the next action worth taking?

    Bites hardest exactly where it should: a Rs 99 subscription is not worth a
    Rs 50 human escalation, and no amount is worth a fifth contact at a 4%
    response rate.
    """
    expected_gain = p_success * amount_paise
    total_cost = action_cost_paise + nuisance_cost_paise(prior_contacts)
    if total_cost <= 0:
        return expected_gain > 0
    return expected_gain >= hurdle * total_cost


def economic_stop_reason(
    amount_paise: int, p_success: float, action_cost_paise: int, prior_contacts: int = 0
) -> str:
    """Human-readable explanation, for the audit trail."""
    expected = p_success * amount_paise
    cost = action_cost_paise + nuisance_cost_paise(prior_contacts)
    return (
        f"expected Rs {expected / 100:.2f} vs hurdle Rs "
        f"{HURDLE_MULTIPLE * cost / 100:.2f} ({HURDLE_MULTIPLE}x cost of Rs {cost / 100:.2f})"
    )
