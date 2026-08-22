"""The agent's belief about whether an action will work.

These numbers are the agent's *prior*, not the simulator's truth. They were
written from the failure taxonomy, the way a payments engineer would reason
about it - and they are deliberately a bit conservative. The agent is not
allowed to be right by construction.

The estimate feeds one thing: the economic stopping rule. An action whose
expected return does not clear the hurdle never gets proposed.
"""
from __future__ import annotations

from datetime import datetime

from recoup.domain import (
    Action,
    ActionType,
    Attempt,
    Case,
    CaseType,
    Channel,
    FailureCode,
    HARD_DECLINES,
    Method,
    NEEDS_HUMAN_REENGAGEMENT,
)
from recoup.policy.base import Context

CHANNEL_PRIOR = {
    Channel.WHATSAPP: 0.19,
    Channel.SMS: 0.11,
    Channel.EMAIL: 0.05,
    Channel.VOICE: 0.21,
}

MESSAGE_FATIGUE = 0.75
RETRY_FATIGUE = 0.85


def _in_salary_window(now: datetime, salary_day: int | None) -> bool:
    if salary_day is None:
        return False
    return 0 <= (now.day - salary_day) <= 2


def estimate_retry(case: Case, action: Action, now: datetime, ctx: Context, prior_retries: int) -> float:
    code = case.failure_code

    if code in HARD_DECLINES or case.case_type is CaseType.RECEIVABLE_OVERDUE:
        return 0.0
    if code in NEEDS_HUMAN_REENGAGEMENT:
        return 0.05

    if code is FailureCode.INSUFFICIENT_FUNDS:
        if _in_salary_window(now, case.observed_salary_day):
            base = 0.40
        elif case.observed_salary_day is None:
            base = 0.20
        else:
            base = 0.10
    elif code in {FailureCode.ISSUER_DOWN, FailureCode.GATEWAY_TIMEOUT}:
        base = 0.03 if ctx.issuer_health.is_degraded(case.issuer, now) else 0.55
    elif code is FailureCode.LIMIT_EXCEEDED:
        base = 0.40 if now.date() > case.created_at.date() else 0.08
    elif code is FailureCode.UPI_COLLECT_EXPIRED:
        base = 0.34 if action.rail in {Method.UPI_INTENT, Method.PAYMENT_LINK} else 0.18
    else:
        base = 0.15

    return min(base * (RETRY_FATIGUE ** prior_retries), 0.9)


def estimate_message(case: Case, action: Action, now: datetime, prior_messages: int) -> float:
    if case.opted_out or action.channel is None:
        return 0.0
    p = CHANNEL_PRIOR[action.channel] * (MESSAGE_FATIGUE ** prior_messages)
    if case.failure_code in NEEDS_HUMAN_REENGAGEMENT:
        p *= 1.3          # a fresh link is exactly what they need
    if case.failure_code in HARD_DECLINES:
        p *= 0.9          # they must supply a new instrument, so slightly harder
    return min(p, 0.9)


def estimate(
    case: Case, action: Action, history: list[Attempt], now: datetime, ctx: Context
) -> float:
    """Probability this action recovers the money."""
    prior_retries = sum(1 for a in history if a.action.type is ActionType.RETRY_PAYMENT)
    prior_messages = sum(1 for a in history if a.action.type is ActionType.SEND_MESSAGE)

    if action.type is ActionType.RETRY_PAYMENT:
        return estimate_retry(case, action, now, ctx, prior_retries)
    if action.type is ActionType.SEND_MESSAGE:
        return estimate_message(case, action, now, prior_messages)
    if action.type is ActionType.ESCALATE_HUMAN:
        if case.opted_out:
            return 0.0
        return 0.28 if case.case_type is CaseType.RECEIVABLE_OVERDUE else 0.18
    return 0.0
