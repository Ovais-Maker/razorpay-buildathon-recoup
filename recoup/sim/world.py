"""Outcome resolution.

Given a case, its hidden latents, and an action taken at a moment in time,
decide what happened. This is the only place in the codebase that reads
`Latents` - policies never see it.

Success probabilities are written as explicit tables rather than a fitted
model so that every number is arguable. If a reviewer disagrees with one,
they can change it here and re-run the batch.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from recoup.domain import (
    Action,
    ActionType,
    Attempt,
    Case,
    CaseType,
    FailureCode,
    HARD_DECLINES,
    Latents,
    Method,
    NEEDS_HUMAN_REENGAGEMENT,
)
from recoup.sim.generator import DEAD_INSTRUMENT


class Outcome(str, Enum):
    RECOVERED = "recovered"
    PROMISE_TO_PAY = "promise_to_pay"
    NO_RESPONSE = "no_response"


@dataclass
class ResolveResult:
    outcome: Outcome
    p_success: float
    promise_at: datetime | None = None
    note: str = ""


# A payment retry is machine-side: whether the money moves depends far more on
# the instrument and timing than on how the customer feels. Intent still bites
# a little because a determined non-payer moves money out of the account.
RETRY_INTENT_FACTOR = {
    "will_pay_anyway": 1.05,
    "needs_nudge": 1.00,
    "low_intent": 0.80,
    "wont_pay": 0.25,
}

# A message is human-side, so intent dominates.
MESSAGE_INTENT_FACTOR = {
    "will_pay_anyway": 1.30,
    "needs_nudge": 1.00,
    "low_intent": 0.45,
    "wont_pay": 0.02,
}

RETRY_FATIGUE = 0.85     # each prior retry decays the next one
MESSAGE_FATIGUE = 0.75   # people tune out fast

# Chance that a successful receivables contact yields a promise rather than
# immediate payment, and the chance that promise is actually honoured.
PROMISE_RATE = 0.45
PROMISE_HONOUR_RATE = 0.70


def _u(latents: Latents, seq: int, salt: int = 0) -> float:
    """Draw from the case's pre-allocated uniform stream.

    Indexing by attempt number rather than calling an RNG means every arm of
    the eval sees identical luck for identical decisions.
    """
    stream = latents.u_stream
    return stream[(seq * 2 + salt) % len(stream)]


def _within_liquidity_window(now: datetime, liquidity_day: int | None) -> bool:
    """True if we are on or just after the day money lands in the account."""
    if liquidity_day is None:
        return False
    return 0 <= (now.day - liquidity_day) <= 2


def _retry_probability(
    case: Case, latents: Latents, action: Action, now: datetime, prior_retries: int
) -> tuple[float, str]:
    code = case.failure_code

    if code in HARD_DECLINES or code in DEAD_INSTRUMENT:
        return 0.0, "hard decline - instrument is gone"

    if case.case_type is CaseType.RECEIVABLE_OVERDUE:
        return 0.0, "no stored instrument to charge"

    if code in NEEDS_HUMAN_REENGAGEMENT:
        # The customer abandoned an interactive step. Re-authorising without
        # them present almost never works; they need a fresh link.
        base, why = 0.04, "interactive step abandoned - needs re-engagement, not a retry"

    elif code is FailureCode.INSUFFICIENT_FUNDS:
        if _within_liquidity_window(now, latents.liquidity_day):
            base, why = 0.45, "retried inside the liquidity window"
        elif latents.liquidity_day is None:
            base, why = 0.14, "no known liquidity cycle"
        else:
            base, why = 0.09, "account still dry"

    elif code in {FailureCode.ISSUER_DOWN, FailureCode.GATEWAY_TIMEOUT}:
        recovered = latents.issuer_recovery_at
        if recovered is not None and now < recovered:
            base, why = 0.02, "issuer still degraded"
        else:
            base, why = 0.60, "issuer healthy again"

    elif code is FailureCode.LIMIT_EXCEEDED:
        if now.date() > case.created_at.date():
            base, why = 0.45, "daily limit reset"
        else:
            base, why = 0.07, "still inside the same limit window"

    elif code is FailureCode.UPI_COLLECT_EXPIRED:
        if action.rail in {Method.UPI_INTENT, Method.PAYMENT_LINK}:
            base, why = 0.38, "switched off collect onto a pull-free rail"
        else:
            base, why = 0.20, "another collect request"

    else:
        base, why = 0.15, "generic soft decline"

    p = base * RETRY_INTENT_FACTOR[latents.intent] * (RETRY_FATIGUE ** prior_retries)
    return min(p, 0.95), why


def _message_probability(
    case: Case, latents: Latents, action: Action, now: datetime, prior_messages: int
) -> tuple[float, str]:
    if case.opted_out:
        return 0.0, "customer has opted out"

    channel = action.channel
    if channel is None:
        return 0.0, "no channel specified"

    base = latents.channel_affinity[channel]
    p = base * MESSAGE_INTENT_FACTOR[latents.intent] * (MESSAGE_FATIGUE ** prior_messages)

    # Messages sent at antisocial hours get read late, if at all. The guardrail
    # blocks these outright; the penalty exists so unguarded arms pay for it.
    if not (9 <= now.hour < 19):
        p *= 0.55

    # A fresh link is exactly what an abandoned 3DS/OTP customer needs.
    if case.failure_code in NEEDS_HUMAN_REENGAGEMENT:
        p *= 1.35

    # Overdue B2B invoices are usually a deliberate cash-flow choice rather
    # than forgetfulness, so a reminder moves them much less.
    if case.case_type is CaseType.RECEIVABLE_OVERDUE:
        p *= 0.62

    return min(p, 0.95), f"{channel.value} contact, {prior_messages} prior"


def _escalation_probability(
    case: Case, latents: Latents, prior_messages: int
) -> tuple[float, str]:
    if case.opted_out:
        return 0.0, "customer has opted out"
    base = 0.32 if case.case_type is CaseType.RECEIVABLE_OVERDUE else 0.22
    factor = {
        "will_pay_anyway": 1.2, "needs_nudge": 1.0,
        "low_intent": 0.70, "wont_pay": 0.08,
    }[latents.intent]
    return min(base * factor, 0.9), "human escalation"


def resolve(
    case: Case,
    latents: Latents,
    action: Action,
    history: list[Attempt],
    now: datetime,
) -> ResolveResult:
    """Decide what happens when `action` is executed at `now`."""
    seq = len(history)
    prior_retries = sum(1 for a in history if a.action.type is ActionType.RETRY_PAYMENT)
    prior_messages = sum(1 for a in history if a.action.type is ActionType.SEND_MESSAGE)

    if action.type is ActionType.RETRY_PAYMENT:
        p, why = _retry_probability(case, latents, action, now, prior_retries)
        hit = _u(latents, seq) < p
        return ResolveResult(
            Outcome.RECOVERED if hit else Outcome.NO_RESPONSE, p, note=why
        )

    if action.type is ActionType.SEND_MESSAGE:
        p, why = _message_probability(case, latents, action, now, prior_messages)
        if _u(latents, seq) >= p:
            return ResolveResult(Outcome.NO_RESPONSE, p, note=why)
        # They responded. For an invoice, responding often means committing to
        # a date rather than paying on the spot.
        if case.case_type is CaseType.RECEIVABLE_OVERDUE and _u(latents, seq, 1) < PROMISE_RATE:
            days = 3 + 7 * _u(latents, seq, 3)
            return ResolveResult(
                Outcome.PROMISE_TO_PAY, p,
                promise_at=now + timedelta(days=days),
                note=why + " - promised to pay",
            )
        return ResolveResult(Outcome.RECOVERED, p, note=why)

    if action.type is ActionType.ESCALATE_HUMAN:
        p, why = _escalation_probability(case, latents, prior_messages)
        hit = _u(latents, seq) < p
        return ResolveResult(
            Outcome.RECOVERED if hit else Outcome.NO_RESPONSE, p, note=why
        )

    return ResolveResult(Outcome.NO_RESPONSE, 0.0, note="no-op action")


def promise_honoured(latents: Latents, seq: int) -> bool:
    """Did the customer actually pay on the date they committed to?"""
    return _u(latents, seq, 5) < PROMISE_HONOUR_RATE
