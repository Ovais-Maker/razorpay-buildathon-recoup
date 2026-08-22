"""The contract between the model and the rest of the system.

The strategist returns exactly one `RecoveryDecision`. It is never trusted:
`to_action` clamps, validates and rewrites it into a bounded `Action` before
the guardrail even sees it. Two layers of defence, in order:

    LLM  ->  schema (structured output, so the shape is guaranteed)
         ->  adapter (this file - clamps values into legal ranges)
         ->  guardrail (pure rules, holds the veto)
         ->  executor

Every field the model can set is constrained here, so the worst a bad
generation can do is waste one decision, never take an illegal action.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, Field

from recoup.domain import Action, ActionType, Case, Channel, Method
from recoup.policy.base import next_contact_slot
from recoup.policy.compliance import DLT_TEMPLATES

# The model may only wait inside the recovery horizon.
MAX_DELAY_HOURS = 24 * 30

# Rationale is for the audit trail, not an essay.
MAX_REASONING_CHARS = 400

ROOT_CAUSES = [
    "dead_instrument",          # stolen/lost/closed/invalid - never re-authorise
    "authentication_abandoned",  # 3DS or OTP dropout - needs the customer back
    "insufficient_funds",
    "issuer_degraded",
    "limit_exceeded",
    "collect_request_expired",
    "mandate_revoked",
    "invoice_overdue",
    "unknown",
]


class RecoveryDecision(BaseModel):
    """One bounded decision about one case."""

    root_cause: Literal[
        "dead_instrument", "authentication_abandoned", "insufficient_funds",
        "issuer_degraded", "limit_exceeded", "collect_request_expired",
        "mandate_revoked", "invoice_overdue", "unknown",
    ] = Field(description="Diagnosed root cause of the money being at risk.")

    action: Literal[
        "retry_payment", "send_message", "escalate_human", "wait", "stop"
    ] = Field(description="The single next action to take on this case.")

    delay_hours: float = Field(
        ge=0, le=MAX_DELAY_HOURS,
        description="Hours from now to execute. Use this to hit a liquidity "
                    "window or wait out an issuer outage.",
    )

    rail: Literal[
        "card", "upi_collect", "upi_intent", "netbanking", "emandate", "payment_link"
    ] | None = Field(default=None, description="Rail for retry_payment.")

    channel: Literal["sms", "whatsapp", "email", "voice"] | None = Field(
        default=None, description="Channel for send_message."
    )

    template_id: str | None = Field(
        default=None, description="DLT-registered template id for send_message."
    )

    p_success: float = Field(
        ge=0, le=1,
        description="Your probability this action recovers the money. Used by "
                    "the economic stopping rule, so be honest rather than optimistic.",
    )

    # No max_length here on purpose. A length cap in the schema turns an
    # over-long rationale into a hard ValidationError that kills the batch;
    # the adapter truncates instead. Constraints belong where they can be
    # clamped, not where they can throw.
    reasoning: str = Field(
        description="Why this action, at this time. Two sentences at most.",
    )


_ACTION_TYPES = {
    "retry_payment": ActionType.RETRY_PAYMENT,
    "send_message": ActionType.SEND_MESSAGE,
    "escalate_human": ActionType.ESCALATE_HUMAN,
    "wait": ActionType.WAIT,
    "stop": ActionType.STOP,
}

_FALLBACK_TEMPLATE = {
    "mandate_failure": "MANDATE_BOUNCE_NOTICE",
    "receivable_overdue": "INVOICE_OVERDUE_REMINDER",
    "payment_failure": "PAYMENT_FAILED_RETRY_LINK",
}


def to_action(decision: RecoveryDecision, case: Case, now: datetime) -> Action:
    """Convert a model decision into a bounded, executable Action.

    Everything the model could get wrong is corrected here rather than
    trusted: out-of-range delays are clamped, unregistered templates are
    replaced with a registered one, contacts are moved into the legal window,
    and a message with no channel falls back to SMS.
    """
    action_type = _ACTION_TYPES[decision.action]

    delay = max(0.0, min(float(decision.delay_hours), MAX_DELAY_HOURS))
    at = now + timedelta(hours=delay)

    rail = None
    if action_type is ActionType.RETRY_PAYMENT:
        try:
            rail = Method(decision.rail) if decision.rail else case.method
        except ValueError:
            rail = case.method

    channel = None
    template_id = None
    if action_type is ActionType.SEND_MESSAGE:
        try:
            channel = Channel(decision.channel) if decision.channel else Channel.SMS
        except ValueError:
            channel = Channel.SMS
        template_id = decision.template_id
        if template_id not in DLT_TEMPLATES:
            # An unregistered template cannot legally be sent, and inventing
            # one is exactly the kind of thing a generation might do.
            template_id = _FALLBACK_TEMPLATE.get(
                case.case_type.value, "PAYMENT_FAILED_RETRY_LINK"
            )

    # Contacts are pulled into the permitted window here rather than left to
    # be vetoed - a veto costs a whole decision round-trip.
    if action_type in {ActionType.SEND_MESSAGE, ActionType.ESCALATE_HUMAN}:
        at = next_contact_slot(at)

    return Action(
        type=action_type,
        at=at,
        rail=rail,
        channel=channel,
        template_id=template_id,
        reason=decision.reasoning.strip()[:MAX_REASONING_CHARS],
    )
