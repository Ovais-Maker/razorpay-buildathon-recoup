"""Policy interface shared by baselines and the agent.

A policy answers one question: given everything observable about this case
right now, what is the next action and when? It never sees `Latents`, and it
is not responsible for compliance - the guardrail handles that. A policy that
proposes something illegal simply gets vetoed and asked again.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from recoup.domain import Action, Attempt, Case, Channel
from recoup.policy.compliance import (
    CONTACT_WINDOW_END_HOUR,
    CONTACT_WINDOW_START_HOUR,
)
from recoup.sim.issuer_health import IssuerHealth

# We aim for mid-morning: inside the legal window and when people actually read.
PREFERRED_CONTACT_HOUR = 10


@dataclass
class Context:
    """Observable signals a policy may consult."""
    issuer_health: IssuerHealth
    horizon: datetime


class Policy(Protocol):
    name: str
    version: str

    def propose(
        self, case: Case, history: list[Attempt], now: datetime, ctx: Context
    ) -> Action:
        ...


def next_contact_slot(at: datetime, hour: int = PREFERRED_CONTACT_HOUR) -> datetime:
    """Move a timestamp to the next moment a contact is permitted.

    Times already inside the window are left alone - shifting a legal 15:00
    send to tomorrow morning would cost recovery for no compliance gain.
    """
    if CONTACT_WINDOW_START_HOUR <= at.hour < CONTACT_WINDOW_END_HOUR:
        return at
    if at.hour < CONTACT_WINDOW_START_HOUR:
        return at.replace(hour=hour, minute=0, second=0, microsecond=0)
    return (at + timedelta(days=1)).replace(hour=hour, minute=0, second=0, microsecond=0)


def best_channel(case: Case, used: set[Channel]) -> Channel:
    """Cheapest channel that is actually permitted for this customer.

    Ordering is deliberate: WhatsApp first where consent exists (highest
    response per rupee), SMS as the universal fallback, email as the free
    last resort. Voice is never chosen here - it is reserved for explicit
    escalation because it is both expensive and the most intrusive.
    """
    if case.consent_whatsapp and Channel.WHATSAPP not in used:
        return Channel.WHATSAPP
    if Channel.SMS not in used:
        return Channel.SMS
    if Channel.WHATSAPP not in used and case.consent_whatsapp:
        return Channel.WHATSAPP
    return Channel.EMAIL


def day_at(base: datetime, day_of_month: int, hour: int = PREFERRED_CONTACT_HOUR) -> datetime:
    """The next occurrence of `day_of_month` at or after `base`."""
    candidate = base.replace(
        day=min(day_of_month, 28), hour=hour, minute=0, second=0, microsecond=0
    )
    if candidate < base:
        month = base.month + 1
        year = base.year + (month > 12)
        month = month - 12 if month > 12 else month
        candidate = candidate.replace(year=year, month=month)
    return candidate
