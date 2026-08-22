"""Compliance rules.

Every rule is a pure function of (case, proposed action, history, now). The
LLM strategist may propose anything it likes; nothing reaches the executor
without passing every rule here. That separation is the whole safety story:
the model proposes, deterministic code disposes.

Each rule returns a Verdict whether it passes or fails, because the audit
trail records the full checklist - not just the failures.

Regulatory basis is noted per rule. Verify current text before quoting exact
clauses externally; the numbers below are the operating limits this system
enforces.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from recoup.domain import (
    Action,
    ActionType,
    Attempt,
    Case,
    Channel,
    HARD_DECLINES,
    Method,
)

# RBI guidance on recovery contact restricts calls/visits to 08:00-19:00 local.
CONTACT_WINDOW_START_HOUR = 8
CONTACT_WINDOW_END_HOUR = 19

# Card network rules cap retry attempts against a declined authorization.
MAX_RETRIES_PER_AUTHORIZATION = 4

# Internal contact policy, stricter than the legal floor.
MAX_CONTACTS_PER_7_DAYS = 3
MIN_HOURS_BETWEEN_CONTACTS = 20

# RBI e-mandate framework requires the customer be notified ahead of an
# auto-debit. We enforce a 24h notice before any mandate retry.
PRE_DEBIT_NOTICE_HOURS = 24
PRE_DEBIT_TEMPLATE = "PRE_DEBIT_NOTICE"

# Templates registered on a DLT platform, as TRAI TCCCPR requires. A template
# not in this map cannot be sent at all.
DLT_TEMPLATES: dict[str, str] = {
    "PRE_DEBIT_NOTICE": "transactional",
    "PAYMENT_FAILED_RETRY_LINK": "transactional",
    "MANDATE_BOUNCE_NOTICE": "transactional",
    "INVOICE_OVERDUE_REMINDER": "transactional",
    "INVOICE_OVERDUE_FIRM": "transactional",
    "CHECKOUT_ABANDONED_NUDGE": "promotional",
    "WINBACK_OFFER": "promotional",
}

CONTACT_ACTIONS = {ActionType.SEND_MESSAGE, ActionType.ESCALATE_HUMAN}


@dataclass(frozen=True)
class Verdict:
    rule: str
    allowed: bool
    detail: str = ""


def _contacts(history: list[Attempt]) -> list[Attempt]:
    return [a for a in history if a.action.type in CONTACT_ACTIONS]


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def hard_decline_never_retried(case: Case, action: Action, history, now) -> Verdict:
    """Retrying a stolen/closed/fraud-flagged instrument breaches network rules."""
    rule = "hard_decline_never_retried"
    if action.type is not ActionType.RETRY_PAYMENT:
        return Verdict(rule, True, "not a retry")
    if case.failure_code in HARD_DECLINES:
        return Verdict(rule, False, f"{case.failure_code.value} is a hard decline")
    return Verdict(rule, True)


def contact_window(case: Case, action: Action, history, now: datetime) -> Verdict:
    """RBI: no recovery contact before 08:00 or after 19:00."""
    rule = "rbi_contact_window"
    if action.type not in CONTACT_ACTIONS:
        return Verdict(rule, True, "not a contact")
    if CONTACT_WINDOW_START_HOUR <= now.hour < CONTACT_WINDOW_END_HOUR:
        return Verdict(rule, True, f"{now.hour:02d}:00 is inside the window")
    return Verdict(rule, False, f"{now.hour:02d}:00 is outside 08:00-19:00")


def opt_out_honoured(case: Case, action: Action, history, now) -> Verdict:
    """A customer who said STOP is never contacted again, on any channel."""
    rule = "opt_out_honoured"
    if action.type not in CONTACT_ACTIONS:
        return Verdict(rule, True, "not a contact")
    if case.opted_out:
        return Verdict(rule, False, "customer previously opted out")
    return Verdict(rule, True)


def dnd_promotional_block(case: Case, action: Action, history, now) -> Verdict:
    """TRAI: DND-registered numbers get transactional messages only."""
    rule = "trai_dnd_promotional"
    if action.type is not ActionType.SEND_MESSAGE:
        return Verdict(rule, True, "not a message")
    kind = DLT_TEMPLATES.get(action.template_id or "")
    if kind is None:
        return Verdict(rule, False, f"template {action.template_id!r} is not DLT-registered")
    if case.dnd_registered and kind == "promotional":
        return Verdict(rule, False, "promotional template to a DND-registered customer")
    return Verdict(rule, True, f"{kind} template")


def whatsapp_consent(case: Case, action: Action, history, now) -> Verdict:
    """WhatsApp business messaging requires prior opt-in."""
    rule = "whatsapp_consent"
    if action.type is not ActionType.SEND_MESSAGE or action.channel is not Channel.WHATSAPP:
        return Verdict(rule, True, "not a WhatsApp message")
    if not case.consent_whatsapp:
        return Verdict(rule, False, "no WhatsApp opt-in on record")
    return Verdict(rule, True)


def emandate_pre_debit_notice(case: Case, action: Action, history: list[Attempt], now: datetime) -> Verdict:
    """RBI e-mandate: notify the customer at least 24h before an auto-debit."""
    rule = "emandate_pre_debit_notice"
    if action.type is not ActionType.RETRY_PAYMENT or case.method is not Method.EMANDATE:
        return Verdict(rule, True, "not a mandate debit")
    for att in history:
        if (
            att.action.type is ActionType.SEND_MESSAGE
            and att.action.template_id == PRE_DEBIT_TEMPLATE
            and now - att.action.at >= timedelta(hours=PRE_DEBIT_NOTICE_HOURS)
        ):
            return Verdict(rule, True, f"notice sent {att.action.at:%Y-%m-%d %H:%M}")
    return Verdict(rule, False, "no pre-debit notice served 24h ahead")


def network_retry_cap(case: Case, action: Action, history: list[Attempt], now) -> Verdict:
    """Card networks cap retries against one declined authorization."""
    rule = "network_retry_cap"
    if action.type is not ActionType.RETRY_PAYMENT:
        return Verdict(rule, True, "not a retry")
    used = sum(1 for a in history if a.action.type is ActionType.RETRY_PAYMENT)
    if used >= MAX_RETRIES_PER_AUTHORIZATION:
        return Verdict(rule, False, f"{used} retries already used")
    return Verdict(rule, True, f"{used}/{MAX_RETRIES_PER_AUTHORIZATION} used")


def contact_frequency_cap(case: Case, action: Action, history: list[Attempt], now: datetime) -> Verdict:
    """No more than 3 contacts in 7 days, and never twice inside 20 hours."""
    rule = "contact_frequency_cap"
    if action.type not in CONTACT_ACTIONS:
        return Verdict(rule, True, "not a contact")
    contacts = _contacts(history)
    recent = [a for a in contacts if now - a.action.at < timedelta(days=7)]
    if len(recent) >= MAX_CONTACTS_PER_7_DAYS:
        return Verdict(rule, False, f"{len(recent)} contacts in the last 7 days")
    if contacts:
        gap = now - max(a.action.at for a in contacts)
        if gap < timedelta(hours=MIN_HOURS_BETWEEN_CONTACTS):
            return Verdict(rule, False, f"only {gap.total_seconds() / 3600:.1f}h since last contact")
    return Verdict(rule, True, f"{len(recent)} contacts in the last 7 days")


def dispute_freeze(case: Case, action: Action, history, now) -> Verdict:
    """Once a dispute is open, all recovery activity halts."""
    rule = "dispute_freeze"
    if case.dispute_open and action.type in (CONTACT_ACTIONS | {ActionType.RETRY_PAYMENT}):
        return Verdict(rule, False, "dispute is open on this case")
    return Verdict(rule, True)


RULES = [
    hard_decline_never_retried,
    dispute_freeze,
    opt_out_honoured,
    contact_window,
    dnd_promotional_block,
    whatsapp_consent,
    emandate_pre_debit_notice,
    network_retry_cap,
    contact_frequency_cap,
]


def evaluate(case: Case, action: Action, history: list[Attempt], now: datetime) -> list[Verdict]:
    """Run the full checklist. Returns every verdict, passing or failing."""
    return [rule(case, action, history, now) for rule in RULES]


def is_allowed(verdicts: list[Verdict]) -> bool:
    return all(v.allowed for v in verdicts)


def violations(verdicts: list[Verdict]) -> list[Verdict]:
    return [v for v in verdicts if not v.allowed]
