"""Prompt construction.

The system prompt is **generated from the same constants the guardrail
enforces**. Hand-written prompts drift: someone tightens the contact window in
`compliance.py`, forgets the prompt, and the model spends months proposing
actions that are silently vetoed. Deriving the text from the code makes that
impossible.

It is also deliberately one large, byte-stable block so it caches. The
per-case facts go in the user message, after the cache breakpoint.
"""
from __future__ import annotations

from datetime import datetime

from recoup.domain import (
    CHANNEL_COST_PAISE,
    HUMAN_ESCALATION_COST_PAISE,
    RETRY_COST_PAISE,
    Attempt,
    Case,
)
from recoup.policy import stopping
from recoup.policy.base import Context
from recoup.policy.compliance import (
    CONTACT_WINDOW_END_HOUR,
    CONTACT_WINDOW_START_HOUR,
    DLT_TEMPLATES,
    MAX_CONTACTS_PER_7_DAYS,
    MAX_RETRIES_PER_AUTHORIZATION,
    MIN_HOURS_BETWEEN_CONTACTS,
    PRE_DEBIT_NOTICE_HOURS,
)


def _templates_block() -> str:
    lines = []
    for name, kind in sorted(DLT_TEMPLATES.items()):
        lines.append(f"  - {name} ({kind})")
    return "\n".join(lines)


def _costs_block() -> str:
    rows = [f"  - payment retry: {RETRY_COST_PAISE} paise"]
    for ch, paise in CHANNEL_COST_PAISE.items():
        rows.append(f"  - {ch.value}: {paise} paise")
    rows.append(f"  - human escalation: {HUMAN_ESCALATION_COST_PAISE} paise")
    return "\n".join(rows)


def build_system_prompt() -> str:
    """The cacheable prefix. Must not contain anything case-specific."""
    return f"""\
You decide how to recover a single at-risk payment for an Indian payments
company. One case, one decision. You will be called again after the outcome.

## What actually drives recovery

Root cause determines the intervention. Timing matters more than the number of
attempts.

- dead_instrument (card stolen/lost/closed/invalid, mandate revoked): the
  instrument is gone. A retry cannot succeed and breaches card network rules.
  The only path is asking the customer for a new instrument.
- authentication_abandoned (3DS or OTP dropout): the customer walked away
  mid-authentication. A silent re-auth cannot complete a step that needs them
  present, so lead with a fresh payment link. A single retry is still worth
  taking afterwards because it costs 5 paise and lands a few percent of the time.
- insufficient_funds: the account was empty, and it is probably still empty an
  hour later. Retry when money lands. If the case has an inferred salary day,
  aim at it. If not, salary credits in India cluster at month start (1st, 2nd,
  5th, 7th) - bet on that rather than an arbitrary timer.
- issuer_degraded: the bank is down. Retrying into an outage burns one of your
  capped attempts for nothing. Wait, then retry once telemetry shows recovery.
- limit_exceeded: a daily cap. Retry after the calendar day rolls over.
- collect_request_expired: the customer never approved the UPI collect. Switch
  rail to upi_intent or payment_link instead of sending another collect.
- invoice_overdue: usually a deliberate cash-flow decision, not forgetfulness.
  Reminders convert poorly; escalation works but costs real money.

## Economics

Costs per action:
{_costs_block()}

An action is only worth taking if expected recovery clears
{stopping.HURDLE_MULTIPLE}x its cost, where cost includes a nuisance penalty
that grows with the square of how many times this customer has already been
contacted. Retries are nearly free - spend them. Contacts are not - hoard them.
Chasing a small subscription with human escalation destroys value.

Set `p_success` honestly. It feeds the stopping rule directly, so inflating it
makes the system waste money.

## Hard constraints

These are enforced in code after you answer. Proposing something that breaks
them wastes the decision - it will be vetoed, not executed.

- Never retry a dead instrument or a revoked mandate.
- Contact only between {CONTACT_WINDOW_START_HOUR:02d}:00 and
  {CONTACT_WINDOW_END_HOUR:02d}:00 IST. Payment retries may run at any hour -
  a debit is not a contact.
- At most {MAX_RETRIES_PER_AUTHORIZATION} retries per authorization.
- At most {MAX_CONTACTS_PER_7_DAYS} contacts per 7 days, and at least
  {MIN_HOURS_BETWEEN_CONTACTS}h between any two.
- An e-mandate debit requires a PRE_DEBIT_NOTICE sent at least
  {PRE_DEBIT_NOTICE_HOURS}h beforehand. If none has been sent, send it first
  and debit later.
- WhatsApp requires consent. Promotional templates cannot go to a
  DND-registered customer. Opt-out is absolute on every channel.
- Once a promise to pay exists, stay quiet until that date.

Only these templates exist:
{_templates_block()}

## Worked examples

These fix the mistakes that are easiest to make.

**Already inside the liquidity window.** Salary day 1, today is the 2nd,
insufficient funds. The money landed yesterday - retry NOW (`delay_hours: 0`),
do not wait for the next cycle. The window is the salary day and the two days
after it. Waiting three more days spends the best moment you will get.

**Issuer telemetry disagrees with the failure code.** The case says
issuer_down but telemetry shows the bank at 94% success. The outage is over.
Retry promptly (`delay_hours: 0` to 1) - do not sit out an outage that has
already ended. Only hold when telemetry currently reads DEGRADED, and then
re-check in about an hour rather than guessing a recovery time.

**Success rate is not your probability.** A bank at 94% health does not mean
`p_success: 0.94`. That is the issuer's availability, not the odds this
customer's money moves. A healthy-issuer retry after a soft decline is
somewhere near 0.5; an insufficient-funds retry inside the window is near 0.4;
a first reminder on an overdue invoice is near 0.1.

**Retries are a budget, not a button.** You get four per authorization, total.
Spacing them at least 8 hours apart matters more than using them quickly - the
condition that caused the decline (an empty account, a degraded bank) almost
never changes within an hour, so four attempts in one hour is four wasted
attempts and nothing left for the moment that would have worked.

**Prefer a cheap retry to an expensive message.** For collect_request_expired,
switch the rail to upi_intent and retry: 5 paise, and it works far more often
than another message at 35 paise. Reach for a contact when the customer has to
*do* something (supply a new card, complete an abandoned authentication), not
when a different rail would do.

**Dead instrument.** Card reported stolen. Never `retry_payment` - it is a
network rule breach and cannot succeed. Send one message asking for a new
payment method, then stop if there is no response.

**Not worth the money.** A Rs 99 subscription, two reminders already ignored.
Human escalation costs Rs 50 against an expected return of a few rupees.
Answer `stop`. Stopping is a correct, common answer, not a failure.

## Your output

Return one decision. `delay_hours` is measured from the case's current time and
is how you express timing - use it to reach a liquidity window or wait out an
outage. Choose `stop` as soon as further work is not worth its cost; a case
that cannot be won is not a failure, and stopping is the correct answer more
often than it feels.
"""


def build_user_message(
    case: Case, history: list[Attempt], now: datetime, ctx: Context
) -> str:
    """Per-case facts. Everything volatile lives here, after the cache break."""
    lines = [
        "## Case",
        f"type: {case.case_type.value}",
        f"amount: Rs {case.amount_paise / 100:,.2f}",
        f"rail: {case.method.value}",
        f"issuer: {case.issuer}",
        f"failure_code: {case.failure_code.value}",
        f"current_time: {now:%Y-%m-%d %H:%M} IST (day {now.day} of the month)",
        f"age: {(now - case.created_at).total_seconds() / 3600:.1f}h since the failure",
    ]
    if case.due_date is not None:
        lines.append(f"invoice_due: {case.due_date:%Y-%m-%d} "
                     f"({(now - case.due_date).days} days overdue)")

    lines += [
        "",
        "## Customer",
        f"inferred_salary_day: {case.observed_salary_day or 'unknown'}"
        + ("  (inferred from past payments - often missing and sometimes wrong)"
           if case.observed_salary_day else ""),
        f"whatsapp_consent: {case.consent_whatsapp}",
        f"dnd_registered: {case.dnd_registered}",
        f"opted_out: {case.opted_out}",
    ]

    degraded = ctx.issuer_health.is_degraded(case.issuer, now)
    lines += [
        "",
        "## Issuer telemetry",
        f"{case.issuer} success rate right now: "
        f"{ctx.issuer_health.health_score(case.issuer, now):.0%}"
        + ("  - DEGRADED" if degraded else "  - healthy"),
    ]

    lines += ["", "## What we have already tried"]
    if not history:
        lines.append("nothing yet - this is the first decision on this case")
    else:
        for att in history:
            a = att.action
            detail = a.channel.value if a.channel else (a.rail.value if a.rail else "")
            tmpl = f" [{a.template_id}]" if a.template_id else ""
            lines.append(
                f"  {a.at:%d %b %H:%M}  {a.type.value} {detail}{tmpl}"
                f"  -> {'RECOVERED' if att.succeeded else 'no response'}"
            )
        contacts = sum(1 for a in history if a.action.channel is not None)
        retries = sum(1 for a in history if a.action.rail is not None)
        lines.append(f"  ({retries} retries used, {contacts} contacts made)")

    lines += ["", "What is the next action on this case?"]
    return "\n".join(lines)
