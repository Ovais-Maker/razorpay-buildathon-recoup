"""Baseline arms.

These exist to make the agent's number mean something. B1 in particular is a
fair, competent opponent - it is what a well-run dunning setup does today,
not a strawman. If the agent cannot beat B1 on incremental recovery, the
project has no result.

  B0  do nothing          - the natural recovery floor
  B1  fixed dunning       - retry at +1h/+24h/+72h, reminders at +2h/+48h
  B2  maximum pressure    - every channel, every few hours, no guardrail
"""
from __future__ import annotations

from datetime import datetime, timedelta

from recoup.domain import (
    Action,
    ActionType,
    Attempt,
    Case,
    CaseType,
    Channel,
    HARD_DECLINES,
    Method,
)
from recoup.policy.base import Context, best_channel, next_contact_slot


class DoNothing:
    """B0. Measures how much money comes back on its own.

    Every other arm is scored against this, which is the difference between
    reporting recovery and reporting *incremental* recovery.
    """
    name = "B0_do_nothing"
    version = "1.0"

    def propose(self, case: Case, history: list[Attempt], now: datetime, ctx: Context) -> Action:
        return Action(ActionType.STOP, now, reason="no intervention by design")


class FixedDunning:
    """B1. The industry default: a fixed retry ladder plus stock reminders.

    Respects hard declines, because that rule is well known. Everything else
    is on a timer - it has no idea whether the issuer is down or whether the
    customer has been paid this month.
    """
    name = "B1_fixed_dunning"
    version = "1.0"

    # (hours after case creation, kind)
    SCHEDULE = [
        (1, "retry"),
        (2, "message"),
        (24, "retry"),
        (48, "message"),
        (72, "retry"),
    ]

    RECEIVABLE_SCHEDULE = [
        (24, "message"),
        (168, "message"),
        (336, "message"),
    ]

    # Mandates need a pre-debit notice 24h ahead, so the ladder is shifted.
    # A competent dunning product does this; leaving it out would make B1
    # structurally unable to retry mandates and rig the comparison.
    MANDATE_SCHEDULE = [
        (1, "notice"),
        (26, "retry"),
        (28, "message"),
        (74, "retry"),
    ]

    def _template(self, case: Case) -> str:
        if case.case_type is CaseType.RECEIVABLE_OVERDUE:
            return "INVOICE_OVERDUE_REMINDER"
        if case.case_type is CaseType.MANDATE_FAILURE:
            return "MANDATE_BOUNCE_NOTICE"
        return "PAYMENT_FAILED_RETRY_LINK"

    def propose(self, case: Case, history: list[Attempt], now: datetime, ctx: Context) -> Action:
        if case.case_type is CaseType.RECEIVABLE_OVERDUE:
            schedule = self.RECEIVABLE_SCHEDULE
        elif case.method is Method.EMANDATE:
            schedule = self.MANDATE_SCHEDULE
        else:
            schedule = self.SCHEDULE
        step = len(history)
        while step < len(schedule):
            offset_h, kind = schedule[step]
            at = case.created_at + timedelta(hours=offset_h)
            if at < now:
                at = now
            if kind == "retry":
                if case.failure_code in HARD_DECLINES:
                    step += 1          # skip retries it knows are pointless
                    continue
                return Action(
                    ActionType.RETRY_PAYMENT, at, rail=case.method,
                    reason=f"fixed ladder step {step + 1}",
                )
            used = {a.action.channel for a in history if a.action.channel}
            template = "PRE_DEBIT_NOTICE" if kind == "notice" else self._template(case)
            return Action(
                ActionType.SEND_MESSAGE,
                next_contact_slot(at),
                # Same channel picker the agent uses. B1 is denied intelligence
                # about timing and root cause, never the ability to reach people.
                channel=best_channel(case, used),
                template_id=template,
                reason=f"fixed ladder step {step + 1}",
            )
        return Action(ActionType.STOP, now, reason="ladder exhausted")


class MaximumPressure:
    """B2. What unbounded recovery looks like.

    Runs with the guardrail switched off: it contacts people at night, ignores
    DND and opt-out, and retries past the network cap. It is here to show that
    raw recovery is the wrong metric - B2 will collect money and still be the
    worst arm once contacts and violations are counted.
    """
    name = "B2_maximum_pressure"
    version = "1.0"
    MAX_ACTIONS = 12

    def propose(self, case: Case, history: list[Attempt], now: datetime, ctx: Context) -> Action:
        step = len(history)
        if step >= self.MAX_ACTIONS:
            return Action(ActionType.STOP, now, reason="pressure exhausted")

        at = now if step == 0 else now + timedelta(hours=6)
        chargeable = case.case_type is not CaseType.RECEIVABLE_OVERDUE

        if step % 2 == 0 and chargeable:
            return Action(
                ActionType.RETRY_PAYMENT, at, rail=case.method,
                reason="retry on a timer, regardless of decline reason",
            )

        rotation = [Channel.SMS, Channel.WHATSAPP, Channel.VOICE, Channel.EMAIL]
        channel = rotation[(step // 2) % len(rotation)]
        return Action(
            ActionType.SEND_MESSAGE, at, channel=channel,
            template_id="WINBACK_OFFER",
            reason="contact on every channel in rotation",
        )
