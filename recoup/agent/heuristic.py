"""Recoup v1 - the deterministic strategist.

This is the arm that has to beat the baselines. Everything it does follows
from one idea: **the right intervention is a function of the root cause**, and
the right *time* matters more than the number of attempts.

Four decisions carry most of the lift:

  1. Never re-authorise a dead instrument or an abandoned 3DS step. Ask the
     human for a fresh one instead.
  2. Park cases on a degraded issuer and retry when telemetry says the bank is
     healthy, rather than firing into an outage on a timer.
  3. Retry insufficient-funds cases inside the customer's liquidity window,
     not one hour after the decline when the account is still empty.
  4. Stop as soon as the next action stops paying for itself.

A later version replaces `_plan` with an LLM call returning the same Action
object. The guardrail, the economics and the audit trail do not change - which
is the point of putting them outside the strategist.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from recoup.agent.estimator import estimate
from recoup.domain import (
    Action,
    ActionType,
    Attempt,
    Case,
    CaseType,
    FailureCode,
    HARD_DECLINES,
    Method,
    NEEDS_HUMAN_REENGAGEMENT,
)
from recoup.policy import stopping
from recoup.policy.base import Context, best_channel, day_at, next_contact_slot
from recoup.policy.compliance import (
    PRE_DEBIT_NOTICE_HOURS,
    PRE_DEBIT_TEMPLATE,
)

# Retries cost 5 paise; a contact costs 20-35 plus escalating nuisance. So the
# agent spends its retry budget freely up to the network cap and is miserly
# with contacts - the opposite of a fixed ladder, which spreads both evenly.
MAX_RETRIES = 4
MAX_MESSAGES = 3

# Salary credits in India cluster hard at month start. With no per-customer
# signal, betting on the population prior beats retrying on an arbitrary timer.
POPULATION_SALARY_DAYS = (1, 2, 5, 7, 10, 15)
BLIND_TARGET_MAX_WAIT_DAYS = 8
UNKNOWN_LIQUIDITY_DELAY_DAYS = 2

# Two debits in the same instant is not a retry strategy. Spacing them still
# lets several attempts land inside a three-day liquidity window.
MIN_RETRY_GAP_HOURS = 8


class RecoupHeuristic:
    name = "B3_recoup_agent"
    version = "v1-heuristic"

    # ------------------------------------------------------------------
    # entry point
    # ------------------------------------------------------------------
    def propose(self, case: Case, history: list[Attempt], now: datetime, ctx: Context) -> Action:
        if stopping.is_paused(case, now):
            return Action(
                ActionType.WAIT, case.promise_to_pay_at,
                reason="promise to pay recorded - staying quiet until the date",
            )

        action = self._plan(case, history, now, ctx)

        # Economic gate. Waiting and stopping are free, so they skip it.
        if action.type in {ActionType.RETRY_PAYMENT, ActionType.SEND_MESSAGE,
                           ActionType.ESCALATE_HUMAN}:
            p = estimate(case, action, history, action.at, ctx)
            contacts = sum(
                1 for a in history
                if a.action.type in {ActionType.SEND_MESSAGE, ActionType.ESCALATE_HUMAN}
            )
            if not stopping.is_economic(case.amount_paise, p, action.cost_paise(), contacts):
                return Action(
                    ActionType.STOP, now,
                    reason="not economic: " + stopping.economic_stop_reason(
                        case.amount_paise, p, action.cost_paise(), contacts
                    ),
                )
        return action

    # ------------------------------------------------------------------
    # planning
    # ------------------------------------------------------------------
    def _plan(self, case: Case, history: list[Attempt], now: datetime, ctx: Context) -> Action:
        if case.case_type is CaseType.RECEIVABLE_OVERDUE:
            return self._plan_receivable(case, history, now)
        return self._plan_chargeable(case, history, now, ctx)

    # --- invoices -----------------------------------------------------
    def _plan_receivable(self, case: Case, history: list[Attempt], now: datetime) -> Action:
        messages = [a for a in history if a.action.type is ActionType.SEND_MESSAGE]
        escalated = any(a.action.type is ActionType.ESCALATE_HUMAN for a in history)
        used = {a.action.channel for a in messages if a.action.channel}

        # A case handed to a human is out of the agent's hands. Without this
        # the sequence below re-escalates forever, because an escalation is
        # not a SEND_MESSAGE and never advances the message count.
        if escalated:
            return Action(ActionType.STOP, now, reason="already with a collections owner")

        if not messages:
            return Action(
                ActionType.SEND_MESSAGE, next_contact_slot(now),
                channel=best_channel(case, used),
                template_id="INVOICE_OVERDUE_REMINDER",
                reason="first reminder on the overdue invoice",
            )
        if len(messages) == 1:
            return Action(
                ActionType.SEND_MESSAGE, next_contact_slot(now + timedelta(days=3)),
                channel=best_channel(case, used),
                template_id="INVOICE_OVERDUE_FIRM",
                reason="no response to the first reminder - escalating tone",
            )
        if len(messages) == 2:
            return Action(
                ActionType.ESCALATE_HUMAN, next_contact_slot(now + timedelta(days=4)),
                reason="two reminders ignored - hand to a collections owner",
            )
        return Action(ActionType.STOP, now, reason="reminder sequence exhausted")

    # --- payments and mandates ---------------------------------------
    def _plan_chargeable(
        self, case: Case, history: list[Attempt], now: datetime, ctx: Context
    ) -> Action:
        retries = sum(1 for a in history if a.action.type is ActionType.RETRY_PAYMENT)
        messages = [a for a in history if a.action.type is ActionType.SEND_MESSAGE
                    and a.action.template_id != PRE_DEBIT_TEMPLATE]
        used = {a.action.channel for a in messages if a.action.channel}
        code = case.failure_code

        # 1. Dead instrument. Retrying is a compliance breach and pointless
        #    anyway - the only path left runs through the customer.
        if code in HARD_DECLINES:
            if len(messages) >= 2:
                return Action(ActionType.STOP, now, reason="instrument gone, customer unresponsive")
            delay = timedelta(0) if not messages else timedelta(days=3)
            template = ("MANDATE_BOUNCE_NOTICE"
                        if code is FailureCode.MANDATE_REVOKED
                        else "PAYMENT_FAILED_RETRY_LINK")
            return Action(
                ActionType.SEND_MESSAGE, next_contact_slot(now + delay),
                channel=best_channel(case, used), template_id=template,
                reason=f"{code.value}: ask for a new instrument, never re-authorise",
            )

        # 2. The customer walked away mid-authentication. Re-engagement is the
        #    primary move - but a re-auth still lands a few percent of the time
        #    and costs 5 paise, so refusing it outright leaves money on the
        #    table. Lead with the link, take the cheap retry as a second shot.
        if code in NEEDS_HUMAN_REENGAGEMENT:
            if not messages:
                return Action(
                    ActionType.SEND_MESSAGE, next_contact_slot(now),
                    channel=best_channel(case, used),
                    template_id="PAYMENT_FAILED_RETRY_LINK",
                    reason=f"{code.value}: fresh link first - a silent re-auth "
                           f"cannot complete a step that needs the customer",
                )
            if retries < 1:
                return Action(
                    ActionType.RETRY_PAYMENT, now + timedelta(hours=6),
                    rail=case.method,
                    reason="cheap second shot: a minority of abandoned "
                           "authentications do complete on re-auth",
                )
            if len(messages) < 2:
                return Action(
                    ActionType.SEND_MESSAGE, next_contact_slot(now + timedelta(days=2)),
                    channel=best_channel(case, used),
                    template_id="PAYMENT_FAILED_RETRY_LINK",
                    reason="final re-engagement attempt",
                )
            return Action(ActionType.STOP, now, reason="re-engagement attempts exhausted")

        # 3. Everything else is a genuine retry candidate - the question is when.
        if retries < MAX_RETRIES:
            retry_at = self._retry_time(case, history, now, ctx)
            if retry_at is None:
                return Action(
                    ActionType.WAIT, ctx.issuer_health.next_poll(now),
                    reason=f"{case.issuer} degraded - holding rather than burning an attempt",
                )
            # Never stack two debits on the same instant.
            last_retry = max(
                (a.action.at for a in history
                 if a.action.type is ActionType.RETRY_PAYMENT),
                default=None,
            )
            if last_retry is not None:
                retry_at = max(retry_at, last_retry + timedelta(hours=MIN_RETRY_GAP_HOURS))

            notice, retry_at = self._pre_debit_gate(case, history, retry_at, now, used)
            if notice is not None:
                return notice

            return Action(
                ActionType.RETRY_PAYMENT, retry_at,
                rail=self._rail_for(case),
                reason=self._retry_reason(case, retry_at, ctx),
            )

        # 4. Retries spent. One nudge, then done.
        if len(messages) < MAX_MESSAGES - 1:
            return Action(
                ActionType.SEND_MESSAGE, next_contact_slot(now),
                channel=best_channel(case, used),
                template_id="PAYMENT_FAILED_RETRY_LINK",
                reason="retries exhausted - ask the customer to complete it",
            )
        return Action(ActionType.STOP, now, reason="retry and contact budget spent")

    # ------------------------------------------------------------------
    # timing
    # ------------------------------------------------------------------
    def _retry_time(
        self, case: Case, history: list[Attempt], now: datetime, ctx: Context
    ) -> datetime | None:
        """When to retry. `None` means hold - the issuer is down right now."""
        code = case.failure_code

        if code in {FailureCode.ISSUER_DOWN, FailureCode.GATEWAY_TIMEOUT}:
            if ctx.issuer_health.is_degraded(case.issuer, now):
                return None
            return now

        if code is FailureCode.INSUFFICIENT_FUNDS:
            day = case.observed_salary_day
            if day is None:
                return self._blind_liquidity_target(now)
            if 0 <= (now.day - day) <= 2:
                return now                      # already inside the window
            return day_at(now, day)

        if code is FailureCode.LIMIT_EXCEEDED:
            return (now + timedelta(days=1)).replace(hour=10, minute=0, second=0, microsecond=0)

        if code is FailureCode.UPI_COLLECT_EXPIRED:
            return now

        return now + timedelta(hours=12)

    def _blind_liquidity_target(self, now: datetime) -> datetime:
        """No liquidity signal for this customer - use the population prior.

        Rather than retrying on an arbitrary timer, aim at the next common
        salary-credit day if one is close enough to be worth waiting for.
        """
        best = None
        for day in POPULATION_SALARY_DAYS:
            candidate = day_at(now, day)
            if (candidate - now) <= timedelta(days=BLIND_TARGET_MAX_WAIT_DAYS):
                if best is None or candidate < best:
                    best = candidate
        if best is not None and best > now:
            return best
        return now + timedelta(days=UNKNOWN_LIQUIDITY_DELAY_DAYS)

    def _rail_for(self, case: Case) -> Method:
        """Switch rails where the rail itself was the problem."""
        if case.failure_code is FailureCode.UPI_COLLECT_EXPIRED:
            return Method.UPI_INTENT        # stop waiting on a collect approval
        return case.method

    def _retry_reason(self, case: Case, at: datetime, ctx: Context) -> str:
        code = case.failure_code
        if code in {FailureCode.ISSUER_DOWN, FailureCode.GATEWAY_TIMEOUT}:
            return f"{case.issuer} back to {ctx.issuer_health.health_score(case.issuer, at):.0%} success"
        if code is FailureCode.INSUFFICIENT_FUNDS:
            if case.observed_salary_day is not None:
                return f"retrying inside the inferred liquidity window (day {case.observed_salary_day})"
            return "no liquidity signal - spacing the retry out"
        if code is FailureCode.LIMIT_EXCEEDED:
            return "waiting for the daily limit to reset"
        if code is FailureCode.UPI_COLLECT_EXPIRED:
            return "collect request expired - switching to intent"
        return "soft decline retry"

    def _pre_debit_gate(
        self, case: Case, history: list[Attempt], retry_at: datetime,
        now: datetime, used: set,
    ) -> tuple[Action | None, datetime]:
        """Serve the e-mandate pre-debit notice before any auto-debit.

        Returns (notice_action_or_None, adjusted_retry_time). A notice already
        sent does not entitle us to debit immediately - the debit slides to 24h
        after the notice instead. Recomputing the deadline from a retry time
        that kept moving is what made an earlier version send the notice twice.
        """
        if case.method is not Method.EMANDATE:
            return None, retry_at

        served = [
            a.action.at for a in history
            if a.action.type is ActionType.SEND_MESSAGE
            and a.action.template_id == PRE_DEBIT_TEMPLATE
        ]
        if served:
            earliest = max(served) + timedelta(hours=PRE_DEBIT_NOTICE_HOURS)
            return None, max(retry_at, earliest)

        return (
            Action(
                ActionType.SEND_MESSAGE, next_contact_slot(now),
                channel=best_channel(case, used),
                template_id=PRE_DEBIT_TEMPLATE,
                reason="e-mandate pre-debit notice, 24h ahead of the debit",
            ),
            retry_at,
        )
