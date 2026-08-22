"""Batch runner.

Drives one policy over one ledger and records what happened. The same loop
runs every arm, so differences between arms come only from the decisions the
policies make - never from how they were executed.

The order of checks inside the loop matters and is deliberate:

  1. state stopping rules  - is this case still legitimately workable?
  2. policy proposes       - what would you like to do, and when?
  3. natural recovery      - did they pay on their own before we got there?
  4. guardrail             - is the proposal actually permitted?
  5. execute and record    - resolve the outcome, chain it into the audit log

Step 3 sitting before step 5 is what stops every arm from claiming credit for
money that was coming back regardless.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path

from recoup.audit.ledger import AuditLedger
from recoup.domain import (
    Action,
    ActionType,
    Attempt,
    Case,
    CaseRecord,
    StopReason,
)
from recoup.policy import compliance, stopping
from recoup.policy.base import Context, next_contact_slot
from recoup.sim import world
from recoup.sim.generator import HORIZON_DAYS, Ledger
from recoup.sim.issuer_health import IssuerHealth

MAX_LOOP_ITERATIONS = 80
MAX_VETOES = 8
MAX_WAITS = 40

# Vetoes a different timestamp can fix. Anything else means the proposal is
# structurally impermissible and re-asking at 11am will not help.
TIME_RESOLVABLE = {"rbi_contact_window", "contact_frequency_cap"}


@dataclass
class CaseResult:
    case_id: str
    arm: str
    amount_paise: int
    segment: str = ""
    failure_code: str = ""
    recovered: bool = False
    recovered_naturally: bool = False
    recovered_paise: int = 0
    cost_paise: int = 0
    n_actions: int = 0
    n_contacts: int = 0
    n_retries: int = 0
    n_violations: int = 0
    n_vetoes: int = 0
    promise_made: bool = False
    stop_reason: StopReason | None = None

    # Per-attempt trace, indexed by attempt position. Needed to show how the
    # return on the Nth action decays - which is what the economic stopping
    # rule is reacting to.
    attempt_values: list[int] = field(default_factory=list)
    attempt_costs: list[int] = field(default_factory=list)


@dataclass
class ArmResult:
    arm: str
    policy_version: str
    results: list[CaseResult] = field(default_factory=list)
    ledger: AuditLedger | None = None


def _state_snapshot(case: Case, history: list[Attempt], now: datetime) -> dict:
    return {
        "case_type": case.case_type.value,
        "amount_paise": case.amount_paise,
        "method": case.method.value,
        "issuer": case.issuer,
        "failure_code": case.failure_code.value,
        "age_hours": round((now - case.created_at).total_seconds() / 3600, 1),
        "attempts_so_far": len(history),
        "observed_salary_day": case.observed_salary_day,
        "dnd_registered": case.dnd_registered,
        "opted_out": case.opted_out,
    }


def _action_dict(action: Action | None) -> dict | None:
    if action is None:
        return None
    return {
        "type": action.type.value,
        "at": action.at.isoformat(),
        "rail": action.rail.value if action.rail else None,
        "channel": action.channel.value if action.channel else None,
        "template_id": action.template_id,
        "reason": action.reason,
        "cost_paise": action.cost_paise(),
    }


def _advance_after_veto(failed: list[compliance.Verdict], at: datetime) -> datetime:
    """Push the clock to the earliest moment the veto could clear."""
    rules = {v.rule for v in failed}
    if "rbi_contact_window" in rules:
        return next_contact_slot(at)
    if "contact_frequency_cap" in rules:
        return at + timedelta(hours=compliance.MIN_HOURS_BETWEEN_CONTACTS)
    return at + timedelta(hours=1)


def run_case(
    record: CaseRecord,
    policy,
    ctx: Context,
    *,
    enforce_compliance: bool = True,
    audit: AuditLedger | None = None,
) -> CaseResult:
    """Run one case to termination under one policy."""
    case = replace(record.case)          # per-arm copy; arms must not bleed
    latents = record.latents
    arm = policy.name

    history: list[Attempt] = []
    now = case.created_at
    horizon = case.created_at + timedelta(days=HORIZON_DAYS)
    res = CaseResult(case_id=case.case_id, arm=arm, amount_paise=case.amount_paise,
                     segment=case.case_type.value, failure_code=case.failure_code.value)
    waits = 0

    def log(proposed, checks, executed, outcome, at):
        if audit is not None:
            audit.decision(
                case_id=case.case_id, arm=arm, now=at,
                state=_state_snapshot(case, history, at),
                diagnosis={"failure_code": case.failure_code.value,
                           "is_hard_decline": case.is_hard_decline},
                proposed=_action_dict(proposed),
                checks=[{"rule": v.rule, "result": "pass" if v.allowed else "FAIL",
                         "detail": v.detail} for v in checks],
                executed=_action_dict(executed),
                outcome=outcome,
                policy_version=policy.version,
            )

    def natural_by(t: datetime) -> bool:
        nat = latents.natural_recovery_at
        return nat is not None and nat <= t

    for _ in range(MAX_LOOP_ITERATIONS):
        stop = stopping.check(case, history, now, horizon, res.recovered)
        if stop is not None:
            res.stop_reason = stop
            break

        action = policy.propose(case, history, now, ctx)

        if action.type is ActionType.STOP:
            res.stop_reason = StopReason.POLICY_STOP
            log(action, [], None, {"stopped": action.reason}, now)
            break

        at = max(action.at, now)
        if at >= horizon:
            res.stop_reason = StopReason.HORIZON_REACHED
            break

        # Would they have paid anyway, before this action lands?
        if natural_by(at):
            res.recovered = True
            res.recovered_naturally = True
            res.recovered_paise = case.amount_paise
            res.stop_reason = StopReason.RECOVERED
            log(action, [], None,
                {"recovered": True, "naturally": True,
                 "note": "customer paid without intervention"},
                latents.natural_recovery_at)
            break

        if action.type is ActionType.WAIT:
            waits += 1
            if waits > MAX_WAITS:
                res.stop_reason = StopReason.POLICY_STOP
                break
            if case.promise_to_pay_at is not None and at >= case.promise_to_pay_at:
                honoured = world.promise_honoured(latents, len(history))
                case.promise_to_pay_at = None
                if honoured:
                    res.recovered = True
                    res.recovered_paise = case.amount_paise
                    res.stop_reason = StopReason.RECOVERED
                    log(action, [], None,
                        {"recovered": True, "note": "promise to pay honoured"}, at)
                    break
                log(action, [], None,
                    {"recovered": False, "note": "promise to pay broken"}, at)
            now = at
            continue

        verdicts = compliance.evaluate(case, action, history, at)
        failed = compliance.violations(verdicts)

        if enforce_compliance and failed:
            res.n_vetoes += 1
            log(action, verdicts, None,
                {"vetoed": True, "rules": [v.rule for v in failed]}, at)
            if res.n_vetoes > MAX_VETOES or not {v.rule for v in failed} & TIME_RESOLVABLE:
                res.stop_reason = StopReason.POLICY_STOP
                break
            now = _advance_after_veto(failed, at)
            continue

        res.n_violations += len(failed)

        outcome = world.resolve(case, latents, action, history, at)
        executed = replace(action, at=at)
        attempt = Attempt(
            seq=len(history), action=executed,
            succeeded=outcome.outcome is world.Outcome.RECOVERED,
            cost_paise=executed.cost_paise(),
            recovered_paise=case.amount_paise if outcome.outcome is world.Outcome.RECOVERED else 0,
            note=outcome.note,
        )
        history.append(attempt)
        res.cost_paise += attempt.cost_paise
        res.n_actions += 1
        if executed.type is ActionType.RETRY_PAYMENT:
            res.n_retries += 1
        if executed.type in {ActionType.SEND_MESSAGE, ActionType.ESCALATE_HUMAN}:
            res.n_contacts += 1
        now = at

        log(action, verdicts, executed,
            {"result": outcome.outcome.value, "p_success": round(outcome.p_success, 4),
             "note": outcome.note,
             "violations": [v.rule for v in failed] if failed else []}, at)

        if outcome.outcome is world.Outcome.RECOVERED:
            res.recovered = True
            res.recovered_paise = case.amount_paise
            res.stop_reason = StopReason.RECOVERED
            break
        if outcome.outcome is world.Outcome.PROMISE_TO_PAY:
            case.promise_to_pay_at = outcome.promise_at
            res.promise_made = True
    else:
        res.stop_reason = StopReason.HORIZON_REACHED

    res.attempt_values = [a.recovered_paise for a in history]
    res.attempt_costs = [a.cost_paise for a in history]

    # A case that never got worked can still recover on its own inside the window.
    if not res.recovered and natural_by(horizon):
        res.recovered = True
        res.recovered_naturally = True
        res.recovered_paise = case.amount_paise
        res.stop_reason = StopReason.RECOVERED

    return res


def run_arm(
    ledger: Ledger,
    policy,
    *,
    enforce_compliance: bool = True,
    audit_path: Path | None = None,
    audit_case_ids: set[str] | None = None,
    issuer_health_seed: int = 11,
    max_workers: int = 1,
) -> ArmResult:
    """Run a policy across the whole batch.

    `max_workers > 1` runs cases concurrently. Cases are independent - the
    sequencing that matters is *within* a case - so this is safe and is the
    difference between a ten-minute LLM run and a one-minute one. Results are
    reassembled in ledger order so output stays deterministic.
    """
    ctx = Context(
        issuer_health=IssuerHealth.from_outages(ledger.outages, seed=issuer_health_seed),
        horizon=datetime.max,
    )
    audit = AuditLedger(path=audit_path) if audit_path else None
    out = ArmResult(arm=policy.name, policy_version=policy.version, ledger=audit)

    def one(record):
        want_audit = (audit if (audit_case_ids is None
                                or record.case.case_id in audit_case_ids) else None)
        return run_case(record, policy, ctx,
                        enforce_compliance=enforce_compliance, audit=want_audit)

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            out.results.extend(pool.map(one, ledger.records))
    else:
        out.results.extend(one(r) for r in ledger.records)

    if audit is not None:
        audit.close()
    return out
