"""Stopping rules, the audit chain, and the guarantees the batch relies on."""
from __future__ import annotations

from datetime import datetime, timedelta

from recoup.audit.ledger import AuditLedger
from recoup.domain import (
    Action,
    ActionType,
    Attempt,
    Case,
    CaseType,
    FailureCode,
    Method,
    StopReason,
)
from recoup.policy import stopping
from recoup.sim.generator import generate_ledger

BASE = datetime(2026, 3, 10, 11, 0)
HORIZON = BASE + timedelta(days=30)


def make_case(**kw) -> Case:
    defaults = dict(
        case_id="c1", merchant_id="acc_1", customer_id="cust_1",
        case_type=CaseType.PAYMENT_FAILURE, amount_paise=250000,
        method=Method.CARD, issuer="HDFC",
        failure_code=FailureCode.INSUFFICIENT_FUNDS, created_at=BASE,
    )
    defaults.update(kw)
    return Case(**defaults)


def attempts(n: int, kind=ActionType.SEND_MESSAGE) -> list[Attempt]:
    return [Attempt(seq=i, action=Action(kind, BASE), succeeded=False, cost_paise=20)
            for i in range(n)]


# --- stopping --------------------------------------------------------------

def test_recovered_case_stops():
    assert stopping.check(make_case(), [], BASE, HORIZON, recovered=True) is StopReason.RECOVERED


def test_opt_out_stops():
    assert stopping.check(make_case(opted_out=True), [], BASE, HORIZON) is StopReason.CUSTOMER_OPT_OUT


def test_dispute_stops():
    assert stopping.check(make_case(dispute_open=True), [], BASE, HORIZON) is StopReason.DISPUTE_OPEN


def test_horizon_stops():
    assert stopping.check(make_case(), [], HORIZON, HORIZON) is StopReason.HORIZON_REACHED


def test_action_budget_stops():
    hist = attempts(stopping.MAX_TOTAL_ACTIONS)
    assert stopping.check(make_case(), hist, BASE, HORIZON) is StopReason.MAX_ATTEMPTS


def test_message_budget_stops_before_action_budget():
    hist = attempts(stopping.MAX_MESSAGES)
    assert stopping.check(make_case(), hist, BASE, HORIZON) is StopReason.MAX_ATTEMPTS


def test_healthy_case_continues():
    assert stopping.check(make_case(), attempts(1), BASE, HORIZON) is None


def test_promise_to_pay_pauses_then_resumes():
    promised = BASE + timedelta(days=5)
    case = make_case(promise_to_pay_at=promised)
    assert stopping.is_paused(case, BASE)
    assert not stopping.is_paused(case, promised + timedelta(hours=1))


# --- economics -------------------------------------------------------------

def test_cheap_retry_on_a_large_amount_is_economic():
    assert stopping.is_economic(amount_paise=250000, p_success=0.10, action_cost_paise=5)


def test_human_escalation_on_a_tiny_subscription_is_not():
    """Rs 50 of human time chasing Rs 99 at a 30% hit rate is value-destroying."""
    assert not stopping.is_economic(
        amount_paise=9900, p_success=0.30, action_cost_paise=5000
    )


def test_nuisance_cost_makes_repeat_contacts_uneconomic():
    """Same action, same odds - only the contact count changes."""
    args = dict(amount_paise=50000, p_success=0.05, action_cost_paise=35)
    assert stopping.is_economic(**args, prior_contacts=0)
    assert not stopping.is_economic(**args, prior_contacts=4)


def test_hurdle_is_respected():
    # Expected return exactly equals cost - below the 3x hurdle, so refused.
    assert not stopping.is_economic(
        amount_paise=1000, p_success=0.10, action_cost_paise=100, prior_contacts=0
    )


# --- audit chain -----------------------------------------------------------

def test_chain_verifies_when_untouched():
    led = AuditLedger()
    for i in range(20):
        led.append({"case_id": f"c{i}", "n": i})
    ok, detail = led.verify()
    assert ok, detail


def test_tampering_with_a_payload_breaks_the_chain():
    led = AuditLedger()
    for i in range(10):
        led.append({"case_id": f"c{i}", "amount": 100 * i})
    led.records[4].payload["amount"] = 999999
    ok, detail = led.verify()
    assert not ok
    assert "record 4" in detail


def test_deleting_a_record_breaks_the_chain():
    led = AuditLedger()
    for i in range(10):
        led.append({"n": i})
    del led.records[5]
    ok, _ = led.verify()
    assert not ok


def test_each_record_links_to_its_predecessor():
    led = AuditLedger()
    a = led.append({"n": 1})
    b = led.append({"n": 2})
    assert b.prev_hash == a.hash


# --- determinism -----------------------------------------------------------

def test_ledger_generation_is_reproducible():
    a = generate_ledger(300, seed=42)
    b = generate_ledger(300, seed=42)
    assert [r.case.case_id for r in a.records] == [r.case.case_id for r in b.records]
    assert [r.case.amount_paise for r in a.records] == [r.case.amount_paise for r in b.records]
    assert [r.latents.intent for r in a.records] == [r.latents.intent for r in b.records]


def test_different_seeds_give_different_worlds():
    a = generate_ledger(300, seed=1)
    b = generate_ledger(300, seed=2)
    assert [r.case.amount_paise for r in a.records] != [r.case.amount_paise for r in b.records]


def test_policies_cannot_mutate_the_shared_ledger():
    """Arms run over the same records - one arm must not leak into the next."""
    from recoup.agent.heuristic import RecoupHeuristic
    from recoup.eval.runner import run_arm

    ledger = generate_ledger(200, seed=5)
    before = [(r.case.promise_to_pay_at, r.case.opted_out) for r in ledger.records]
    run_arm(ledger, RecoupHeuristic())
    after = [(r.case.promise_to_pay_at, r.case.opted_out) for r in ledger.records]
    assert before == after
