"""One test per compliance rule.

This file is the evidence behind "compliant escalation". A reviewer should be
able to read it and see that each rule fails when it should, passes when it
should, and cannot be bypassed by the strategist.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from recoup.domain import (
    Action,
    ActionType,
    Attempt,
    Case,
    CaseType,
    Channel,
    FailureCode,
    Method,
)
from recoup.policy import compliance

BASE = datetime(2026, 3, 10, 11, 0)


def make_case(**kw) -> Case:
    defaults = dict(
        case_id="c1", merchant_id="acc_1", customer_id="cust_1",
        case_type=CaseType.PAYMENT_FAILURE, amount_paise=250000,
        method=Method.CARD, issuer="HDFC",
        failure_code=FailureCode.INSUFFICIENT_FUNDS, created_at=BASE,
    )
    defaults.update(kw)
    return Case(**defaults)


def msg(at=BASE, channel=Channel.SMS, template="PAYMENT_FAILED_RETRY_LINK") -> Action:
    return Action(ActionType.SEND_MESSAGE, at, channel=channel, template_id=template)


def retry(at=BASE, rail=Method.CARD) -> Action:
    return Action(ActionType.RETRY_PAYMENT, at, rail=rail)


def executed(action: Action, seq: int = 0) -> Attempt:
    return Attempt(seq=seq, action=action, succeeded=False, cost_paise=action.cost_paise())


def rule_verdict(verdicts, name):
    return next(v for v in verdicts if v.rule == name)


# --- hard declines ---------------------------------------------------------

@pytest.mark.parametrize("code", sorted(compliance.HARD_DECLINES, key=lambda c: c.value))
def test_hard_declines_are_never_retried(code):
    case = make_case(failure_code=code)
    v = compliance.hard_decline_never_retried(case, retry(), [], BASE)
    assert not v.allowed


def test_soft_decline_may_be_retried():
    assert compliance.hard_decline_never_retried(make_case(), retry(), [], BASE).allowed


def test_hard_decline_does_not_block_contacting_the_customer():
    """The instrument is dead, the relationship is not."""
    case = make_case(failure_code=FailureCode.CARD_STOLEN)
    assert compliance.hard_decline_never_retried(case, msg(), [], BASE).allowed


# --- contact window --------------------------------------------------------

@pytest.mark.parametrize("hour,ok", [(7, False), (8, True), (12, True), (18, True), (19, False), (23, False)])
def test_contact_window(hour, ok):
    at = BASE.replace(hour=hour)
    assert compliance.contact_window(make_case(), msg(at), [], at).allowed is ok


def test_retries_are_allowed_outside_the_contact_window():
    """A machine-side debit is not a contact - 3am retries are fine."""
    at = BASE.replace(hour=3)
    assert compliance.contact_window(make_case(), retry(at), [], at).allowed


# --- opt out and DND -------------------------------------------------------

def test_opt_out_blocks_every_channel():
    case = make_case(opted_out=True)
    for ch in Channel:
        assert not compliance.opt_out_honoured(case, msg(channel=ch), [], BASE).allowed
    assert not compliance.opt_out_honoured(
        case, Action(ActionType.ESCALATE_HUMAN, BASE), [], BASE).allowed


def test_dnd_blocks_promotional_but_allows_transactional():
    case = make_case(dnd_registered=True)
    assert not compliance.dnd_promotional_block(case, msg(template="WINBACK_OFFER"), [], BASE).allowed
    assert compliance.dnd_promotional_block(case, msg(template="MANDATE_BOUNCE_NOTICE"), [], BASE).allowed


def test_unregistered_template_is_rejected():
    """TRAI requires DLT registration - an ad-hoc template cannot be sent."""
    v = compliance.dnd_promotional_block(make_case(), msg(template="hello_please_pay"), [], BASE)
    assert not v.allowed


def test_whatsapp_requires_consent():
    no = make_case(consent_whatsapp=False)
    yes = make_case(consent_whatsapp=True)
    assert not compliance.whatsapp_consent(no, msg(channel=Channel.WHATSAPP), [], BASE).allowed
    assert compliance.whatsapp_consent(yes, msg(channel=Channel.WHATSAPP), [], BASE).allowed
    assert compliance.whatsapp_consent(no, msg(channel=Channel.SMS), [], BASE).allowed


# --- e-mandate pre-debit notice -------------------------------------------

def test_mandate_debit_requires_24h_notice():
    case = make_case(method=Method.EMANDATE, case_type=CaseType.MANDATE_FAILURE)
    assert not compliance.emandate_pre_debit_notice(case, retry(rail=Method.EMANDATE), [], BASE).allowed


def test_notice_must_be_a_full_24h_ahead():
    case = make_case(method=Method.EMANDATE, case_type=CaseType.MANDATE_FAILURE)
    late = [executed(msg(at=BASE - timedelta(hours=23), template="PRE_DEBIT_NOTICE"))]
    early = [executed(msg(at=BASE - timedelta(hours=25), template="PRE_DEBIT_NOTICE"))]
    assert not compliance.emandate_pre_debit_notice(case, retry(), late, BASE).allowed
    assert compliance.emandate_pre_debit_notice(case, retry(), early, BASE).allowed


def test_card_retry_needs_no_pre_debit_notice():
    assert compliance.emandate_pre_debit_notice(make_case(), retry(), [], BASE).allowed


# --- caps ------------------------------------------------------------------

def test_network_retry_cap():
    history = [executed(retry(), i) for i in range(compliance.MAX_RETRIES_PER_AUTHORIZATION)]
    assert not compliance.network_retry_cap(make_case(), retry(), history, BASE).allowed
    assert compliance.network_retry_cap(make_case(), retry(), history[:-1], BASE).allowed


def test_contact_frequency_cap_weekly():
    history = [
        executed(msg(at=BASE - timedelta(days=d)), i)
        for i, d in enumerate([1, 3, 5])
    ]
    assert not compliance.contact_frequency_cap(make_case(), msg(), history, BASE).allowed


def test_contact_frequency_cap_minimum_gap():
    history = [executed(msg(at=BASE - timedelta(hours=2)))]
    assert not compliance.contact_frequency_cap(make_case(), msg(), history, BASE).allowed
    old = [executed(msg(at=BASE - timedelta(hours=30)))]
    assert compliance.contact_frequency_cap(make_case(), msg(), old, BASE).allowed


def test_dispute_freezes_everything():
    case = make_case(dispute_open=True)
    assert not compliance.dispute_freeze(case, msg(), [], BASE).allowed
    assert not compliance.dispute_freeze(case, retry(), [], BASE).allowed


# --- the checklist as a whole ---------------------------------------------

def test_evaluate_returns_a_verdict_for_every_rule():
    verdicts = compliance.evaluate(make_case(), msg(), [], BASE)
    assert len(verdicts) == len(compliance.RULES)
    assert len({v.rule for v in verdicts}) == len(compliance.RULES)


def test_a_single_failure_blocks_the_action():
    """Rules are conjunctive - passing eight of nine is still a veto."""
    case = make_case(opted_out=True)
    verdicts = compliance.evaluate(case, msg(), [], BASE)
    assert not compliance.is_allowed(verdicts)
    assert len(compliance.violations(verdicts)) >= 1


def test_clean_action_passes_everything():
    case = make_case(consent_whatsapp=True)
    verdicts = compliance.evaluate(case, msg(channel=Channel.WHATSAPP), [], BASE)
    assert compliance.is_allowed(verdicts), compliance.violations(verdicts)
