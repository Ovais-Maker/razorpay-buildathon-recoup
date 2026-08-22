"""Synthetic at-risk ledger.

The generator is deliberately transparent: every distribution below is stated
in code so a reader can judge whether the world is fair. Nothing here is tuned
to make the agent look good - the agent has to earn its lift against the same
world the baselines run in.

Each case carries a `u_stream` of pre-drawn uniforms. Every arm of the eval
consumes the same stream for the same case, so arms differ only by the
decisions they make (common random numbers, a variance reduction technique).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

from recoup.domain import (
    Case,
    CaseRecord,
    CaseType,
    Channel,
    FailureCode,
    Latents,
    Method,
)

SIM_START = datetime(2026, 3, 1, 0, 0, 0)
HORIZON_DAYS = 30
U_STREAM_LEN = 48

ISSUERS = [
    "HDFC", "ICICI", "SBI", "AXIS", "KOTAK", "PNB",
    "BOB", "YES", "INDUSIND", "IDFC", "CANARA", "UNION",
]

# Issuer share of volume - the big four carry most of it.
ISSUER_WEIGHTS = [22, 18, 20, 12, 7, 5, 4, 3, 3, 2, 2, 2]

# For a payments company the overwhelming majority of at-risk events are
# failed payments and bounced mandates; overdue invoices are a smaller book
# with a far larger ticket size.
CASE_TYPE_MIX = {
    CaseType.PAYMENT_FAILURE: 0.62,
    CaseType.MANDATE_FAILURE: 0.30,
    CaseType.RECEIVABLE_OVERDUE: 0.08,
}

# Failure-code mix per rail. Proportions are illustrative; the taxonomy and
# the hard/soft split are real.
CARD_FAILURES = {
    FailureCode.INSUFFICIENT_FUNDS: 0.24,
    FailureCode.THREE_DS_DROPOFF: 0.21,
    FailureCode.ISSUER_DOWN: 0.11,
    FailureCode.GATEWAY_TIMEOUT: 0.08,
    FailureCode.OTP_TIMEOUT: 0.07,
    FailureCode.LIMIT_EXCEEDED: 0.06,
    FailureCode.DO_NOT_HONOUR_PERMANENT: 0.08,
    FailureCode.INVALID_CARD: 0.05,
    FailureCode.ACCOUNT_CLOSED: 0.04,
    FailureCode.FRAUD_SUSPECTED: 0.03,
    FailureCode.CARD_STOLEN: 0.02,
    FailureCode.CARD_LOST: 0.01,
}

UPI_FAILURES = {
    FailureCode.UPI_COLLECT_EXPIRED: 0.38,
    FailureCode.INSUFFICIENT_FUNDS: 0.24,
    FailureCode.ISSUER_DOWN: 0.16,
    FailureCode.GATEWAY_TIMEOUT: 0.12,
    FailureCode.LIMIT_EXCEEDED: 0.10,
}

MANDATE_FAILURES = {
    FailureCode.INSUFFICIENT_FUNDS: 0.54,
    FailureCode.LIMIT_EXCEEDED: 0.12,
    FailureCode.ISSUER_DOWN: 0.11,
    FailureCode.MANDATE_REVOKED: 0.13,
    FailureCode.ACCOUNT_CLOSED: 0.10,
}

# How likely someone is to act on a message that reaches them, before
# per-customer noise and contact fatigue are applied.
BASE_CHANNEL_AFFINITY = {
    Channel.WHATSAPP: 0.22,
    Channel.SMS: 0.12,
    Channel.EMAIL: 0.05,
    Channel.VOICE: 0.24,
}

INTENT_MIX = {
    "will_pay_anyway": 0.12,   # recovers with zero intervention
    "needs_nudge": 0.45,       # the population the agent actually wins
    "low_intent": 0.28,        # convertible, but expensive
    "wont_pay": 0.15,          # nothing works; contacting them is pure cost
}

# Weighted toward month-start because that is when salaries land.
SALARY_DAYS = [1, 1, 1, 2, 5, 7, 7, 10, 15]

# Instruments that are simply gone - no channel or timing recovers these.
DEAD_INSTRUMENT = {
    FailureCode.CARD_STOLEN,
    FailureCode.CARD_LOST,
    FailureCode.ACCOUNT_CLOSED,
    FailureCode.INVALID_CARD,
}


def _weighted_choice(rng: random.Random, mapping: dict):
    keys = list(mapping.keys())
    weights = list(mapping.values())
    return rng.choices(keys, weights=weights, k=1)[0]


def _amount_for(rng: random.Random, case_type: CaseType) -> int:
    """Realistic ticket sizes, returned in paise."""
    if case_type is CaseType.RECEIVABLE_OVERDUE:
        rupees = min(rng.lognormvariate(10.2, 0.75), 2_500_000)  # B2B invoices
    elif case_type is CaseType.MANDATE_FAILURE:
        rupees = rng.choice([99, 149, 199, 299, 499, 599, 999, 1499, 2999])
    else:
        rupees = min(rng.lognormvariate(6.9, 0.85), 200_000)     # checkout
    return int(round(rupees)) * 100


def _plan_issuer_outages(rng: random.Random) -> dict[str, tuple[datetime, datetime]]:
    """A few banks degrade for a stretch during the window.

    This is what makes fixed-timer retries lose: they fire straight into a
    dead issuer and burn an attempt for nothing.
    """
    outages: dict[str, tuple[datetime, datetime]] = {}
    for issuer in rng.sample(ISSUERS, k=3):
        start = SIM_START + timedelta(
            days=rng.randint(0, HORIZON_DAYS - 1), hours=rng.randint(0, 23)
        )
        outages[issuer] = (start, start + timedelta(hours=rng.uniform(1.5, 9.0)))
    return outages


@dataclass
class Ledger:
    """A batch of at-risk cases plus the observable signals that accompany it.

    `outages` is ground truth about issuer degradation. It is NOT handed to
    policies directly - `recoup.sim.issuer_health` derives a lagged, noisy
    observable view from it, which is what a real merchant would actually see
    in their own success-rate telemetry.
    """
    records: list[CaseRecord]
    outages: dict[str, tuple[datetime, datetime]]

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)


def generate_ledger(n: int = 10_000, seed: int = 7) -> Ledger:
    """Build `n` at-risk cases together with their ground truth."""
    rng = random.Random(seed)
    outages = _plan_issuer_outages(rng)
    records: list[CaseRecord] = []

    for i in range(n):
        case_type = _weighted_choice(rng, CASE_TYPE_MIX)
        issuer = rng.choices(ISSUERS, weights=ISSUER_WEIGHTS, k=1)[0]

        if case_type is CaseType.RECEIVABLE_OVERDUE:
            method = Method.PAYMENT_LINK
            failure = FailureCode.NONE
        elif case_type is CaseType.MANDATE_FAILURE:
            method = Method.EMANDATE
            failure = _weighted_choice(rng, MANDATE_FAILURES)
        else:
            method = rng.choices(
                [Method.CARD, Method.UPI_COLLECT, Method.NETBANKING],
                weights=[45, 45, 10], k=1,
            )[0]
            table = UPI_FAILURES if method is Method.UPI_COLLECT else CARD_FAILURES
            failure = _weighted_choice(rng, table)

        created_at = SIM_START + timedelta(
            days=rng.uniform(0, 2), hours=rng.uniform(0, 24)
        )
        due_date = (
            created_at - timedelta(days=rng.randint(1, 45))
            if case_type is CaseType.RECEIVABLE_OVERDUE
            else None
        )

        intent = _weighted_choice(rng, INTENT_MIX)

        # Natural recovery: some people pay with no prompting at all. Measuring
        # against this is what separates real lift from taking credit for it.
        natural_recovery_at = None
        if intent == "will_pay_anyway":
            natural_recovery_at = created_at + timedelta(days=rng.uniform(0.5, 7))
        elif intent == "needs_nudge" and rng.random() < 0.15:
            natural_recovery_at = created_at + timedelta(days=rng.uniform(5, 26))

        if failure in DEAD_INSTRUMENT:
            natural_recovery_at = None

        liquidity_day = None
        if failure is FailureCode.INSUFFICIENT_FUNDS and rng.random() < 0.82:
            liquidity_day = rng.choice(SALARY_DAYS)

        # What our own analytics would have inferred about this customer:
        # right most of the time, off by a day sometimes, missing often,
        # and occasionally confidently wrong about someone who has none.
        observed_salary_day = None
        if liquidity_day is not None:
            roll = rng.random()
            if roll < 0.62:
                observed_salary_day = liquidity_day
            elif roll < 0.74:
                observed_salary_day = max(1, min(28, liquidity_day + rng.choice([-1, 1])))
        elif rng.random() < 0.06:
            observed_salary_day = rng.choice(SALARY_DAYS)

        issuer_recovery_at = None
        if failure in {FailureCode.ISSUER_DOWN, FailureCode.GATEWAY_TIMEOUT}:
            window = outages.get(issuer)
            issuer_recovery_at = (
                window[1] if window is not None
                else created_at + timedelta(hours=rng.uniform(0.5, 6.0))
            )

        affinity = {
            ch: max(0.01, min(0.95, rng.gauss(base, base * 0.35)))
            for ch, base in BASE_CHANNEL_AFFINITY.items()
        }

        case = Case(
            case_id=f"case_{i:06d}",
            merchant_id=f"acc_{rng.randint(1, 40):03d}",
            customer_id=f"cust_{rng.randint(1, max(2, n // 3)):06d}",
            case_type=case_type,
            amount_paise=_amount_for(rng, case_type),
            method=method,
            issuer=issuer,
            failure_code=failure,
            created_at=created_at,
            due_date=due_date,
            dnd_registered=rng.random() < 0.18,
            opted_out=rng.random() < 0.04,
            consent_whatsapp=rng.random() < 0.72,
            observed_salary_day=observed_salary_day,
        )

        latents = Latents(
            intent=intent,
            liquidity_day=liquidity_day,
            channel_affinity=affinity,
            natural_recovery_at=natural_recovery_at,
            issuer_recovery_at=issuer_recovery_at,
            u_stream=[rng.random() for _ in range(U_STREAM_LEN)],
        )
        records.append(CaseRecord(case=case, latents=latents))

    return Ledger(records=records, outages=outages)


def ledger_value_paise(records) -> int:
    return sum(r.case.amount_paise for r in records)
