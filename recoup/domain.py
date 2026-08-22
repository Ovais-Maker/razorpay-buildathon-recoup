"""Core domain types for Recoup.

Everything the agent, the guardrail and the simulator pass around is defined
here. Amounts are always in paise (integer) - never floats for money.
All datetimes are naive and interpreted as IST.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# --------------------------------------------------------------------------
# What kind of money is at risk
# --------------------------------------------------------------------------

class CaseType(str, Enum):
    PAYMENT_FAILURE = "payment_failure"        # one-off checkout payment declined
    MANDATE_FAILURE = "mandate_failure"        # subscription / eMandate debit bounced
    RECEIVABLE_OVERDUE = "receivable_overdue"  # B2B invoice past due date


class Method(str, Enum):
    CARD = "card"
    UPI_COLLECT = "upi_collect"
    UPI_INTENT = "upi_intent"
    NETBANKING = "netbanking"
    EMANDATE = "emandate"
    PAYMENT_LINK = "payment_link"


class FailureCode(str, Enum):
    # --- hard declines: retrying is a network-rule violation, never do it ---
    CARD_STOLEN = "card_stolen"
    CARD_LOST = "card_lost"
    ACCOUNT_CLOSED = "account_closed"
    INVALID_CARD = "invalid_card"
    DO_NOT_HONOUR_PERMANENT = "do_not_honour_permanent"
    FRAUD_SUSPECTED = "fraud_suspected"
    MANDATE_REVOKED = "mandate_revoked"

    # --- soft declines: retryable, but only with the right strategy ---
    INSUFFICIENT_FUNDS = "insufficient_funds"
    ISSUER_DOWN = "issuer_down"
    GATEWAY_TIMEOUT = "gateway_timeout"
    THREE_DS_DROPOFF = "3ds_dropoff"
    OTP_TIMEOUT = "otp_timeout"
    UPI_COLLECT_EXPIRED = "upi_collect_expired"
    LIMIT_EXCEEDED = "limit_exceeded"

    # --- receivables never had a gateway attempt ---
    NONE = "none"


HARD_DECLINES: frozenset[FailureCode] = frozenset({
    FailureCode.CARD_STOLEN,
    FailureCode.CARD_LOST,
    FailureCode.ACCOUNT_CLOSED,
    FailureCode.INVALID_CARD,
    FailureCode.DO_NOT_HONOUR_PERMANENT,
    FailureCode.FRAUD_SUSPECTED,
    FailureCode.MANDATE_REVOKED,
})

# Retrying the charge itself is pointless for these - the customer dropped out
# of an interactive step. They need a fresh link, not another auth attempt.
NEEDS_HUMAN_REENGAGEMENT: frozenset[FailureCode] = frozenset({
    FailureCode.THREE_DS_DROPOFF,
    FailureCode.OTP_TIMEOUT,
})


# --------------------------------------------------------------------------
# How we can reach the customer
# --------------------------------------------------------------------------

class Channel(str, Enum):
    SMS = "sms"
    WHATSAPP = "whatsapp"
    EMAIL = "email"
    VOICE = "voice"


# Per-contact cost in paise. Voice and human escalation are the expensive ones,
# which is what makes the economic stopping rule bite.
CHANNEL_COST_PAISE: dict[Channel, int] = {
    Channel.SMS: 20,
    Channel.WHATSAPP: 35,
    Channel.EMAIL: 2,
    Channel.VOICE: 150,
}

RETRY_COST_PAISE = 5          # gateway attempt
HUMAN_ESCALATION_COST_PAISE = 5000   # a person picks up the case


# --------------------------------------------------------------------------
# Actions the agent may take
# --------------------------------------------------------------------------

class ActionType(str, Enum):
    RETRY_PAYMENT = "retry_payment"
    SEND_MESSAGE = "send_message"
    ESCALATE_HUMAN = "escalate_human"
    WAIT = "wait"
    STOP = "stop"


@dataclass(frozen=True)
class Action:
    """A single bounded intervention, scheduled for a specific moment."""
    type: ActionType
    at: datetime
    rail: Method | None = None          # RETRY_PAYMENT: which rail to use
    channel: Channel | None = None      # SEND_MESSAGE: how to reach them
    template_id: str | None = None      # SEND_MESSAGE: DLT-registered template
    reason: str = ""                    # why the policy chose this

    def cost_paise(self) -> int:
        if self.type is ActionType.RETRY_PAYMENT:
            return RETRY_COST_PAISE
        if self.type is ActionType.SEND_MESSAGE and self.channel is not None:
            return CHANNEL_COST_PAISE[self.channel]
        if self.type is ActionType.ESCALATE_HUMAN:
            return HUMAN_ESCALATION_COST_PAISE
        return 0


@dataclass
class Attempt:
    """An action that was actually executed, plus what happened."""
    seq: int
    action: Action
    succeeded: bool
    cost_paise: int
    recovered_paise: int = 0
    note: str = ""


class StopReason(str, Enum):
    HARD_DECLINE = "hard_decline"
    MAX_ATTEMPTS = "max_attempts"
    CUSTOMER_OPT_OUT = "customer_opt_out"
    ALREADY_PAID = "already_paid"
    DISPUTE_OPEN = "dispute_open"
    NOT_ECONOMIC = "not_economic"
    HORIZON_REACHED = "horizon_reached"
    RECOVERED = "recovered"
    POLICY_STOP = "policy_stop"


# --------------------------------------------------------------------------
# The case itself
# --------------------------------------------------------------------------

@dataclass
class Case:
    """One unit of at-risk money.

    Only the fields in this class are visible to a policy. Everything the
    simulator uses to decide outcomes lives in `Latents`, which policies
    never see.
    """
    case_id: str
    merchant_id: str
    customer_id: str
    case_type: CaseType
    amount_paise: int
    method: Method
    issuer: str
    failure_code: FailureCode
    created_at: datetime

    due_date: datetime | None = None      # receivables only
    dnd_registered: bool = False          # TRAI DND registry
    opted_out: bool = False               # customer said STOP previously
    consent_whatsapp: bool = False
    promise_to_pay_at: datetime | None = None
    dispute_open: bool = False

    # Inferred from this customer's past successful payments. Noisy and
    # often absent - it is a feature, not ground truth.
    observed_salary_day: int | None = None

    @property
    def is_hard_decline(self) -> bool:
        return self.failure_code in HARD_DECLINES


@dataclass
class Latents:
    """Ground truth. The simulator reads this; no policy ever does."""
    intent: str                    # will_pay_anyway | needs_nudge | low_intent | wont_pay
    liquidity_day: int | None      # day of month their salary/receipts land
    channel_affinity: dict[Channel, float]
    natural_recovery_at: datetime | None   # when they'd pay with zero intervention
    issuer_recovery_at: datetime | None    # when a degraded issuer comes back
    u_stream: list[float] = field(default_factory=list)  # common random numbers


@dataclass
class CaseRecord:
    """A case bundled with its ground truth, as emitted by the generator."""
    case: Case
    latents: Latents
