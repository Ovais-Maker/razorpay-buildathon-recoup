"""Metrics.

The headline number is **incremental** recovery, not recovery. Because every
arm runs on the same cases with the same pre-drawn luck, a case that recovers
under B0 (do nothing) and also under B3 was never won by B3 - it was always
coming back. Counting it would be taking credit for someone else's money, and
it is the single easiest way for a recovery product to look better than it is.
"""
from __future__ import annotations

from dataclasses import dataclass

from recoup.eval.runner import ArmResult, CaseResult


@dataclass
class Metrics:
    arm: str
    policy_version: str
    n_cases: int
    at_risk_paise: int
    recovered_paise: int
    recovered_cases: int
    incremental_paise: int
    incremental_cases: int
    cannibalised_paise: int      # money B0 would have recovered but this arm lost
    cost_paise: int
    n_actions: int
    n_contacts: int
    n_retries: int
    n_violations: int
    n_vetoes: int

    @property
    def recovery_rate(self) -> float:
        return self.recovered_paise / self.at_risk_paise if self.at_risk_paise else 0.0

    @property
    def incremental_rate(self) -> float:
        return self.incremental_paise / self.at_risk_paise if self.at_risk_paise else 0.0

    @property
    def contacts_per_case(self) -> float:
        return self.n_contacts / self.n_cases if self.n_cases else 0.0

    @property
    def cost_per_incremental_rupee(self) -> float:
        return self.cost_paise / self.incremental_paise if self.incremental_paise > 0 else float("inf")

    @property
    def attempts_per_incremental_recovery(self) -> float:
        return self.n_actions / self.incremental_cases if self.incremental_cases else float("inf")


def compute(
    arm: ArmResult, baseline: ArmResult | None, segment: str | None = None
) -> Metrics:
    """Score one arm, using `baseline` (B0) to strip out natural recovery.

    Pass `segment` to score only one case type. Portfolio totals are dominated
    by B2B invoice value, so the per-segment view is the one that actually
    says whether the payment-failure logic works.
    """
    base_recovered: dict[str, CaseResult] = {}
    if baseline is not None:
        base_recovered = {r.case_id: r for r in baseline.results}

    results = [r for r in arm.results if segment is None or r.segment == segment]

    incremental_paise = 0
    incremental_cases = 0
    cannibalised_paise = 0

    for r in results:
        b = base_recovered.get(r.case_id)
        would_have = b.recovered if b is not None else False
        if r.recovered and not would_have:
            incremental_paise += r.recovered_paise
            incremental_cases += 1
        elif would_have and not r.recovered:
            # Intervening made things worse than leaving them alone.
            cannibalised_paise += b.recovered_paise

    return Metrics(
        arm=arm.arm,
        policy_version=arm.policy_version,
        n_cases=len(results),
        at_risk_paise=sum(r.amount_paise for r in results),
        recovered_paise=sum(r.recovered_paise for r in results),
        recovered_cases=sum(1 for r in results if r.recovered),
        incremental_paise=incremental_paise,
        incremental_cases=incremental_cases,
        cannibalised_paise=cannibalised_paise,
        cost_paise=sum(r.cost_paise for r in results),
        n_actions=sum(r.n_actions for r in results),
        n_contacts=sum(r.n_contacts for r in results),
        n_retries=sum(r.n_retries for r in results),
        n_violations=sum(r.n_violations for r in results),
        n_vetoes=sum(r.n_vetoes for r in results),
    )


def _rs(paise: int) -> str:
    """Indian-style short currency formatting."""
    rupees = paise / 100
    if rupees >= 1_00_00_000:
        return f"Rs {rupees / 1_00_00_000:.2f}Cr"
    if rupees >= 1_00_000:
        return f"Rs {rupees / 1_00_000:.2f}L"
    if rupees >= 1_000:
        return f"Rs {rupees / 1_000:.1f}k"
    return f"Rs {rupees:.0f}"


def table(metrics: list[Metrics]) -> str:
    rows = [
        ("arm", lambda m: m.arm.replace("_", " ")),
        ("recovered", lambda m: _rs(m.recovered_paise)),
        ("incremental", lambda m: _rs(m.incremental_paise)),
        ("incr. rate", lambda m: f"{m.incremental_rate:6.2%}"),
        ("cases won", lambda m: f"{m.incremental_cases}"),
        ("contacts", lambda m: f"{m.n_contacts}"),
        ("per case", lambda m: f"{m.contacts_per_case:.2f}"),
        ("retries", lambda m: f"{m.n_retries}"),
        ("spend", lambda m: _rs(m.cost_paise)),
        ("cost/Re won", lambda m: ("n/a" if m.incremental_paise <= 0
                                   else f"{m.cost_per_incremental_rupee:.4f}")),
        ("violations", lambda m: f"{m.n_violations}"),
    ]
    label_w = max(len(label) for label, _ in rows)
    col_w = max(14, max(len(m.arm.replace("_", " ")) for m in metrics) + 2)

    lines = []
    for label, fn in rows:
        cells = "".join(str(fn(m)).rjust(col_w) for m in metrics)
        lines.append(f"{label.rjust(label_w)}  {cells}")
        if label == "arm":
            lines.append("-" * (label_w + 2 + col_w * len(metrics)))
    return "\n".join(lines)


def headline(metrics: list[Metrics]) -> str:
    """The sentence that opens the pitch video."""
    by_name = {m.arm: m for m in metrics}
    agent = next((m for m in metrics if m.arm.startswith("B3")), None)
    b1 = by_name.get("B1_fixed_dunning")
    if agent is None or b1 is None:
        return ""
    lift = agent.incremental_paise - b1.incremental_paise
    pct = (lift / b1.incremental_paise * 100) if b1.incremental_paise else 0.0
    contact_delta = (
        (b1.contacts_per_case - agent.contacts_per_case) / b1.contacts_per_case * 100
        if b1.contacts_per_case else 0.0
    )
    return (
        f"Across {agent.n_cases:,} at-risk cases worth {_rs(agent.at_risk_paise)}, "
        f"the agent recovered {_rs(agent.incremental_paise)} incremental "
        f"({agent.incremental_rate:.1%} of value at risk) - "
        f"{_rs(lift)} more than a standard dunning ladder ({pct:+.0f}%), "
        f"using {contact_delta:.0f}% fewer customer contacts, "
        f"with {agent.n_violations} compliance violations."
    )


SEGMENT_LABELS = {
    "payment_failure": "Failed payments",
    "mandate_failure": "Bounced mandates",
    "receivable_overdue": "Overdue invoices",
}


def segment_report(arms: list[ArmResult], baseline: ArmResult) -> str:
    """Per-segment scoreboard.

    Reported separately because a single portfolio number is meaningless when
    one segment carries most of the value at a hundred times the ticket size.
    """
    out = []
    segments = [s for s in SEGMENT_LABELS if any(r.segment == s for r in baseline.results)]
    for seg in segments:
        ms = [compute(a, baseline, segment=seg) for a in arms]
        n = ms[0].n_cases
        out.append(
            f"\n{SEGMENT_LABELS[seg]}  -  {n:,} cases, "
            f"{_rs(ms[0].at_risk_paise)} at risk"
        )
        out.append(table(ms))
    return "\n".join(out)
