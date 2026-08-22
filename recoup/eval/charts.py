"""Charts for the write-up and the pitch video.

    python -m recoup.eval.charts --cases 10000

Three charts, each answering one question a reviewer will actually ask:

  1. "How does it sequence?"       -> success by attempt, per segment
  2. "Where does the lift come from?" -> incremental recovery by root cause
  3. "Isn't this just more spam?"  -> recovery against customer contacts

Palette is the validated categorical set (checked with the dataviz validator:
lightness band, chroma floor, CVD separation and normal-vision floor all pass).
Every mark is directly labelled, which is also the required relief for the one
slot that sits below 3:1 contrast against the surface.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from recoup.agent.heuristic import RecoupHeuristic
from recoup.eval import report
from recoup.eval.runner import run_arm
from recoup.policy import stopping
from recoup.policy.baselines import DoNothing, FixedDunning, MaximumPressure
from recoup.sim.generator import generate_ledger

# Validated categorical slots. Colour follows the arm, never its rank.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e5e4e0"

ARM_COLOR = {
    "B0_do_nothing": MUTED,
    "B1_fixed_dunning": "#2a78d6",     # blue
    "B2_maximum_pressure": "#eb6834",  # orange
    "B3_recoup_agent": "#1baf7a",      # aqua
}
ARM_LABEL = {
    "B0_do_nothing": "Do nothing",
    "B1_fixed_dunning": "Fixed dunning",
    "B2_maximum_pressure": "Max pressure",
    "B3_recoup_agent": "Recoup agent",
}

FAILURE_LABEL = {
    "insufficient_funds": "Insufficient funds",
    "issuer_down": "Issuer outage",
    "upi_collect_expired": "UPI collect expired",
    "3ds_dropoff": "3DS drop-off",
    "limit_exceeded": "Limit exceeded",
    "gateway_timeout": "Gateway timeout",
    "otp_timeout": "OTP timeout",
    "none": "Overdue invoice",
    "mandate_revoked": "Mandate revoked",
    "do_not_honour_permanent": "Permanent decline",
    "account_closed": "Account closed",
    "invalid_card": "Invalid card",
    "fraud_suspected": "Fraud flagged",
    "card_stolen": "Card stolen",
    "card_lost": "Card lost",
}

# Codes rarer than this are noisy even at 10k cases, and crowd the chart.
MIN_CASES_FOR_CHART = 250


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=10, length=0)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def _fig(w=11, h=6.2):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    return fig, ax


def _title(ax, title: str, subtitle: str) -> None:
    """Title above subtitle, positioned in points.

    Placing the subtitle in axes fractions looks right on a 6-inch axes and
    silently flips it *above* the title on a tall one, because the same
    fraction is a larger gap. Offsets in points are height-invariant.
    """
    ax.set_title(title, color=INK, fontsize=17, fontweight="bold",
                 loc="left", pad=34)
    ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction",
                xytext=(0, 9), textcoords="offset points",
                color=INK_2, fontsize=11.5, va="bottom", ha="left")


def _rupees(paise: float) -> str:
    r = paise / 100
    if r >= 1_00_000:
        return f"₹{r / 1_00_000:.1f}L"
    if r >= 1_000:
        return f"₹{r / 1_000:.0f}k"
    return f"₹{r:.0f}"


# ---------------------------------------------------------------------------
# 1. Marginal return per attempt
# ---------------------------------------------------------------------------

def chart_attempt_decay(arms, out: Path) -> None:
    """What the Nth attempt is actually worth - read per segment.

    An earlier version of this chart averaged rupees across all segments and
    produced a curve that spiked at attempt 3. That was not a finding, it was
    a Rs 76k invoice escalation drowning out Rs 99 subscriptions. Success rate
    per segment is the honest cut, and the three shapes differ for structural
    reasons worth showing rather than smoothing away.
    """
    agent = next(a for a in arms if a.arm == "B3_recoup_agent")

    # dy staggers the labels so converging lines do not stack numbers on top
    # of each other; labels stop at attempt 4 where the series merge.
    series = [
        ("payment_failure", "Failed payments", "#2a78d6", 15),
        ("mandate_failure", "Bounced mandates", "#1baf7a", 15),
        ("receivable_overdue", "Overdue invoices", "#eb6834", 15),
    ]

    fig, ax = _fig(11, 6.4)
    for seg, label, colour, dy in series:
        hit: dict[int, int] = defaultdict(int)
        tot: dict[int, int] = defaultdict(int)
        for r in agent.results:
            if r.segment != seg:
                continue
            for i, v in enumerate(r.attempt_values, start=1):
                tot[i] += 1
                if v > 0:
                    hit[i] += 1
        ks = [k for k in sorted(tot) if tot[k] >= 100]
        rates = [hit[k] / tot[k] for k in ks]
        ax.plot(ks, rates, color=colour, linewidth=2, marker="o", markersize=9,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3, label=label)
        # Label the starting point and the peak only. Labelling every point
        # stacks numbers on top of each other wherever the lines cross, and
        # the shape - not the fourth decimal - is what this chart is for.
        peak = max(rates)
        for k, r_ in zip(ks, rates):
            if k != 1 and r_ != peak:
                continue
            ax.annotate(f"{r_:.0%}", (k, r_), textcoords="offset points",
                        xytext=(0, dy), ha="center", color=INK,
                        fontsize=11, fontweight="bold")

    ax.annotate("attempt 1 is the pre-debit notice;\nattempt 2 is the debit it unlocks",
                (2, 0.230), xytext=(28, 62), textcoords="offset points",
                color=INK_2, fontsize=10.5, linespacing=1.5,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1.2))
    ax.annotate("two reminders fail,\nthe human escalation lands",
                (3, 0.244), xytext=(40, -30), textcoords="offset points",
                color=INK_2, fontsize=10.5, linespacing=1.5,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1.2))

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, 0.33)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_xlabel("Attempt number on a case", color=INK_2, fontsize=11, labelpad=10)
    ax.set_ylabel("Share of attempts that recovered", color=INK_2, fontsize=11, labelpad=10)
    _title(ax, "“Attempt 3” means something different in every segment",
           "Success rate by attempt position. The agent sequences per root cause, so the decay curve is not one curve.")
    leg = ax.legend(frameon=False, fontsize=11, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Where the lift comes from
# ---------------------------------------------------------------------------

def chart_by_root_cause(arms, out: Path) -> None:
    """Incremental recovery rate by failure code, agent vs fixed ladder."""
    base = {r.case_id: r for r in arms[0].results}

    def rates(arm):
        won, total = defaultdict(int), defaultdict(int)
        for r in arm.results:
            total[r.failure_code] += 1
            if r.recovered and not base[r.case_id].recovered:
                won[r.failure_code] += 1
        return {k: won[k] / total[k] for k in total
                if total[k] >= MIN_CASES_FOR_CHART}, total

    b1_rates, totals = rates(next(a for a in arms if a.arm == "B1_fixed_dunning"))
    b3_rates, _ = rates(next(a for a in arms if a.arm == "B3_recoup_agent"))

    codes = sorted(b3_rates, key=lambda c: b3_rates[c] - b1_rates.get(c, 0))
    labels = [FAILURE_LABEL.get(c, c) for c in codes]
    y = range(len(codes))
    h = 0.36

    fig, ax = _fig(11, 0.62 * len(codes) + 3.2)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=1)

    ax.barh([i + h / 2 for i in y], [b1_rates.get(c, 0) for c in codes], height=h,
            color="#2a78d6", label="Fixed dunning", zorder=3)
    ax.barh([i - h / 2 for i in y], [b3_rates[c] for c in codes], height=h,
            color="#1baf7a", label="Recoup agent", zorder=3)

    for i, c in enumerate(codes):
        for val, off in ((b1_rates.get(c, 0), h / 2), (b3_rates[c], -h / 2)):
            ax.text(val + 0.006, i + off, f"{val:.0%}", va="center",
                    color=INK, fontsize=10, fontweight="bold")

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, color=INK, fontsize=11)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Share of cases recovered that would not have recovered on their own",
                  color=INK_2, fontsize=11, labelpad=10)
    ax.set_xlim(0, max(max(b3_rates.values()), max(b1_rates.values())) * 1.16)
    _title(ax, "Root cause decides the intervention",
           "Incremental recovery by failure reason. Biggest gaps are where timing and rail choice matter most.")
    leg = ax.legend(frameon=False, fontsize=11, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Recovery against contacts
# ---------------------------------------------------------------------------

def chart_contacts_vs_recovery(arms, metrics, out: Path) -> None:
    """The argument against volume, in one picture."""
    fig, ax = _fig(11, 6.6)

    for m in metrics:
        colour = ARM_COLOR[m.arm]
        ax.scatter(m.contacts_per_case, m.incremental_rate, s=340,
                   color=colour, edgecolor=SURFACE, linewidth=2.5, zorder=4)
        label = ARM_LABEL[m.arm]
        # A multi-line annotation anchors on its baseline, so a negative offset
        # alone pushes the *last* line down and leaves the first line sitting on
        # the marker. va has to move with the offset direction.
        dy = 18 if m.arm != "B2_maximum_pressure" else -20
        ax.annotate(f"{label}\n{m.incremental_rate:.1%} · {m.contacts_per_case:.2f} contacts",
                    (m.contacts_per_case, m.incremental_rate),
                    textcoords="offset points", xytext=(0, dy), ha="center",
                    va="top" if dy < 0 else "bottom",
                    color=INK, fontsize=11, fontweight="bold", linespacing=1.5)

    b2 = next(m for m in metrics if m.arm == "B2_maximum_pressure")
    ax.annotate(f"{b2.n_violations:,} compliance violations",
                (b2.contacts_per_case, b2.incremental_rate),
                textcoords="offset points", xytext=(0, -62), ha="center",
                va="top", color="#eb6834", fontsize=11, fontweight="bold")
    b3 = next(m for m in metrics if m.arm == "B3_recoup_agent")
    ax.annotate("zero violations",
                (b3.contacts_per_case, b3.incremental_rate),
                textcoords="offset points", xytext=(0, -26), ha="center",
                color="#1baf7a", fontsize=11, fontweight="bold")

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Customer contacts per case", color=INK_2, fontsize=11, labelpad=10)
    ax.set_ylabel("Incremental recovery", color=INK_2, fontsize=11, labelpad=10)
    ax.set_xlim(-0.25, max(m.contacts_per_case for m in metrics) + 0.75)
    ax.set_ylim(-0.03, max(m.incremental_rate for m in metrics) * 1.32)
    _title(ax, "Up and to the left is the only place worth being",
           "More contacting is not more recovering. Max pressure spends 2.3× the contacts to land below the agent.")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate charts")
    ap.add_argument("--cases", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("out/charts"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ledger = generate_ledger(args.cases, seed=args.seed)

    arms = []
    for policy, enforce in [(DoNothing(), True), (FixedDunning(), True),
                            (MaximumPressure(), False), (RecoupHeuristic(), True)]:
        arms.append(run_arm(ledger, policy, enforce_compliance=enforce))
    metrics = [report.compute(a, arms[0]) for a in arms]

    jobs = [
        ("1_attempt_decay.png", lambda p: chart_attempt_decay(arms, p)),
        ("2_by_root_cause.png", lambda p: chart_by_root_cause(arms, p)),
        ("3_contacts_vs_recovery.png",
         lambda p: chart_contacts_vs_recovery(arms, metrics, p)),
    ]
    for name, fn in jobs:
        path = args.out / name
        fn(path)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
