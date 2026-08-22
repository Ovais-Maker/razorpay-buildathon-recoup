"""Run every arm over one ledger and print the scoreboard.

    python -m recoup.eval.run_batch --cases 10000 --audit out/audit.jsonl
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from recoup.agent.heuristic import RecoupHeuristic
from recoup.eval import report
from recoup.eval.runner import run_arm
from recoup.policy.baselines import DoNothing, FixedDunning, MaximumPressure
from recoup.sim.generator import generate_ledger, ledger_value_paise

# (policy, enforce_compliance). B2 runs unguarded on purpose - that is the
# only way to show what the guardrail is actually buying.
ARMS = [
    (DoNothing(), True),
    (FixedDunning(), True),
    (MaximumPressure(), False),
    (RecoupHeuristic(), True),
]


def run_llm_comparison(ledger, args) -> None:
    """Score the LLM strategist against the same arms on a shared sub-batch.

    Every arm is re-run on the identical subset. Comparing an LLM arm scored
    over 500 cases against baselines scored over 10,000 would be meaningless,
    so the sub-batch is explicit rather than implied.
    """
    from recoup.agent.llm import LLMStrategist, LLMUnavailable
    from recoup.sim.generator import Ledger

    n = min(args.llm_cases, len(ledger.records))
    sub = Ledger(records=ledger.records[:n], outages=ledger.outages)

    print("\n" + "=" * report.MAX_WIDTH)
    print(f"LLM STRATEGIST COMPARISON  -  {n:,} cases, model {args.llm}")
    print("=" * report.MAX_WIDTH)

    strategist = LLMStrategist(
        model=args.llm, cache_path=args.llm_cache, offline=args.llm_offline,
    )

    try:
        # Fail in two seconds on a bad key, not part-way through the batch.
        strategist.preflight()
    except LLMUnavailable as exc:
        print(f"\n  LLM arm skipped: {exc}")
        return

    sub_arms = []
    for policy, enforce in ARMS:
        sub_arms.append(run_arm(sub, policy, enforce_compliance=enforce))

    try:
        sub_arms.append(run_arm(sub, strategist, enforce_compliance=True,
                                max_workers=args.llm_workers))
    except LLMUnavailable as exc:
        print(f"\n  LLM arm skipped: {exc}")
        print("  Set ANTHROPIC_API_KEY, then re-run. Everything else above is unaffected.")
        return
    finally:
        strategist.close()

    base = sub_arms[0]
    print()
    print(report.table([report.compute(a, base) for a in sub_arms]))
    print(f"\n  {strategist.usage.summary(args.llm)}")
    if args.llm_cache:
        print(f"  decision cache: {args.llm_cache}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Recoup batch evaluation")
    ap.add_argument("--cases", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--audit", type=Path, default=None,
                    help="write the agent's audit chain to this JSONL path")
    ap.add_argument("--audit-cases", type=int, default=200,
                    help="how many cases to write full audit records for")
    ap.add_argument("--llm", metavar="MODEL", default=None,
                    help="also run an LLM strategist arm, e.g. claude-haiku-4-5")
    ap.add_argument("--llm-cases", type=int, default=500,
                    help="sub-batch size for the LLM comparison (it costs money)")
    ap.add_argument("--llm-cache", type=Path, default=Path("out/decisions.json"),
                    help="decision cache; reused across runs to cut spend")
    ap.add_argument("--llm-workers", type=int, default=8,
                    help="concurrent cases for the LLM arm")
    ap.add_argument("--llm-offline", action="store_true",
                    help="use only cached decisions, never call the API")
    args = ap.parse_args()

    t0 = time.perf_counter()
    ledger = generate_ledger(args.cases, seed=args.seed)
    print(f"ledger: {len(ledger):,} at-risk cases worth "
          f"{report._rs(ledger_value_paise(ledger))}\n")

    audit_ids = {r.case.case_id for r in ledger.records[:args.audit_cases]}

    arm_results = []
    for policy, enforce in ARMS:
        is_agent = policy.name.startswith("B3")
        result = run_arm(
            ledger, policy,
            enforce_compliance=enforce,
            audit_path=args.audit if (is_agent and args.audit) else None,
            audit_case_ids=audit_ids,
        )
        arm_results.append(result)
        print(f"  ran {policy.name:22s} guardrail={'on ' if enforce else 'OFF'}")

    baseline = arm_results[0]
    metrics = [report.compute(a, baseline) for a in arm_results]

    print("PORTFOLIO")
    print(report.table(metrics))
    print()
    print(report.segment_report(arm_results, baseline))
    print()
    print(report.headline(metrics))
    print(f"\ncompleted in {time.perf_counter() - t0:.1f}s")

    if args.llm:
        run_llm_comparison(ledger, args)

    agent_arm = arm_results[-1]
    if agent_arm.ledger is not None:
        ok, detail = agent_arm.ledger.verify()
        print(f"audit chain: {'VERIFIED' if ok else 'BROKEN'} - {detail}")
        if args.audit:
            print(f"audit written to {args.audit}")


if __name__ == "__main__":
    main()
