"""Walk a single case from detection to cash, straight out of the audit chain.

    python -m recoup.eval.explain_case                 # pick a good example
    python -m recoup.eval.explain_case --case case_000123

This is the view to screen-record: pick one recovered rupee, show every
decision, every compliance check, and the hash that links them.
"""
from __future__ import annotations

import argparse
from datetime import datetime

from recoup.agent.heuristic import RecoupHeuristic
from recoup.audit.ledger import AuditLedger
from recoup.domain import CaseType, FailureCode, Method
from recoup.eval.runner import run_case
from recoup.policy.base import Context
from recoup.sim.generator import generate_ledger
from recoup.sim.issuer_health import IssuerHealth

RULER = "-" * 78


def _rs(paise: int) -> str:
    return f"Rs {paise / 100:,.2f}"


def _pick(records, want: str):
    """Find a case that shows off a particular mechanism."""
    def match(r):
        c = r.case
        if want == "mandate":
            return (c.method is Method.EMANDATE
                    and c.failure_code is FailureCode.INSUFFICIENT_FUNDS
                    and c.observed_salary_day is not None
                    and not c.opted_out)
        if want == "issuer":
            return c.failure_code is FailureCode.ISSUER_DOWN and not c.opted_out
        if want == "hard":
            return c.is_hard_decline and not c.opted_out
        if want == "invoice":
            return c.case_type is CaseType.RECEIVABLE_OVERDUE and not c.opted_out
        return True
    return [r for r in records if match(r)]


def render(audit: AuditLedger, case_id: str) -> None:
    records = audit.for_case(case_id)
    if not records:
        print(f"no audit records for {case_id}")
        return

    first = records[0].payload
    state = first["state"]
    print(RULER)
    print(f"CASE {case_id}   {state['case_type']}   {_rs(state['amount_paise'])}")
    print(f"  rail={state['method']}  issuer={state['issuer']}  "
          f"failure={state['failure_code']}")
    print(f"  inferred salary day={state['observed_salary_day']}  "
          f"DND={state['dnd_registered']}  opted_out={state['opted_out']}")
    print(RULER)

    for rec in records:
        p = rec.payload
        ts = datetime.fromisoformat(p["ts"])
        proposed = p.get("proposed_action")
        final = p.get("final_action")
        outcome = p.get("outcome", {})

        print(f"\n[{rec.seq:>4}] {ts:%d %b %H:%M}   hash {rec.hash[:12]}...")

        if proposed:
            print(f"       proposed : {proposed['type']}"
                  + (f" via {proposed['channel']}" if proposed.get("channel") else "")
                  + (f" on {proposed['rail']}" if proposed.get("rail") else "")
                  + (f" [{proposed['template_id']}]" if proposed.get("template_id") else ""))
            if proposed.get("reason"):
                print(f"       because  : {proposed['reason']}")

        checks = p.get("guardrail_checks") or []
        if checks:
            failed = [c for c in checks if c["result"] != "pass"]
            passed = len(checks) - len(failed)
            print(f"       guardrail: {passed}/{len(checks)} passed", end="")
            print(" - ALL CLEAR" if not failed else "")
            for c in failed:
                print(f"                  VETO {c['rule']}: {c['detail']}")

        if outcome.get("vetoed"):
            print(f"       action   : blocked, replanning")
        elif final:
            cost = final.get("cost_paise", 0)
            print(f"       executed : {final['type']}  (cost {_rs(cost)})")
            if "p_success" in outcome:
                print(f"       odds     : {outcome['p_success']:.1%}  ({outcome.get('note','')})")
            print(f"       result   : {outcome.get('result', '-')}")
        elif "stopped" in outcome:
            print(f"       STOPPED  : {outcome['stopped']}")
        elif outcome.get("recovered"):
            print(f"       result   : recovered - {outcome.get('note', '')}")

    print()
    print(RULER)
    ok, detail = audit.verify()
    print(f"audit chain: {'VERIFIED' if ok else 'BROKEN'} - {detail}")
    print(RULER)


def main() -> None:
    ap = argparse.ArgumentParser(description="Explain one recovery case")
    ap.add_argument("--case", default=None, help="case id, e.g. case_000123")
    ap.add_argument("--kind", default="mandate",
                    choices=["mandate", "issuer", "hard", "invoice", "any"],
                    help="what kind of example to pick when --case is omitted")
    ap.add_argument("--cases", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ledger = generate_ledger(args.cases, seed=args.seed)
    ctx = Context(
        issuer_health=IssuerHealth.from_outages(ledger.outages, seed=11),
        horizon=datetime.max,
    )
    policy = RecoupHeuristic()

    if args.case:
        chosen = [r for r in ledger.records if r.case.case_id == args.case]
        if not chosen:
            print(f"no such case: {args.case}")
            return
    else:
        candidates = _pick(ledger.records, args.kind)
        # Prefer a case that recovered *after* a few decisions - a one-message
        # win demonstrates nothing about sequencing or the guardrail.
        best, best_len = None, -1
        for r in candidates[:600]:
            probe = AuditLedger()
            res = run_case(r, policy, ctx, audit=probe)
            n = len(probe.for_case(r.case.case_id))
            score = n + (10 if res.recovered else 0)
            if score > best_len:
                best, best_len = r, score
            if res.recovered and n >= 5:
                best = r
                break
        chosen = [best] if best is not None else candidates[:1]
        if not chosen:
            print("no matching case found")
            return

    audit = AuditLedger()
    result = run_case(chosen[0], policy, ctx, audit=audit)
    render(audit, chosen[0].case.case_id)
    print(f"outcome: {'RECOVERED ' + _rs(result.recovered_paise) if result.recovered else 'not recovered'}"
          f"   spend {_rs(result.cost_paise)}"
          f"   actions {result.n_actions}   vetoes {result.n_vetoes}"
          f"   stop reason: {result.stop_reason.value if result.stop_reason else '-'}")


if __name__ == "__main__":
    main()
