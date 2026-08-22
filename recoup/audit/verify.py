"""Verify a written audit chain, and prove it detects tampering.

    python -m recoup.audit.verify out/audit.jsonl

Reads the chain back off disk, checks every link, then edits one record and
re-checks so you can see the failure rather than take it on trust. An audit
trail nobody has tried to break is just a log file.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from recoup.audit.ledger import AuditLedger, read_chain

RULER = "-" * 74


def _load(path: Path) -> AuditLedger:
    led = AuditLedger()
    led.records = list(read_chain(path))
    return led


def _show_record(led: AuditLedger, index: int) -> None:
    rec = led.records[index]
    p = rec.payload
    print(f"record #{rec.seq}")
    print(f"  prev_hash : {rec.prev_hash[:32]}...")
    print(f"  hash      : {rec.hash[:32]}...")
    print(f"  case      : {p['case_id']}   {p['state']['failure_code']}"
          f"   Rs {p['state']['amount_paise'] / 100:,.2f}")
    action = p.get("final_action") or p.get("proposed_action") or {}
    if action:
        print(f"  action    : {action.get('type')} "
              f"{action.get('channel') or action.get('rail') or ''}")
        print(f"  because   : {action.get('reason', '')[:60]}")
    checks = p.get("guardrail_checks") or []
    passed = sum(1 for c in checks if c["result"] == "pass")
    print(f"  guardrail : {passed}/{len(checks)} checks recorded")
    print(f"  outcome   : {json.dumps(p.get('outcome', {}))[:64]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify an audit chain")
    ap.add_argument("path", nargs="?", type=Path, default=Path("out/audit.jsonl"))
    ap.add_argument("--record", type=int, default=None,
                    help="index of the record to tamper with (default: middle)")
    args = ap.parse_args()

    if not args.path.exists():
        print(f"no audit file at {args.path}")
        print("generate one with:")
        print("  python -m recoup.eval.run_batch --cases 10000 --audit out/audit.jsonl")
        return

    led = _load(args.path)
    target = args.record if args.record is not None else len(led.records) // 2

    print(RULER)
    print(f"AUDIT CHAIN  {args.path}")
    print(RULER)
    print(f"{len(led.records):,} decisions recorded, each carrying the hash of the one before it.\n")
    _show_record(led, target)

    print()
    ok, detail = led.verify()
    print(f"  verification: {'VERIFIED' if ok else 'BROKEN'} - {detail}")

    # Now break it. A chain nobody has tried to forge proves nothing.
    print(f"\n{RULER}")
    print("TAMPER TEST - rewriting the amount on one record")
    print(RULER)

    forged = AuditLedger()
    forged.records = copy.deepcopy(led.records)
    rec = forged.records[target]
    before = rec.payload["state"]["amount_paise"]
    rec.payload["state"]["amount_paise"] = 9_99_99_999

    print(f"record #{target}: Rs {before / 100:,.2f} -> Rs {9_99_99_999 / 100:,.2f}")
    print("(the stored hash is left untouched, exactly as a forger would)\n")

    ok2, detail2 = forged.verify()
    print(f"  verification: {'VERIFIED' if ok2 else 'BROKEN'} - {detail2}")
    print()
    if not ok2:
        print("Every later record chains off that hash, so one edit invalidates")
        print("the rest of the chain too. There is nowhere to hide a change.")
    print(RULER)


if __name__ == "__main__":
    main()
