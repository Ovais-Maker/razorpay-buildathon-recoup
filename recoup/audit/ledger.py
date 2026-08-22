"""Hash-chained, append-only audit trail.

Every decision the system makes is written here before it is acted on: the
state it saw, what it proposed, every compliance check it ran, what it
actually did, and what happened. Each record carries the hash of the one
before it, so a record cannot be altered or removed after the fact without
breaking the chain.

The chain is what turns "trust our agent" into "audit our agent".
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

GENESIS = "0" * 64


def _canonical(payload: dict[str, Any]) -> str:
    """Stable serialisation - key order must never vary or hashes drift."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _digest(seq: int, prev_hash: str, payload: dict[str, Any]) -> str:
    material = f"{seq}|{prev_hash}|{_canonical(payload)}".encode()
    return hashlib.sha256(material).hexdigest()


@dataclass
class Record:
    seq: int
    prev_hash: str
    hash: str
    payload: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {"seq": self.seq, "prev_hash": self.prev_hash,
             "hash": self.hash, "payload": self.payload},
            sort_keys=True, default=str,
        )


@dataclass
class AuditLedger:
    """In-memory chain with optional JSONL persistence."""
    path: Path | None = None
    records: list[Record] = field(default_factory=list)
    _fh: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("w", encoding="utf-8")

    @property
    def head(self) -> str:
        return self.records[-1].hash if self.records else GENESIS

    _lock: Any = field(default_factory=threading.Lock, repr=False)

    def append(self, payload: dict[str, Any]) -> Record:
        with self._lock:
            return self._append_locked(payload)

    def _append_locked(self, payload: dict[str, Any]) -> Record:
        seq = len(self.records)
        prev = self.head
        rec = Record(seq=seq, prev_hash=prev, hash=_digest(seq, prev, payload), payload=payload)
        self.records.append(rec)
        if self._fh is not None:
            self._fh.write(rec.to_json() + "\n")
        return rec

    def decision(
        self,
        *,
        case_id: str,
        arm: str,
        now: datetime,
        state: dict[str, Any],
        diagnosis: dict[str, Any] | None,
        proposed: dict[str, Any] | None,
        checks: list[dict[str, Any]],
        executed: dict[str, Any] | None,
        outcome: dict[str, Any],
        policy_version: str,
        model: str | None = None,
    ) -> Record:
        """Write one decision record in the standard shape."""
        return self.append({
            "ts": now.isoformat(),
            "case_id": case_id,
            "arm": arm,
            "state": state,
            "diagnosis": diagnosis,
            "proposed_action": proposed,
            "guardrail_checks": checks,
            "final_action": executed,
            "outcome": outcome,
            "policy_version": policy_version,
            "model": model,
        })

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def for_case(self, case_id: str) -> list[Record]:
        return [r for r in self.records if r.payload.get("case_id") == case_id]

    def verify(self) -> tuple[bool, str]:
        """Recompute the whole chain. Any edit anywhere shows up here."""
        prev = GENESIS
        for i, rec in enumerate(self.records):
            if rec.seq != i:
                return False, f"record {i} has seq {rec.seq}"
            if rec.prev_hash != prev:
                return False, f"record {i} prev_hash does not match record {i - 1}"
            expected = _digest(rec.seq, rec.prev_hash, rec.payload)
            if rec.hash != expected:
                return False, f"record {i} payload does not match its hash"
            prev = rec.hash
        return True, f"chain intact across {len(self.records)} records"


def read_chain(path: Path) -> Iterator[Record]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                yield Record(d["seq"], d["prev_hash"], d["hash"], d["payload"])
