"""Observable issuer health.

A merchant cannot see a bank's status page - they infer degradation from
their own rolling success rate per issuer. That inference lags reality in
both directions: an outage takes a few minutes of failures to become
visible, and recovery takes a few minutes of successes to confirm.

This class exposes exactly what a real monitor could know at time T, and
nothing more. In particular there is no `when_will_it_recover()` - the agent
has to wait and re-check, the same as a real system would.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

DETECTION_LAG_MIN = 25      # failures must accumulate before we call it down
RECOVERY_LAG_MIN = 20       # successes must accumulate before we call it up
POLL_INTERVAL_MIN = 45      # how often a waiting case re-checks


@dataclass
class IssuerHealth:
    """Lagged, observable view of per-issuer degradation."""
    observed: dict[str, tuple[datetime, datetime]]

    @classmethod
    def from_outages(
        cls, outages: dict[str, tuple[datetime, datetime]], seed: int = 11
    ) -> "IssuerHealth":
        rng = random.Random(seed)
        observed: dict[str, tuple[datetime, datetime]] = {}
        for issuer, (start, end) in outages.items():
            detect = start + timedelta(minutes=rng.uniform(DETECTION_LAG_MIN * 0.6,
                                                           DETECTION_LAG_MIN * 1.6))
            clear = end + timedelta(minutes=rng.uniform(RECOVERY_LAG_MIN * 0.6,
                                                        RECOVERY_LAG_MIN * 1.6))
            observed[issuer] = (detect, clear)
        return cls(observed=observed)

    def is_degraded(self, issuer: str, at: datetime) -> bool:
        window = self.observed.get(issuer)
        if window is None:
            return False
        return window[0] <= at < window[1]

    def health_score(self, issuer: str, at: datetime) -> float:
        """Rolling success rate proxy, as a dashboard would show it."""
        return 0.12 if self.is_degraded(issuer, at) else 0.94

    def next_poll(self, at: datetime) -> datetime:
        """When a case parked on a degraded issuer should look again."""
        return at + timedelta(minutes=POLL_INTERVAL_MIN)
