"""LLM strategist.

Drop-in replacement for `RecoupHeuristic`: same `propose` signature, same
`Action` out. Everything that makes the system safe - the guardrail, the
economic stop, the audit chain - sits outside this file and is unchanged by
swapping the brain. That was the point of building them first.

Three things make this practical at batch scale:

  * **Prompt caching, where the model supports it.** The system prompt is a
    byte-stable prefix with one breakpoint; per-case facts go after it. NOTE:
    measured, not assumed - at ~1.4K tokens this prefix is below Haiku 4.5's
    4096-token minimum cacheable prefix, so caching silently does not engage
    on that model (it does on Opus 5, minimum 512). The marker is left in
    place because it costs nothing and activates on capable models. The
    saving at this prompt size is cents; concurrency was the real lever.
  * **A decision cache.** Identical case states reuse a decision. A 10k-case
    batch has far fewer distinct states than decisions, so this cuts spend
    hard and makes a run reproducible.
  * **Fallback to the heuristic.** An API error on case 4,312 must not kill a
    batch. Failures are counted and reported, never swallowed silently.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading

import pydantic
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from recoup.agent.heuristic import RecoupHeuristic
from recoup.agent.prompts import build_system_prompt, build_user_message
from recoup.agent.schema import RecoveryDecision, to_action
from recoup.domain import Action, ActionType, Attempt, Case
from recoup.policy import stopping
from recoup.policy.base import Context

# Input / output USD per million tokens, for the spend report.
PRICING = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

DEFAULT_MODEL = "claude-opus-5"
MAX_TOKENS = 2048


class LLMUnavailable(RuntimeError):
    """No credentials, or the SDK is not installed."""


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0          # served from our own decision cache
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cache_misses: int = 0        # offline replay only: state not in the cache
    failures: int = 0
    fallbacks: int = 0

    def cost_usd(self, model: str) -> float:
        rate_in, rate_out = PRICING.get(model, PRICING[DEFAULT_MODEL])
        # Cache reads bill at a fraction of input; treated as input here so the
        # figure stays an upper bound rather than an optimistic one.
        billed_in = self.input_tokens + self.cache_read_tokens + self.cache_write_tokens
        return (billed_in / 1e6) * rate_in + (self.output_tokens / 1e6) * rate_out

    def summary(self, model: str) -> str:
        total = self.calls + self.cache_hits
        hit_rate = self.cache_hits / total if total else 0.0
        if self.calls == 0 and (self.cache_hits or self.cache_misses):
            replayed = self.cache_hits + self.cache_misses
            miss = self.cache_misses / replayed if replayed else 0.0
            return (
                f"REPLAY of recorded decisions - no API calls, no key required\n"
                f"  {self.cache_hits:,} real model decisions replayed, "
                f"{self.cache_misses:,} misses ({miss:.1%} fell back to the heuristic)"
            ) + (
                # A mostly-missing replay is the heuristic wearing the LLM's
                # name. Say so loudly rather than let the column be misread.
                "\n\n  *** WARNING: this column is NOT a valid LLM result ***\n"
                f"  {miss:.0%} of decisions came from the fallback policy, not the model.\n"
                "  The recorded cache was built against one specific ledger. Use:\n"
                "    --cases 500 --llm claude-haiku-4-5 --llm-cases 120 --llm-offline\n"
                "  or drop --llm-offline to record fresh decisions for your ledger."
                if miss > 0.10 else ""
            )
        return (
            f"{self.calls:,} API calls ({self.cache_hits:,} served from the "
            f"decision cache, {hit_rate:.0%} hit rate)\n"
            f"  tokens: {self.input_tokens:,} in / {self.output_tokens:,} out / "
            f"{self.cache_read_tokens:,} cache-read\n"
            f"  estimated spend: ${self.cost_usd(model):.2f}\n"
            f"  failures: {self.failures} (fell back to heuristic {self.fallbacks} times)"
        )


def _state_key(case: Case, history: list[Attempt], now: datetime,
               fingerprint: str = "") -> str:
    """Identity of a decision point.

    Deliberately coarse on time - rounding to the hour means two cases at the
    same point in the same situation share a decision, which is what makes the
    cache effective. Anything that should change the answer is in the key.
    """
    material = {
        # Without this, editing the prompt silently reuses decisions made by
        # the previous one and prompt changes cannot be measured at all.
        "fingerprint": fingerprint,
        "type": case.case_type.value,
        "amount_bucket": case.amount_paise // 10000,
        "method": case.method.value,
        "failure": case.failure_code.value,
        "salary_day": case.observed_salary_day,
        "whatsapp": case.consent_whatsapp,
        "dnd": case.dnd_registered,
        "opted_out": case.opted_out,
        "day_of_month": now.day,
        "hour": now.hour,
        "age_h": int((now - case.created_at).total_seconds() // 3600),
        "history": [
            (a.action.type.value,
             a.action.channel.value if a.action.channel else None,
             a.action.template_id, a.succeeded)
            for a in history
        ],
    }
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


class DecisionCache:
    """On-disk cache of model decisions, keyed by state."""

    def __init__(self, path: Path | None):
        self.path = path
        self.lock = threading.Lock()   # cases run concurrently
        self.entries: dict[str, dict] = {}
        if path and path.exists():
            with path.open(encoding="utf-8") as fh:
                self.entries = json.load(fh)

    def get(self, key: str) -> RecoveryDecision | None:
        with self.lock:
            raw = self.entries.get(key)
        return RecoveryDecision.model_validate(raw) if raw else None

    def put(self, key: str, decision: RecoveryDecision) -> None:
        with self.lock:
            self.entries[key] = decision.model_dump()

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as fh:
                json.dump(self.entries, fh, indent=0, sort_keys=True)


class LLMStrategist:
    """Policy whose planning step is a Claude call."""

    version = "v2-llm"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        cache_path: Path | None = None,
        offline: bool = False,
        fallback: bool = True,
    ):
        self.model = model
        self.name = f"B4_recoup_llm[{model}]"
        self.cache = DecisionCache(cache_path)
        self.usage = Usage()
        self.offline = offline
        self.fallback_policy = RecoupHeuristic() if fallback else None
        self.system_prompt = build_system_prompt()
        # Decisions are only interchangeable if the model and the prompt that
        # produced them are the same.
        self.fingerprint = hashlib.sha256(
            (model + "|" + self.system_prompt).encode()
        ).hexdigest()[:12]
        self._client = None

    # -- client ---------------------------------------------------------
    @property
    def client(self):
        if self._client is None:
            if self.offline:
                raise LLMUnavailable("running offline - decision cache only")
            try:
                import anthropic
            except ImportError as exc:
                raise LLMUnavailable("pip install anthropic") from exc
            if not (os.environ.get("ANTHROPIC_API_KEY")
                    or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
                raise LLMUnavailable(
                    "no credentials - set ANTHROPIC_API_KEY or run `ant auth login`"
                )
            self._client = anthropic.Anthropic()
        return self._client

    def preflight(self) -> None:
        """Prove the model is reachable before committing to a batch.

        One cheap call. Without it, a bad key or an exhausted balance is only
        discovered part-way through a run, after real time has been spent and
        with a partial ledger to throw away.
        """
        if self.offline:
            return
        import anthropic
        try:
            self.client.messages.create(
                model=self.model,
                max_tokens=4,
                messages=[{"role": "user", "content": "ok"}],
            )
        except anthropic.APIStatusError as exc:
            raise LLMUnavailable(
                f"preflight failed: {type(exc).__name__} ({exc.status_code})\n"
                f"    {getattr(exc, 'message', str(exc))}"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMUnavailable(f"preflight failed: cannot reach the API - {exc}") from exc

    # -- the policy interface -------------------------------------------
    def propose(
        self, case: Case, history: list[Attempt], now: datetime, ctx: Context
    ) -> Action:
        if stopping.is_paused(case, now):
            return Action(
                ActionType.WAIT, case.promise_to_pay_at,
                reason="promise to pay recorded - staying quiet until the date",
            )

        key = _state_key(case, history, now, self.fingerprint)
        decision = self.cache.get(key)

        if decision is not None:
            self.usage.cache_hits += 1
        else:
            try:
                decision = self._ask(case, history, now, ctx)
            except LLMUnavailable:
                # Offline replay: a miss means the cache was recorded against a
                # different ledger. Fall back and report the miss rate rather
                # than aborting - a partial replay is still worth having.
                # Note this wraps the *call* rather than short-circuiting
                # before it, so a subclass that overrides `_ask` (the test
                # stubs) still gets to answer.
                if self.offline:
                    self.usage.cache_misses += 1
                    return self._fallback(case, history, now, ctx)
                raise
            if decision is None:
                return self._fallback(case, history, now, ctx)
            self.cache.put(key, decision)

        action = to_action(decision, case, now)
        return self._apply_economics(action, decision, case, history, ctx)

    # -- internals ------------------------------------------------------
    def _ask(
        self, case: Case, history: list[Attempt], now: datetime, ctx: Context
    ) -> RecoveryDecision | None:
        try:
            import anthropic
        except ImportError:
            self.usage.failures += 1
            return None

        try:
            response = self.client.messages.parse(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=[{
                    "type": "text",
                    "text": self.system_prompt,
                    # One breakpoint, at the end of the stable prefix. Every
                    # case-specific byte lives after it, in the user message.
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{
                    "role": "user",
                    "content": build_user_message(case, history, now, ctx),
                }],
                output_format=RecoveryDecision,
            )
        except LLMUnavailable:
            raise
        except pydantic.ValidationError:
            # A generation that does not satisfy the schema. Per-case and
            # transient, unlike a 4xx - count it and let this one case fall
            # back rather than losing the batch.
            self.usage.failures += 1
            return None
        except anthropic.RateLimitError:
            # Transient by definition - back off and fall back for this case.
            self.usage.failures += 1
            return None
        except anthropic.APIStatusError as exc:
            # The distinction that matters is not which error, but whether
            # retrying could ever help. A 400 (bad request, exhausted credit),
            # 401/403 (bad key, no permission) or 404 (wrong or retired model)
            # will fail identically on every remaining case. Falling back
            # silently would turn one misconfiguration into ten thousand
            # heuristic decisions and a run that looks like it worked.
            if exc.status_code < 500:
                raise LLMUnavailable(
                    f"{type(exc).__name__} ({exc.status_code}) - this will fail "
                    f"the same way on every case, so the run is stopping here.\n"
                    f"    {getattr(exc, 'message', str(exc))}"
                ) from exc
            self.usage.failures += 1      # 5xx: the server's problem, retry later
            return None
        except anthropic.APIConnectionError:
            self.usage.failures += 1
            return None

        u = response.usage
        self.usage.calls += 1
        self.usage.input_tokens += u.input_tokens
        self.usage.output_tokens += u.output_tokens
        self.usage.cache_read_tokens += getattr(u, "cache_read_input_tokens", 0) or 0
        self.usage.cache_write_tokens += getattr(u, "cache_creation_input_tokens", 0) or 0
        return response.parsed_output

    def _fallback(
        self, case: Case, history: list[Attempt], now: datetime, ctx: Context
    ) -> Action:
        self.usage.fallbacks += 1
        if self.fallback_policy is None:
            return Action(ActionType.STOP, now, reason="strategist unavailable")
        return self.fallback_policy.propose(case, history, now, ctx)

    def _apply_economics(
        self, action: Action, decision: RecoveryDecision,
        case: Case, history: list[Attempt], ctx: Context,
    ) -> Action:
        """The stopping rule uses the model's own p_success.

        Deliberate: the model is asked to be honest about its odds, and is then
        held to them. It cannot talk its way past the hurdle, because the
        arithmetic runs in code.
        """
        if action.type not in {ActionType.RETRY_PAYMENT, ActionType.SEND_MESSAGE,
                               ActionType.ESCALATE_HUMAN}:
            return action
        contacts = sum(
            1 for a in history
            if a.action.type in {ActionType.SEND_MESSAGE, ActionType.ESCALATE_HUMAN}
        )
        if not stopping.is_economic(
            case.amount_paise, decision.p_success, action.cost_paise(), contacts
        ):
            return Action(
                ActionType.STOP, action.at,
                reason="not economic: " + stopping.economic_stop_reason(
                    case.amount_paise, decision.p_success, action.cost_paise(), contacts
                ),
            )
        return action

    def close(self) -> None:
        self.cache.save()
