"""A canned-response stand-in for the LLM, used by every test.

Tests that hit the real model are slow, cost money, need a key, and -- worst of
all -- return a different answer every run, which makes them useless as tests
(build-plan Section 2). ``FakeLLM`` implements the same ``LLMClient`` interface
with scripted replies, so the exact code paths an agent uses in production run
in a test in microseconds, offline, deterministically.

Two things it is built to exercise:

* **The retry path.** Script a malformed reply followed by a good one and the
  structured-output layer's re-ask logic runs for real against the fake -- the
  most important behaviour to prove without a network.
* **Cost accounting.** It reports (estimated) token counts and fires the usage
  callback exactly like the real client, so usage-logging tests need no network.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from app.core.llm.base import (
    ChatMessage,
    GenerationParams,
    LLMClient,
    LLMResponse,
    ModelTier,
    UsageCallback,
)
from app.core.llm.rate_limit import estimate_input_tokens


@dataclass(slots=True)
class RecordedCall:
    """What a test can inspect after the fact."""

    messages: list[ChatMessage]
    tier: ModelTier
    response: LLMResponse


# A scripted reply is either literal text to return, or an exception to raise
# (for simulating a 429 or a provider outage at the call site).
ScriptedReply = str | Exception


class FakeLLM(LLMClient):
    """Returns scripted replies in order, then falls back to a default.

    Every call is recorded on ``calls`` for assertions. Passing an ``Exception``
    in the script raises it instead of returning -- handy for driving the 429
    backoff and failure paths without a real provider.
    """

    def __init__(
        self,
        replies: Sequence[ScriptedReply] | None = None,
        *,
        default: str = "{}",
        model: str = "fake",
    ) -> None:
        self._replies: deque[ScriptedReply] = deque(replies or [])
        self._default = default
        self._model = model
        self.calls: list[RecordedCall] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tier: ModelTier = ModelTier.SMALL,
        params: GenerationParams | None = None,
        on_usage: UsageCallback | None = None,
    ) -> LLMResponse:
        reply = self._replies.popleft() if self._replies else self._default
        if isinstance(reply, Exception):
            # Recorded implicitly is impossible (nothing to record), but callers
            # of the backoff layer expect the raise -- that is the point.
            raise reply

        prompt = "\n".join(m.content for m in messages)
        response = LLMResponse(
            text=reply,
            model=f"{self._model}-{tier.value}",
            tier=tier,
            input_tokens=estimate_input_tokens(prompt),
            output_tokens=estimate_input_tokens(reply),
            estimated=True,
        )
        self.calls.append(RecordedCall(messages=list(messages), tier=tier, response=response))
        if on_usage is not None:
            on_usage(response)
        return response

    # --- Convenience for readable test setup ---------------------------------

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last_prompt(self) -> str:
        """The concatenated content of the most recent call's messages."""
        if not self.calls:
            raise AssertionError("FakeLLM has not been called yet")
        return "\n".join(m.content for m in self.calls[-1].messages)

    def queue(self, *replies: ScriptedReply) -> None:
        """Append more scripted replies after construction."""
        self._replies.extend(replies)
