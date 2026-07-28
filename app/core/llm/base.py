"""The shared LLM interface every agent talks to.

This is the type contract, built *before* any agent (build-plan Section 2): the
one place that defines what "call the model" means. Both the real client
(``GeminiLLM``) and the test double (``FakeLLM``) implement ``LLMClient``, and
the structured-output and rate-limiting machinery is written against this
interface -- never against a concrete provider. That is what lets the whole
test suite run offline, for free, and deterministically: swap the
implementation, keep every code path.

Nothing here imports the provider SDK or the database. Keeping this module
dependency-free is deliberate -- it means an agent can depend on the interface
without dragging in ``google-genai`` or a live network.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass


class ModelTier(enum.StrEnum):
    """Which of the two models a call should use.

    Agents ask for a *tier*, not a model name (spec section 6.1): cheap,
    high-throughput work goes to SMALL; the heavy reasoning of Critic / Report /
    Chat goes to LARGE. The concrete model id behind each tier is a setting, so
    the mapping can change without touching a single agent.
    """

    SMALL = "small"
    LARGE = "large"


class Role(enum.StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One turn in a conversation, provider-neutral.

    Providers disagree on how to represent a system prompt (some open models,
    Gemma included, have no dedicated system role). We store the intent here and
    let each concrete client translate it, rather than baking one provider's
    shape into the interface.
    """

    role: Role
    content: str


def system(content: str) -> ChatMessage:
    return ChatMessage(Role.SYSTEM, content)


def user(content: str) -> ChatMessage:
    return ChatMessage(Role.USER, content)


def assistant(content: str) -> ChatMessage:
    return ChatMessage(Role.ASSISTANT, content)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """A single completion plus the metadata cost tracking needs.

    ``input_tokens`` / ``output_tokens`` come from the provider's own usage
    metadata when available (the real client) or are estimated (the fake). They
    flow straight into the ``token_usage`` table via the usage callback, which
    is the "cost-aware AI" talking point (spec section 13).
    """

    text: str
    model: str
    tier: ModelTier
    input_tokens: int
    output_tokens: int
    # True when the token counts are heuristic rather than provider-reported,
    # so a reader of token_usage knows not to trust them to the digit.
    estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# A callback the client invokes once per completed request, used to record token
# usage per agent per job. The client stays database-agnostic: it emits the
# response, and whatever wired the callback decides what to do with it (see
# app/core/llm/usage.py). Threaded through retries too, so a re-ask is costed
# honestly rather than hidden.
UsageCallback = Callable[[LLMResponse], None]


class LLMError(RuntimeError):
    """Base class for every failure originating in the LLM layer."""


class LLMConfigError(LLMError):
    """The client cannot be constructed -- e.g. no API key configured."""


class TransientLLMError(LLMError):
    """The provider failed in a way that a retry could have fixed, and did not.

    A 503, a gateway timeout, a connection reset, a request that outran its
    deadline. Distinguished from every other failure for one reason: **it is not
    a bug in this codebase**, and an agent that falls back needs to say so. Before
    this class existed, a provider blip reached the agents as a bare ``Exception``
    and was recorded as "fell back after an unexpected error" -- which sends
    whoever reads it looking for a defect that is not there.

    Raised only after the retry budget is spent, so seeing one means the provider
    was unavailable for the whole of it rather than for an instant.
    """


class RateLimitError(TransientLLMError):
    """The provider is rate-limiting us and backoff retries were exhausted.

    A specialisation of ``TransientLLMError``: being throttled is the one
    transient failure with a known cause and a known remedy (wait), which is why
    it keeps its own type and its own proactive defence in ``rate_limit.py``.
    """


class StructuredOutputError(LLMError):
    """The model never returned output matching the requested schema.

    Carries the last raw text and the underlying validation error so the failure
    is debuggable instead of a bare "invalid output".
    """

    def __init__(self, message: str, *, raw: str = "", attempts: int = 0) -> None:
        super().__init__(message)
        self.raw = raw
        self.attempts = attempts


@dataclass(frozen=True, slots=True)
class GenerationParams:
    """Knobs a caller may tune per request, with agent-friendly defaults.

    A low default temperature suits this project: almost every LLM job is a
    structured decision (which encoding? what is the target?), where we want the
    most probable answer, not a creative one.
    """

    temperature: float = 0.2
    max_output_tokens: int | None = None


class LLMClient(ABC):
    """The one method every agent uses, and the two clients implement."""

    @abstractmethod
    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tier: ModelTier = ModelTier.SMALL,
        params: GenerationParams | None = None,
        on_usage: UsageCallback | None = None,
    ) -> LLMResponse:
        """Send ``messages`` to the model at ``tier`` and return its reply.

        Implementations must call ``on_usage`` exactly once with the resulting
        ``LLMResponse`` before returning, so no request goes uncosted.
        """
        raise NotImplementedError
