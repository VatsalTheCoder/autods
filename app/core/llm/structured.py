"""Force an LLM reply into a validated Pydantic shape, or fail loudly.

This is the load-bearing half of the project's core rule -- *LLM decides, code
executes* (spec section 6.2). Agents never let the model touch data or emit
code; the model returns a small JSON decision, and this function is the gate
that guarantees the decision is well-formed before any deterministic code acts
on it.

The pattern, provider-neutral by design:

1. Ask the model, appending a compact schema description to the prompt.
2. Pull JSON out of the reply (open models love to wrap it in prose or a
   fenced code block).
3. Validate it against the caller's Pydantic model.
4. On failure, *re-ask* -- feeding the exact error back to the model, which is
   far more effective than a blind retry -- up to a configured budget.
5. If the budget runs out, raise ``StructuredOutputError`` with the last raw
   text attached, so the job fails with something debuggable (spec section 10).

It runs identically over ``GeminiLLM`` and ``FakeLLM``, which is the whole
point: the retry logic is tested for real against scripted malformed replies,
no network required. Caller-supplied Pydantic validators (e.g. "this column
name must exist in the dataset") are enforced here too -- an invented column is
just another validation error that triggers a re-ask, which is exactly how
Section 7 rejects hallucinated feature names.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from app.core.config import get_settings
from app.core.llm.base import (
    ChatMessage,
    GenerationParams,
    LLMClient,
    LLMResponse,
    ModelTier,
    StructuredOutputError,
    UsageCallback,
    assistant,
    user,
)

# Matches a ```json ... ``` (or bare ``` ... ```) fenced block. Open models wrap
# JSON in these constantly; stripping the fence first avoids a spurious parse
# failure that would waste a whole retry.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


@dataclass(slots=True)
class StructuredResult[T: BaseModel]:
    """The validated object plus what it took to get it.

    ``attempts`` and ``responses`` are exposed so a caller can log how many
    re-asks a flaky prompt needed -- a small but genuine observability win, and
    it makes the retry behaviour assertable in tests.
    """

    data: T
    responses: list[LLMResponse]

    @property
    def attempts(self) -> int:
        return len(self.responses)


def extract_json(text: str) -> str:
    """Return the JSON substring of a model reply, best-effort.

    Handles the three shapes open models produce in practice: a clean JSON
    document, a fenced block, or JSON buried in surrounding prose. Falls back to
    the span between the first ``{``/``[`` and its matching last bracket, which
    covers "Sure! Here is the JSON: {...}". Raises ``ValueError`` when there is
    no bracket at all, so the caller can re-ask rather than crash.
    """
    fenced = _FENCE.search(text)
    if fenced:
        return fenced.group(1).strip()

    stripped = text.strip()
    if stripped[:1] in "{[":
        return stripped

    # Prose around a JSON body: grab the outermost bracket pair.
    start = min(
        (i for i in (stripped.find("{"), stripped.find("[")) if i != -1),
        default=-1,
    )
    if start == -1:
        raise ValueError("no JSON object or array found in model output")
    opener = stripped[start]
    closer = "}" if opener == "{" else "]"
    end = stripped.rfind(closer)
    if end <= start:
        raise ValueError("unbalanced JSON brackets in model output")
    return stripped[start : end + 1]


def _schema_hint(model: type[BaseModel]) -> str:
    """A compact instruction telling the model exactly what shape to return.

    We send the JSON Schema rather than a hand-written example: it is always in
    sync with the Pydantic model, and open models follow an explicit schema far
    better than a vague "return JSON" plea. Kept terse to respect the input-TPM
    cap (spec section 10) -- this text is part of every structured prompt.
    """
    schema = json.dumps(model.model_json_schema(), separators=(",", ":"))
    return (
        "Respond with a single JSON object and nothing else -- no prose, no code "
        f"fence. It must validate against this JSON Schema:\n{schema}"
    )


def _correction(error: str, raw: str) -> ChatMessage:
    """The re-ask message: show the model its own bad output and the error."""
    return user(
        "Your previous response could not be parsed into the required shape.\n"
        f"Response was:\n{raw}\n\n"
        f"The problem:\n{error}\n\n"
        "Return corrected JSON only."
    )


def structured_complete[T: BaseModel](
    client: LLMClient,
    messages: Sequence[ChatMessage],
    response_model: type[T],
    *,
    tier: ModelTier = ModelTier.SMALL,
    params: GenerationParams | None = None,
    max_retries: int | None = None,
    on_usage: UsageCallback | None = None,
) -> StructuredResult[T]:
    """Get a ``response_model`` instance from the LLM, retrying on bad output.

    ``max_retries`` counts *re-asks* after the first attempt, defaulting to the
    configured ``llm_max_retries``. Every underlying call is costed through
    ``on_usage`` -- retries included -- so ``token_usage`` reflects the true
    spend of a prompt that needed three tries, not a flattering one.
    """
    if max_retries is None:
        max_retries = get_settings().llm_max_retries

    # Append the schema hint once; corrections are added per failed attempt.
    convo: list[ChatMessage] = [*messages, user(_schema_hint(response_model))]
    responses: list[LLMResponse] = []
    last_error = "unknown"
    last_raw = ""

    for _ in range(max_retries + 1):
        response = client.complete(convo, tier=tier, params=params, on_usage=on_usage)
        responses.append(response)
        last_raw = response.text
        try:
            payload = json.loads(extract_json(response.text))
            data = response_model.model_validate(payload)
        except (ValueError, ValidationError) as exc:
            last_error = str(exc)
            convo = [*convo, response_to_message(response), _correction(last_error, response.text)]
            continue
        return StructuredResult(data=data, responses=responses)

    raise StructuredOutputError(
        f"model did not produce valid {response_model.__name__} after "
        f"{len(responses)} attempts: {last_error}",
        raw=last_raw,
        attempts=len(responses),
    )


def response_to_message(response: LLMResponse) -> ChatMessage:
    """Fold a model reply back into the conversation as an assistant turn.

    So the re-ask sees genuine turn-taking (assistant said X, user objected)
    rather than two stacked user messages, which some models handle poorly.
    """
    return assistant(response.text)
