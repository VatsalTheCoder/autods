"""Tests for FakeLLM itself -- the test double the rest of the suite leans on.

If the fake does not honour the interface faithfully, every test that uses it is
worthless, so its contract is pinned here directly.
"""

from __future__ import annotations

import pytest

from app.core.llm import FakeLLM, ModelTier, system, user


def test_returns_scripted_replies_in_order():
    llm = FakeLLM(["first", "second"])
    assert llm.complete([user("a")]).text == "first"
    assert llm.complete([user("b")]).text == "second"


def test_falls_back_to_default_when_script_exhausted():
    llm = FakeLLM(["only one"], default="DEFAULT")
    llm.complete([user("a")])
    assert llm.complete([user("b")]).text == "DEFAULT"


def test_raises_a_scripted_exception():
    boom = RuntimeError("provider down")
    llm = FakeLLM([boom])
    with pytest.raises(RuntimeError, match="provider down"):
        llm.complete([user("a")])


def test_records_every_call_for_inspection():
    llm = FakeLLM(["x"])
    llm.complete([system("be terse"), user("hello")], tier=ModelTier.LARGE)

    assert llm.call_count == 1
    call = llm.calls[0]
    assert call.tier is ModelTier.LARGE
    assert [m.content for m in call.messages] == ["be terse", "hello"]


def test_reports_estimated_token_counts_and_fires_usage_callback():
    seen = []
    llm = FakeLLM(["a reply"])

    response = llm.complete([user("a longish prompt here")], on_usage=seen.append)

    assert response.estimated is True
    assert response.input_tokens >= 1
    assert response.output_tokens >= 1
    assert seen == [response]


def test_tier_is_reflected_in_the_model_name():
    llm = FakeLLM(["x"], model="fake")
    assert llm.complete([user("a")], tier=ModelTier.SMALL).model == "fake-small"


def test_queue_appends_more_replies():
    llm = FakeLLM(["a"])
    llm.queue("b", "c")
    assert [llm.complete([user("x")]).text for _ in range(3)] == ["a", "b", "c"]
